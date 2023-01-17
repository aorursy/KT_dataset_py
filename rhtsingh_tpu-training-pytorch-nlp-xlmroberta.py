import tensorflow as tf

try:

   tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  

   print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])

except ValueError:

   tpu = None

if tpu:

   tf.config.experimental_connect_to_cluster(tpu)

   tf.tpu.experimental.initialize_tpu_system(tpu)

   strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

   strategy = tf.distribute.get_strategy()
!pip install nlp

!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py

!python pytorch-xla-env-setup.py --version nightly  --apt-packages libomp5 libopenblas-dev
%%time

%autosave 60



import os

os.environ['XLA_USE_BF16'] = "1"

os.environ['XLA_TENSOR_ALLOCATOR_MAXSIZE'] = '100000000'



import gc

gc.enable()

import time



import numpy as np

import pandas as pd

from tqdm import tqdm 



import nlp

import transformers

from transformers import (AdamW, 

                          XLMRobertaTokenizer, 

                          XLMRobertaModel, 

                          get_cosine_schedule_with_warmup)



import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset

from torch.utils.data.distributed import DistributedSampler



import torch_xla

import torch_xla.core.xla_model as xm

import torch_xla.debug.metrics as met

import torch_xla.distributed.parallel_loader as pl

import torch_xla.distributed.xla_multiprocessing as xmp

import torch_xla.utils.serialization as xser

import torch_xla.version as xv



import warnings

warnings.filterwarnings("ignore")



print('PYTORCH:', xv.__torch_gitrev__)

print('XLA:', xv.__xla_gitrev__)
train = pd.read_csv('../input/contradictory-my-dear-watson/train.csv')

test = pd.read_csv('../input/contradictory-my-dear-watson/test.csv')

sample_submission = pd.read_csv('../input/contradictory-my-dear-watson/sample_submission.csv')
TRAIN_BATCH_SIZE = 16

VALID_BATCH_SIZE = 16

EPOCHS = 4

MAX_LEN = 80

# Scale learning rate to 8 TPU's

LR = 2e-5 * xm.xrt_world_size() 

METRICS_DEBUG = True

tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')
# mnli data

mnli = nlp.load_dataset(path='glue', name='mnli', split='train[:5%]')



# xnli data

xnli = nlp.load_dataset(path='xnli')

xnli = nlp.concatenate_datasets([xnli['test'], xnli['validation']])



# snli data

snli = nlp.load_dataset(path='snli', split='train[:5%]')
print("#"*25)

print("  MNLI"); print("#"*25)

print("Shape: ", mnli.shape)

print("Num of Samples: ", mnli.num_rows)

print("Num of Columns: ", mnli.num_columns)

print("Column Names: ", mnli.column_names)

print("Features: ", mnli.features)

print("Num of Classes: ", mnli.features['label'].num_classes)

print("Split: ", mnli.split)

print("Description: ", mnli.description)

print(f"Labels Count - 0's:{len(mnli.filter(lambda x: x['label']==0))}, 1's:{len(mnli.filter(lambda x: x['label']==1))}, 2's: 0's:{len(mnli.filter(lambda x: x['label']==2))}")

print()

print("#"*25)

print("  XNLI"); print("#"*25)

print("Shape: ", xnli.shape)

print("Num of Samples: ", xnli.num_rows)

print("Num of Columns: ", xnli.num_columns)

print("Column Names: ", xnli.column_names)

print("Features: ", xnli.features)

print("Split: ", xnli.split)

print("Description: ", xnli.description)

print(f"Labels Count - 0's:{len(xnli.filter(lambda x: x['label']==0))}, 1's:{len(xnli.filter(lambda x: x['label']==1))}, 2's: 0's:{len(xnli.filter(lambda x: x['label']==2))}")

print()

print("#"*25)

print("  SNLI"); print("#"*25)

print("Shape: ", snli.shape)

print("Num of Samples: ", snli.num_rows)

print("Num of Columns: ", snli.num_columns)

print("Column Names: ", snli.column_names)

print("Features: ", snli.features)

print("Num of Classes: ", snli.features['label'].num_classes)

print("Split: ", snli.split)

print("Description: ", snli.description)

print(f"Labels Count - 0's:{len(snli.filter(lambda x: x['label']==0))}, 1's:{len(snli.filter(lambda x: x['label']==1))}, 2's: 0's:{len(snli.filter(lambda x: x['label']==2))}")
# encoding

def convert_to_features(batch):

    input_pairs = list(zip(batch['premise'], batch['hypothesis']))

    encodings = tokenizer.batch_encode_plus(input_pairs, 

                                            add_special_tokens=True, 

                                            pad_to_max_length=True, 

                                            max_length=MAX_LEN, 

                                            truncation=True, 

                                            return_attention_mask=True, 

                                            return_token_type_ids=True)

    return encodings
# function to preprocess special structure of xnli

def preprocess_xnli(example):

    premise_output = []

    hypothesis_output = []

    label_output = []

    for prem, hyp, lab in zip(example['premise'],  example['hypothesis'], example["label"]):

        label = lab

        langs = hyp['language']

        translations = hyp['translation']

        hypothesis = {k: v for k, v in zip(langs, translations)}

        for lang in prem:

            if lang in hypothesis:

                premise_output += [prem[lang]]

                hypothesis_output += [hypothesis[lang]]

                label_output += [label]

    return {'premise':premise_output, 'hypothesis':hypothesis_output, 'label':label_output}
# encode mnli and convert to torch tensor

mnli_encoded = mnli.map(convert_to_features, batched=True, remove_columns=['idx', 'premise', 'hypothesis'])

mnli_encoded.set_format("torch", columns=['attention_mask', 'input_ids', 'token_type_ids', 'label'])
# preprocess xnli, encode and convert to torch tensor

xnli_processed = xnli.map(preprocess_xnli, batched=True)

xnli_encoded = xnli_processed.map(convert_to_features, batched=True, remove_columns=['premise', 'hypothesis'])

xnli_encoded.set_format("torch", columns=['attention_mask', 'input_ids', 'token_type_ids', 'label']) 
# encode snli and convert to torch tensor

snli_encoded = snli.map(convert_to_features, batched=True, remove_columns=['premise', 'hypothesis'])

snli_encoded.set_format("torch", columns=['attention_mask', 'input_ids', 'token_type_ids', 'label']) 
print(mnli_encoded.column_names)

print(snli_encoded.column_names)

print(xnli_encoded.column_names)



print(mnli_encoded.num_rows)

print(snli_encoded.num_rows)

print(xnli_encoded.num_rows)
train_dataset = nlp.load_dataset('csv', data_files=['../input/contradictory-my-dear-watson/train.csv'])['train']



print(train_dataset.num_rows)

print(train_dataset.column_names)

drop_columns = train_dataset.column_names[:-1]



encoded_train_dataset = train_dataset.map(convert_to_features, batched=True, remove_columns=drop_columns)

encoded_train_dataset.set_format("torch", columns=['attention_mask', 'input_ids', 'token_type_ids', 'label']) 

print(encoded_train_dataset.num_rows)

print(encoded_train_dataset.column_names)
train_dataset = nlp.concatenate_datasets([mnli_encoded, 

                                          xnli_encoded, 

                                          snli_encoded,

                                          encoded_train_dataset

                                         ])



print(train_dataset.num_rows)

print(train_dataset.column_names)
train_dataset.cleanup_cache_files()

del mnli, mnli_encoded

del xnli, xnli_encoded, xnli_processed

del snli, snli_encoded

gc.collect()
class DatasetRetriever(Dataset):

    def __init__(self, dataset:nlp.arrow_dataset.Dataset):

        self.dataset = dataset

        self.ids = self.dataset['input_ids']

        self.mask = self.dataset['attention_mask']

        self.type_ids = self.dataset['token_type_ids']

        self.targets = self.dataset["label"]

        

    def __len__(self):

        return self.dataset.num_rows

    

    def __getitem__(self, index):   

        ids = self.ids[index]

        mask = self.mask[index]

        type_ids = self.type_ids[index]

        targets = self.targets[index]

        return {

            'ids':torch.tensor(ids),

            'mask':torch.tensor(mask),

            'type_ids':torch.tensor(type_ids),

            'targets':targets

        }
class XLMRoberta(nn.Module):

    def __init__(self, num_labels, multisample):

        super(XLMRoberta, self).__init__()

        output_hidden_states = False

        self.num_labels = num_labels

        self.multisample= multisample

        self.roberta = XLMRobertaModel.from_pretrained("xlm-roberta-large", 

                                                       output_hidden_states=output_hidden_states, 

                                                       num_labels=1)

        self.layer_norm = nn.LayerNorm(1024*2)

        self.dropout = nn.Dropout(p=0.2)

        self.high_dropout = nn.Dropout(p=0.5)        

        self.classifier = nn.Linear(1024*2, self.num_labels)

    

    def forward(self,

        input_ids=None,

        attention_mask=None,

        token_type_ids=None,

        position_ids=None,

        head_mask=None,

        inputs_embeds=None):

        outputs = self.roberta(input_ids,

                               attention_mask=attention_mask,

                               token_type_ids=token_type_ids,

                               position_ids=position_ids,

                               head_mask=head_mask,

                               inputs_embeds=inputs_embeds)

        average_pool = torch.mean(outputs[0], 1)

        max_pool, _ = torch.max(outputs[0], 1)

        concatenate_layer = torch.cat((average_pool, max_pool), 1)

        normalization = self.layer_norm(concatenate_layer)

        if self.multisample:

            # Multisample Dropout

            logits = torch.mean(

                torch.stack(

                    [self.classifier(self.dropout(normalization)) for _ in range(5)],

                    dim=0,

                ),

                dim=0,

            )

        else:

            logits = self.dropout(normalization)

            logits = self.classifier(logits)       

        outputs = logits

        return outputs  
class AverageMeter(object):

    def __init__(self, name, fmt=':f'):

        self.name = name

        self.fmt = fmt

        self.reset()



    def reset(self):

        self.val = 0

        self.avg = 0

        self.sum = 0

        self.count = 0



    def update(self, val, n=1):

        self.val = val

        self.sum += val * n

        self.count += n

        self.avg = self.sum / self.count



    def __str__(self):

        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'

        return fmtstr.format(**self.__dict__)



class ProgressMeter(object):

    def __init__(self, num_batches, meters, prefix=""):

        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)

        self.meters = meters

        self.prefix = prefix



    def display(self, batch):

        entries = [self.prefix + self.batch_fmtstr.format(batch)]

        entries += [str(meter) for meter in self.meters]

        print('\t'.join(entries))



    def _get_batch_fmtstr(self, num_batches):

        num_digits = len(str(num_batches // 1))

        fmt = '{:' + str(num_digits) + 'd}'

        return '[' + fmt + '/' + fmt.format(num_batches) + ']'



def accuracy(output, target, topk=(1,)):

    with torch.no_grad():

        maxk = max(topk)

        batch_size = target.size(0)



        _, pred = output.topk(maxk, 1, True, True)

        pred = pred.t()

        correct = pred.eq(target.view(1, -1).expand_as(pred))



        res = []

        for k in topk:

            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)

            res.append(correct_k.mul_(100.0 / batch_size))

        return res
def get_model_optimizer(model):

    # Differential Learning Rate

    def is_backbone(name):

        return "roberta" in name

    

    optimizer_grouped_parameters = [

       {'params': [param for name, param in model.named_parameters() if is_backbone(name)], 'lr': LR},

       {'params': [param for name, param in model.named_parameters() if not is_backbone(name)], 'lr': 1e-3} 

    ]

    

    optimizer = AdamW(

        optimizer_grouped_parameters, lr=LR, weight_decay=1e-2

    )

    

    return optimizer
def loss_fn(outputs, targets):

    return nn.CrossEntropyLoss()(outputs, targets)
def train_loop_fn(train_loader, model, optimizer, device, scheduler, epoch=None):

    # Train

    batch_time = AverageMeter('Time', ':6.3f')

    data_time = AverageMeter('Data', ':6.3f')

    losses = AverageMeter('Loss', ':.4e')

    top1 = AverageMeter('Acc@1', ':6.2f')

    progress = ProgressMeter(

        len(train_loader),

        [batch_time, data_time, losses, top1],

        prefix="[xla:{}]Train:  Epoch: [{}]".format(xm.get_ordinal(), epoch)

    )

    model.train()

    end = time.time()

    for i, data in enumerate(train_loader):

        data_time.update(time.time()-end)

        ids, mask, type_ids, targets = data["input_ids"], data["attention_mask"], data['token_type_ids'], data["label"]

        ids = ids.to(device, dtype=torch.long)

        mask = mask.to(device, dtype=torch.long)

        type_ids = type_ids.to(device, dtype=torch.long)

        targets = targets.to(device, dtype=torch.float)

        optimizer.zero_grad()

        outputs = model(

            input_ids = ids,

            attention_mask = mask,

            token_type_ids = type_ids

        )

        loss = loss_fn(outputs, targets)

        loss.backward()

        xm.optimizer_step(optimizer)

        loss = loss_fn(outputs, targets)

        acc1= accuracy(outputs, targets, topk=(1,))

        losses.update(loss.item(), ids.size(0))

        top1.update(acc1[0].item(), ids.size(0))

        scheduler.step()

        batch_time.update(time.time() - end)

        end = time.time()

        if i % 50 == 0:

            progress.display(i)

    del loss

    del outputs

    del ids

    del mask

    del targets

    gc.collect()
def eval_loop_fn(validation_loader, model, device):

    #Validation

    model.eval()

    batch_time = AverageMeter('Time', ':6.3f')

    losses = AverageMeter('Loss', ':.4e')

    top1 = AverageMeter('Acc@1', ':6.2f')

    learning_rate = AverageMeter('LR',':2.8f')

    progress = ProgressMeter(

        len(validation_loader),

        [batch_time, losses, top1],

        prefix='[xla:{}]Validation: '.format(xm.get_ordinal()))

    with torch.no_grad():

        end = time.time()

        for i, data in enumerate(validation_loader):

            ids, mask, type_ids, targets = data["input_ids"], data["attention_mask"], data['token_type_ids'], data["label"]

            ids = ids.to(device, dtype=torch.long)

            mask = mask.to(device, dtype=torch.long)

            type_ids = type_ids.to(device, dtype=torch.long)

            targets = targets.to(device, dtype=torch.float)

            outputs = model(

                input_ids = ids,

                attention_mask = mask,

                token_type_ids = type_ids

            )

            loss = loss_fn(outputs, targets)

            acc1= accuracy(outputs, targets, topk=(1,))

            losses.update(loss.item(), ids.size(0))

            top1.update(acc1[0].item(), ids.size(0))

            batch_time.update(time.time() - end)

            end = time.time()

            if i % 50 == 0:

                progress.display(i)

    del loss

    del outputs

    del ids

    del mask

    del targets

    gc.collect()
WRAPPED_MODEL = xmp.MpModelWrapper(XLMRoberta(num_labels=3, multisample=False))



dataset = train_dataset.train_test_split(test_size=0.1)

train_dataset = dataset['train']

valid_dataset = dataset['test']

train_dataset.set_format("torch", columns=['attention_mask', 'input_ids', 'token_type_ids', 'label']) 

valid_dataset.set_format("torch", columns=['attention_mask', 'input_ids', 'token_type_ids', 'label']) 
def _run():

    xm.master_print('Starting Run ...')

    train_sampler = DistributedSampler(

        train_dataset,

        num_replicas=xm.xrt_world_size(),

        rank=xm.get_ordinal(),

        shuffle=False

    )

    

    train_data_loader = DataLoader(

        train_dataset,

        batch_size=TRAIN_BATCH_SIZE,

        sampler=train_sampler,

        drop_last=True,

        num_workers=0

    )

    xm.master_print('Train Loader Created.')

    

    valid_sampler = DistributedSampler(

        valid_dataset,

        num_replicas=xm.xrt_world_size(),

        rank=xm.get_ordinal(),

        shuffle=False

    )

    

    valid_data_loader = DataLoader(

        valid_dataset,

        batch_size=VALID_BATCH_SIZE,

        sampler=valid_sampler,

        drop_last=True,

        num_workers=0

    )

    xm.master_print('Valid Loader Created.')

    

    num_train_steps = int(len(train_dataset) / TRAIN_BATCH_SIZE / xm.xrt_world_size())

    device = xm.xla_device()

    model = WRAPPED_MODEL.to(device)

    xm.master_print('Done Model Loading.')

    optimizer = get_model_optimizer(model)

    scheduler = get_cosine_schedule_with_warmup(

        optimizer,

        num_warmup_steps = 0,

        num_training_steps = num_train_steps * EPOCHS

    )

    xm.master_print(f'Num Train Steps= {num_train_steps}, XRT World Size= {xm.xrt_world_size()}.')

    

    for epoch in range(EPOCHS):

        para_loader = pl.ParallelLoader(train_data_loader, [device])

        xm.master_print('Parallel Loader Created. Training ...')

        train_loop_fn(para_loader.per_device_loader(device),

                      model,  

                      optimizer, 

                      device, 

                      scheduler, 

                      epoch

                     )

        

        xm.master_print("Finished training epoch {}".format(epoch))

            

        para_loader = pl.ParallelLoader(valid_data_loader, [device])

        xm.master_print('Parallel Loader Created. Validating ...')

        eval_loop_fn(para_loader.per_device_loader(device), 

                     model,  

                     device

                    )

        

        # Serialized and Memory Reduced Model Saving

        if epoch == EPOCHS-1:

            xm.master_print('Saving Model ..')

            xm.save(model.state_dict(), "model.bin")

            xm.master_print('Model Saved.')

            

    if METRICS_DEBUG:

      xm.master_print(met.metrics_report(), flush=True)
def _mp_fn(rank, flags):

    # torch.set_default_tensor_type('torch.FloatTensor')

    _run()



FLAGS={}

xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=8, start_method='fork')