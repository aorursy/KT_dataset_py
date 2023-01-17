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
!pip install git+https://github.com/ssut/py-googletrans.git

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

from googletrans import Translator

from dask import bag, diagnostics



import transformers

from transformers import (AdamW, 

                          DistilBertTokenizer, 

                          DistilBertModel, 

                          DistilBertTokenizerFast,                          

                          get_cosine_schedule_with_warmup)

from tokenizers import BertWordPieceTokenizer



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
def translate(words):

    translator = Translator()

    decoded = translator.translate(words, dest='en').text

    return decoded



other_langs = train.loc[train.lang_abv != "en"].copy()



#TODO: use a dask dataframe instead of bags

premise_bag = bag.from_sequence(other_langs.premise.tolist()).map(translate)

hypo_bag =  bag.from_sequence(other_langs.hypothesis.tolist()).map(translate)

with diagnostics.ProgressBar():

    premises = premise_bag.compute()

    hypos = hypo_bag.compute()

    

    

other_langs[['premise', 'hypothesis']] = list(zip(premises, hypos))

train = train.append(other_langs)

train.shape
class DatasetRetriever(Dataset):

    def __init__(self, df, ids, mask):

        self.df = df

        self.ids = ids

        self.mask = mask

        

    def __len__(self):

        return len(self.df)

    

    def __getitem__(self, index):   

        ids = self.ids[index]

        mask = self.mask[index]

        targets = self.df.iloc[index].label

        return {

            'ids':torch.tensor(ids),

            'mask':torch.tensor(mask),

            'targets':targets

        }
class DistillBERT(nn.Module):

    def __init__(self, num_labels, multisample):

        super(DistillBERT, self).__init__()

        output_hidden_states = True

        self.num_labels = num_labels

        self.multisample= multisample

        self.distillbert = DistilBertModel.from_pretrained("distilbert-base-multilingual-cased", 

                                                           output_hidden_states=output_hidden_states,

                                                           num_labels=1)

        self.layer_norm = nn.LayerNorm(768*2)

        self.dropout = nn.Dropout(p=0.2)

        self.high_dropout = nn.Dropout(p=0.5)        

        self.classifier = nn.Linear(768*2, self.num_labels)

    

    def forward(self,

        input_ids=None,

        attention_mask=None,

        head_mask=None,

        inputs_embeds=None):

        outputs = self.distillbert(input_ids,

                                   attention_mask=attention_mask,

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

        outputs = F.log_softmax(logits, dim=1)

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

        return "distillbert" in name

    

    optimizer_grouped_parameters = [

       {'params': [param for name, param in model.named_parameters() if is_backbone(name)], 'lr': LR},

       {'params': [param for name, param in model.named_parameters() if not is_backbone(name)], 'lr': 1e-3} 

    ]

    

    optimizer = AdamW(

        optimizer_grouped_parameters, lr=LR, weight_decay=1e-2

    )

    

    return optimizer
def loss_fn(outputs, targets):

    return nn.NLLLoss()(outputs, targets)
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

        ids = data["ids"]

        mask = data["mask"]

        targets = data["targets"]

        ids = ids.to(device, dtype=torch.long)

        mask = mask.to(device, dtype=torch.long)

        targets = targets.to(device, dtype=torch.float)

        optimizer.zero_grad()

        outputs = model(

            input_ids = ids,

            attention_mask = mask

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

        if i % 30 == 0:

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

            ids = data["ids"]

            mask = data["mask"]

            targets = data["targets"]

            ids = ids.to(device, dtype=torch.long)

            mask = mask.to(device, dtype=torch.long)

            targets = targets.to(device, dtype=torch.float)

            outputs = model(

                input_ids = ids,

                attention_mask = mask

            )

            loss = loss_fn(outputs, targets)

            acc1= accuracy(outputs, targets, topk=(1,))

            losses.update(loss.item(), ids.size(0))

            top1.update(acc1[0].item(), ids.size(0))

            batch_time.update(time.time() - end)

            end = time.time()

            if i % 10 == 0:

                progress.display(i)

    del loss

    del outputs

    del ids

    del mask

    del targets

    gc.collect()
def fast_encode(df, fast_tokenizer):

    fast_tokenizer.enable_truncation(max_length=MAX_LEN)

    fast_tokenizer.enable_padding(max_length=MAX_LEN)

    

    text = list(zip(df.premise, df.hypothesis))

    encoded = fast_tokenizer.encode_batch(

        text

    )

    

    all_ids = []

    all_masks = []

    all_ids.extend([enc.ids for enc in encoded])

    all_masks.extend([enc.attention_mask for enc in encoded])

    

    return np.array(all_ids), np.array(all_masks)
TRAIN_BATCH_SIZE = 16

VALID_BATCH_SIZE = 16

EPOCHS = 40

MAX_LEN = 80

# Scale learning rate to 8 TPU's

LR = 2e-5 * xm.xrt_world_size() 

METRICS_DEBUG = True



WRAPPED_MODEL = xmp.MpModelWrapper(DistillBERT(num_labels=3, multisample=False))

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')

# Save the loaded tokenizer locally

tokenizer.save_pretrained('.')

# Reload it with the huggingface tokenizers library

fast_tokenizer = BertWordPieceTokenizer('vocab.txt', lowercase=False)



# Train Validation Split

mask = np.random.rand(len(train)) < 0.95

train_df = train[mask]

valid_df = train[~mask]



train_ids, train_mask = fast_encode(train_df, fast_tokenizer)

valid_ids, valid_mask = fast_encode(valid_df, fast_tokenizer)



train_dataset = DatasetRetriever(df=train_df, ids=train_ids, mask=train_mask)

valid_dataset = DatasetRetriever(df=valid_df, ids=valid_ids, mask=valid_mask)
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

    

    num_train_steps = int(len(train_df) / TRAIN_BATCH_SIZE / xm.xrt_world_size())

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

            xser.save(model.state_dict(), f"model.bin", master_only=True)

            xm.master_print('Model Saved.')

            

    if METRICS_DEBUG:

      xm.master_print(met.metrics_report(), flush=True)
def _mp_fn(rank, flags):

    # torch.set_default_tensor_type('torch.FloatTensor')

    _run()



FLAGS={}

xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=8, start_method='fork')
class TestDatasetRetriever(Dataset):

    def __init__(self, df, ids, mask):

        self.df = df

        self.ids = ids

        self.mask = mask

        

    def __len__(self):

        return len(self.df)

    

    def __getitem__(self, index):   

        ids = self.ids[index]

        mask = self.mask[index]

        return {

            'ids':torch.tensor(ids),

            'mask':torch.tensor(mask)

        }
TEST_BATCH_SIZE = 32



test_ids, test_mask = fast_encode(test, fast_tokenizer)



test_dataset = TestDatasetRetriever(test, test_ids, test_mask)



test_data_loader = DataLoader(

    test_dataset, 

    batch_size=TEST_BATCH_SIZE,

    drop_last=False,

    num_workers=4,

    shuffle=False

)



# Load Serialized Model

device = xm.xla_device()

model = WRAPPED_MODEL.to(device).eval()

model.load_state_dict(xser.load("model.bin"))
test_preds = []



for i, data in tqdm(enumerate(test_data_loader), total=len(test_data_loader)):

    ids = data["ids"]

    mask = data["mask"]

    ids = ids.to(device, dtype=torch.long)

    mask = mask.to(device, dtype=torch.long)

    outputs = model(

        input_ids = ids,

        attention_mask = mask,

    )

    outputs_np = outputs.cpu().detach().numpy().tolist()

    test_preds.extend(outputs_np)  

    

test_preds = torch.FloatTensor(test_preds)

top1_prob, top1_label = torch.topk(test_preds, 1)

y = top1_label.cpu().detach().numpy()

sample_submission.prediction = y

sample_submission.to_csv('submission.csv', index=False)