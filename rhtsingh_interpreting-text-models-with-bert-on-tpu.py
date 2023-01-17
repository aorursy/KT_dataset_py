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

import transformers
from transformers.modeling_bert import BertPreTrainedModel
from transformers import (
    BertTokenizer,
    BertModel,
    BertForSequenceClassification,
    BertConfig,
    AdamW,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
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
class DatasetRetriever(Dataset):
    def __init__(self, df, encoded):
        self.df = df
        self.encoded = encoded
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):   
        ids = self.encoded['input_ids'][index]
        type_ids = self.encoded['token_type_ids'][index]
        mask = self.encoded['attention_mask'][index]
        targets = self.df.iloc[index].label
        return {
            'ids':torch.tensor(ids),
            'token_type_ids':torch.tensor(type_ids),
            'mask':torch.tensor(mask),
            'targets':targets
        }
class BertUncased(BertPreTrainedModel):
    def __init__(self, config):
        config.output_hidden_states = True
        super(BertUncased, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(p=0.1)
        self.high_dropout = nn.Dropout(p=0.5)

        n_weights = config.num_hidden_layers + 1
        weights_init = torch.zeros(n_weights).float()
        weights_init.data[:-1] = -3
        self.layer_weights = torch.nn.Parameter(weights_init)
        self.init_weights()
      
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
    
    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)
        hidden_layers = outputs[2]
        last_hidden = outputs[0]
        cls_outputs = torch.stack(
            [self.dropout(layer[:, 0, :]) for layer in hidden_layers], dim=2
        )
        cls_output = (torch.softmax(self.layer_weights, dim=0) * cls_outputs).sum(-1)
        logits = self.classifier(cls_output)
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
        return "bert" in name
    
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
        if i % 20 == 0:
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
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 16
EPOCHS = 3
MAX_LEN = 96
# Scale learning rate to 8 TPU's
LR = 2e-5 * xm.xrt_world_size() 
METRICS_DEBUG = True
bert_model = "bert-base-multilingual-uncased"
NUM_CLASSES = 3

tokenizer = BertTokenizer.from_pretrained(bert_model)
# Train Validation Split
mask = np.random.rand(len(train)) < 0.95
train_df = train[mask]
valid_df = train[~mask]

train_text = train_df[['premise', 'hypothesis']].values.tolist()
train_encoded = tokenizer.batch_encode_plus(
    train_text,
    pad_to_max_length=True,
    max_length=MAX_LEN
)

valid_text = valid_df[['premise', 'hypothesis']].values.tolist()
valid_encoded = tokenizer.batch_encode_plus(
    valid_text,
    pad_to_max_length=True,
    max_length=MAX_LEN
)

train_df.reset_index(drop=True, inplace=True)
valid_df.reset_index(drop=True, inplace=True)
WRAPPED_MODEL = xmp.MpModelWrapper(BertUncased.from_pretrained(bert_model, num_labels=NUM_CLASSES))
train_dataset = DatasetRetriever(df=train_df, encoded=train_encoded)
valid_dataset = DatasetRetriever(df=valid_df, encoded=valid_encoded)
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
            xm.save(model.state_dict(), "model.bin")
            xm.master_print('Model Saved.')
            
    if METRICS_DEBUG:
      xm.master_print(met.metrics_report(), flush=True)
def _mp_fn(rank, flags):
    _run()

FLAGS={}
xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=8, start_method='fork')
!pip install captum
import matplotlib.pyplot as plt
import captum
from captum.attr import visualization as viz
from captum.attr import IntegratedGradients, LayerConductance, LayerIntegratedGradients
from captum.attr import configure_interpretable_embedding_layer, remove_interpretable_embedding_layer
device = xm.xla_device() if tpu else torch.deice("cpu")
model = WRAPPED_MODEL.to(device)
model.load_state_dict(torch.load("model.bin"))
model.eval()
model.zero_grad()
ref_token_id = tokenizer.pad_token_id # A token used for generating token reference
sep_token_id = tokenizer.sep_token_id # A token used as a separator between question and text and it is also added to the end of the text.
cls_token_id = tokenizer.cls_token_id # A token used for prepending to the concatenated question-text word sequence
def construct_input_ref_pair(text, ref_token_id, sep_token_id, cls_token_id):
    text_ids = tokenizer.encode(text, add_special_tokens=False)
    # construct input token ids
    input_ids = [cls_token_id] + text_ids + [sep_token_id]
    # construct reference token ids 
    ref_input_ids = [cls_token_id] + [ref_token_id] * len(text_ids) + [sep_token_id]
    return torch.tensor([input_ids], device=device), torch.tensor([ref_input_ids], device=device), len(text_ids)

def construct_input_ref_token_type_pair(input_ids, sep_ind=0):
    seq_len = input_ids.size(1)
    token_type_ids = torch.tensor([[0 if i <= sep_ind else 1 for i in range(seq_len)]], device=device)
    ref_token_type_ids = torch.zeros_like(token_type_ids, device=device)# * -1
    return token_type_ids, ref_token_type_ids

def construct_input_ref_pos_id_pair(input_ids):
    seq_length = input_ids.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
    # we could potentially also use random permutation with `torch.randperm(seq_length, device=device)`
    ref_position_ids = torch.zeros(seq_length, dtype=torch.long, device=device)
    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
    ref_position_ids = ref_position_ids.unsqueeze(0).expand_as(input_ids)
    return position_ids, ref_position_ids
    
def construct_attention_mask(input_ids):
    return torch.ones_like(input_ids)
def predict(inputs, token_type_ids, attention_mask):
    return model(inputs, attention_mask, token_type_ids)

def custom_forward_0(inputs, token_type_ids, attention_mask):
    preds = predict(inputs, token_type_ids, attention_mask)
    return torch.softmax(preds, dim = 1)[:, 0]

def custom_forward_1(inputs, token_type_ids, attention_mask):
    preds = predict(inputs, token_type_ids, attention_mask)
    return torch.softmax(preds, dim = 1)[:, 1]

def custom_forward_2(inputs, token_type_ids, attention_mask):
    preds = predict(inputs, token_type_ids, attention_mask)
    return torch.softmax(preds, dim = 1)[:, 2]

def save_act(module, inp, out):
  return saved_act

hook = model.bert.embeddings.register_forward_hook(save_act)
hook.remove()
def process(text, label):
    input_ids, ref_input_ids, sep_id = construct_input_ref_pair(text, ref_token_id, sep_token_id, cls_token_id)
    token_type_ids, ref_token_type_ids = construct_input_ref_token_type_pair(input_ids, sep_id)
    position_ids, ref_position_ids = construct_input_ref_pos_id_pair(input_ids)
    attention_mask = construct_attention_mask(input_ids)

    indices = input_ids[0].detach().tolist()
    all_tokens = tokenizer.convert_ids_to_tokens(indices)
    
    if label == 0:
        lig = LayerIntegratedGradients(custom_forward_0, model.bert.embeddings)
    elif label == 1:
        lig = LayerIntegratedGradients(custom_forward_1, model.bert.embeddings)
    elif label == 2:
        lig = LayerIntegratedGradients(custom_forward_2, model.bert.embeddings)
    
    attributions_main, delta_main = lig.attribute(inputs=input_ids,
                                                  baselines=ref_input_ids,
                                                  n_steps = 150,
                                                  additional_forward_args=(token_type_ids, attention_mask),
                                                  return_convergence_delta=True)
    
    score = predict(input_ids, token_type_ids, attention_mask)
    attributions_main = attributions_main.cpu()
    delta_main = delta_main.cpu()
    score = score.cpu()
    add_attributions_to_visualizer(attributions_main, delta_main, text, score, label, all_tokens)
    
def add_attributions_to_visualizer(attributions, delta, text, score, label, all_tokens):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    attributions = attributions.cpu()

    score_vis.append(
        viz.VisualizationDataRecord(
            attributions,
            torch.softmax(score, dim = 1)[0][label],
            torch.argmax(torch.softmax(score, dim = 1)[0]),
            label,
            text,
            attributions.sum(),
            all_tokens,
            delta
        )
    ) 
score_vis = []
for i, text in enumerate(train_text[:2]):
    # print(text)
    text = " ".join(text)
    label = train_df.iloc[i].label
    process(text, label)
viz.visualize_text(score_vis)
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
# test_preds = []

# for i, data in tqdm(enumerate(test_data_loader), total=len(test_data_loader)):
#     ids = data["ids"]
#     mask = data["mask"]
#     ids = ids.to(device, dtype=torch.long)
#     mask = mask.to(device, dtype=torch.long)
#     outputs = model(
#         input_ids = ids,
#         attention_mask = mask,
#     )
#     outputs_np = outputs.cpu().detach().numpy().tolist()
#     test_preds.extend(outputs_np)  
    
# test_preds = torch.FloatTensor(test_preds)
# top1_prob, top1_label = torch.topk(test_preds, 1)
# y = top1_label.cpu().detach().numpy()
# sample_submission.prediction = y
# sample_submission.to_csv('submission.csv', index=False)