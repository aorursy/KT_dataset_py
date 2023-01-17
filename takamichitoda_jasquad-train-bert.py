!yes | pip install transformers==2.10.0



!apt install aptitude -y

!aptitude install mecab libmecab-dev mecab-ipadic-utf8 git make curl xz-utils file -y

!pip install mecab-python3==0.996.6rc2



!pip install unidic-lite





!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py

!python pytorch-xla-env-setup.py --apt-packages libomp5 libopenblas-dev
import pandas as pd



import MeCab

import tqdm

import pickle



import torch

import torch.nn as nn



from transformers import *

import tokenizers





import torch.utils.data



from matplotlib import pyplot as plt



import torch_xla.core.xla_model as xm

import torch_xla.distributed.parallel_loader as pl

import torch_xla.distributed.xla_multiprocessing as xmp
INPUT = "/kaggle/input/squad-japanese/"

MAX_LEN = 512



TOKENIZER = BertJapaneseTokenizer.from_pretrained("bert-base-japanese")

TOKENIZER.save_pretrained("./")

TOKENIZER = tokenizers.BertWordPieceTokenizer("./vocab.txt", lowercase=True)



BATCH_SIZE = 16

WARM_UP_RATIO = 0

EPOCHS = 3

LEARNING_RATE = 1e-6
class JaSQuADDataset:

    def __init__(self, data):

        self.data = data

        

    def __len__(self):

        return len(self.data)



    def __getitem__(self, item):

        data = self.data[item]

        return {

            'ids': torch.tensor(data["ids"], dtype=torch.long),

            'mask': torch.tensor(data["mask"], dtype=torch.long),

            'token_type_ids': torch.tensor(data["token_type_ids"], dtype=torch.long),

            'targets_start': torch.tensor(data["targets_start"], dtype=torch.long),

            'targets_end': torch.tensor(data["targets_end"], dtype=torch.long),

            'offsets': torch.tensor(data["offsets"], dtype=torch.long),

            'uuid': data["uuid"]

        }
with open("/kaggle/input/jasquad-make-tokeids/squad_train_data.pkl", "rb") as f:

    train_data = pickle.load(f)

    

with open("/kaggle/input/jasquad-make-tokeids/squad_valid_data.pkl", "rb") as f:

    valid_data = pickle.load(f)

    

train_dataset = JaSQuADDataset(train_data)

valid_dataset = JaSQuADDataset(valid_data)
class JaSQuADBert(nn.Module):

    def __init__(self):

        super(JaSQuADBert, self).__init__()



        self.bert = BertModel.from_pretrained("bert-base-japanese-whole-word-masking")

        self.qa_outputs = nn.Linear(768, 2)



    

    def forward(self, ids, mask, token_type_ids):

        out, _ = self.bert(

            ids,

            attention_mask=mask,

            token_type_ids=token_type_ids

        )

        logits = self.qa_outputs(out)

        

        start_logits, end_logits = logits.split(1, dim=-1)



        start_logits = start_logits.squeeze(-1)

        end_logits = end_logits.squeeze(-1)



        return start_logits, end_logits

    

    

def loss_fn(start_logits, end_logits, start_positions, end_positions):

    loss_fct = nn.CrossEntropyLoss()

    start_loss = loss_fct(start_logits, start_positions)

    end_loss = loss_fct(end_logits, end_positions)

    

    total_loss = start_loss + end_loss



    return total_loss
def get_predict_text(ids, outputs_start, outputs_end):

    lst = []

    for idx, (st, ed) in enumerate(zip(outputs_start.argmax(1), outputs_end.argmax(1))):

        if st > ed:

            st, ed = 0, -1

        pred_text = TOKENIZER.decode(ids[idx][st:ed].tolist())

        lst.append(pred_text)

    return lst



def reduce_fn(vals):

    return sum(vals) / len(vals)



class AverageMeter:

    def __init__(self):

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
def train_fn(data_loader, model, optimizer, device, scheduler=None):

    model.train()

    losses = AverageMeter()

    for bi, d in enumerate(data_loader):    

        ids = d["ids"].to(device, dtype=torch.long)

        mask = d["mask"].to(device, dtype=torch.long)

        token_type_ids = d["token_type_ids"].to(device, dtype=torch.long)

        start_positions = d["targets_start"].to(device, dtype=torch.long)

        end_positions = d["targets_end"].to(device, dtype=torch.long)



        model.zero_grad()

        outputs_start, outputs_end = model(ids, mask, token_type_ids)

        loss = loss_fn(outputs_start, outputs_end, start_positions, end_positions)

        loss.backward()

        xm.optimizer_step(optimizer)

        scheduler.step()

        #if bi % 1 == 0:

        #print(xm.xla_device(), bi, loss)

    

        losses.update(loss.item(), ids.size(0))



    return losses.avg



        

def eval_fn(data_loader, model, device):

    model.eval()

    losses = AverageMeter()

    pred_texts, uuids = [], []

    with torch.no_grad():

        for bi, d in enumerate(data_loader):

            ids = d["ids"].to(device, dtype=torch.long)

            mask = d["mask"].to(device, dtype=torch.long)

            token_type_ids = d["token_type_ids"].to(device, dtype=torch.long)

            start_positions = d["targets_start"].to(device, dtype=torch.long)

            end_positions = d["targets_end"].to(device, dtype=torch.long)



            outputs_start, outputs_end = model(ids, mask, token_type_ids)

            loss = loss_fn(outputs_start, outputs_end, start_positions, end_positions)



            losses.update(loss.item(), ids.size(0))

            uuids += d["uuid"]

            

            pred_texts += get_predict_text(ids, outputs_start, outputs_end)

    

    pd.DataFrame(zip(uuids, pred_texts), columns=["uuid", "predict text"]).to_csv(f"result_{xm.xla_device()}.csv", index=None)



    return losses.avg
def run():

    device = xm.xla_device()

    model = MX.to(device)



    # set train dataloder    

    train_sampler = torch.utils.data.distributed.DistributedSampler(

        train_dataset,

        num_replicas=xm.xrt_world_size(),

        rank=xm.get_ordinal(),

        shuffle=True

    )



    train_data_loader = torch.utils.data.DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        sampler=train_sampler,

        drop_last=True,

        num_workers=2

    )



    # set valid dataloder

    valid_sampler = torch.utils.data.distributed.DistributedSampler(

        valid_dataset,

        num_replicas=xm.xrt_world_size(),

        rank=xm.get_ordinal(),

        shuffle=False

    )

    valid_data_loader = torch.utils.data.DataLoader(

        valid_dataset,

        batch_size=BATCH_SIZE,

        sampler=valid_sampler,

        drop_last=False,

        num_workers=1

    )



    # optimizer setting

    param_optimizer = list(model.named_parameters())

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    optimizer_parameters = [

        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},

        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},

    ]



    optimizer = AdamW(

        optimizer_parameters,

        lr=LEARNING_RATE * xm.xrt_world_size(),

    )



    # scheduler setting

    num_train_steps = int(

        len(train_data) / BATCH_SIZE / xm.xrt_world_size() * EPOCHS

    )

    n_warmup = int(num_train_steps * WARM_UP_RATIO)



    scheduler = get_cosine_schedule_with_warmup(

        optimizer,

        num_warmup_steps=n_warmup,

        num_training_steps=num_train_steps

    )



    

    for epoch in range(EPOCHS):



        para_loader = pl.ParallelLoader(train_data_loader, [device])

        trn_loss = train_fn(

            para_loader.per_device_loader(device), 

            model, 

            optimizer, 

            device,

            scheduler

        )



        para_loader = pl.ParallelLoader(valid_data_loader, [device])

        val_loss = eval_fn(para_loader.per_device_loader(device), model, device)

        

        xm.master_print(f'Epoch={epoch}, train_loss={trn_loss}, valid_loss={val_loss}')

    xm.save(model.state_dict(), f"model_{epoch}.bin")

    



def _mp_fn(rank, flags):

    torch.set_default_tensor_type('torch.FloatTensor')

    a = run()
MX = JaSQuADBert()
FLAGS={}

xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=8, start_method='fork')
!head -n3 result_*.csv
!ls