import os

import time

import gc

import random

from tqdm._tqdm_notebook import tqdm_notebook as tqdm



import pandas as pd

import numpy as np

import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset

from torch.utils.data.sampler import BatchSampler, SequentialSampler

from torch.optim import AdamW

from torch.optim.lr_scheduler import CosineAnnealingLR
def seed_everything(seed=2020):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

seed_everything()
train = pd.read_json('/kaggle/input/stanford-covid-vaccine/train.json', lines=True)

test = pd.read_json('/kaggle/input/stanford-covid-vaccine/test.json', lines=True)

sample_df = pd.read_csv('/kaggle/input/stanford-covid-vaccine/sample_submission.csv')
cols = ['sequence', 'structure', 'predicted_loop_type']

pred_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']

token2int = {x:i for i, x in enumerate('().ACGUBEHIMSX')}



def preprocess_inputs(data):

    '''

    Credits goes to @xhlulu: https://www.kaggle.com/xhlulu/openvaccine-simple-gru-model

    '''

    return np.transpose(

        np.array(

            data[cols]

            .applymap(lambda seq: [token2int[x] for x in seq])

            .values

            .tolist()

        ),

        (0, 2, 1)

    )
class OpenVaccineDataset(Dataset):

    def __init__(self, data, labels):

        super(OpenVaccineDataset, self).__init__()

        self.data = data

        self.labels = labels

    

    def __len__(self):

        return len(self.data)

    

    def __getitem__(self, idx):

        return {

            'x': torch.tensor(self.data[idx]),

            'y': torch.tensor(self.labels[idx])

        }
max_features = None

max_features = max_features or len(token2int)

pred_len = 68

EMBEDDING_DIM = 100

LSTM_UNITS = 128

DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS

NUM_TARGETS = 5

LR = 1e-3

BATCH_SIZE = 32

EPOCHS = 150
class SpatialDropout(nn.Dropout2d):

    

    def forward(self, x):

        x = x.permute(0, 3, 2, 1)  

        x = super(SpatialDropout, self).forward(x)  

        x = x.permute(0, 3, 2, 1)  

        return x

    

class NeuralNet(nn.Module):

    

    def __init__(self, embed_size, num_targets):

        super(NeuralNet, self).__init__()

        

        self.embedding = nn.Embedding(max_features, embed_size)

        self.embedding_dropout = SpatialDropout(0.3)

        

        self.lstm1 = nn.LSTM(embed_size * 3, LSTM_UNITS, bidirectional=True, batch_first=True)

        self.lstm2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True)

        

        self.linear_out = nn.Linear(LSTM_UNITS * 2, num_targets)



    def forward(self, x):

        h_embedding = self.embedding(x)

        h_embedding = self.embedding_dropout(h_embedding)

        

        h_reshaped = torch.reshape(

            h_embedding, 

            shape=(-1, h_embedding.shape[1],  h_embedding.shape[2] * h_embedding.shape[3])

        )

        

        h_lstm1, _ = self.lstm1(h_reshaped)

        h_lstm2, _ = self.lstm2(h_lstm1)

        h_truncated = h_lstm2[:, :pred_len]

        

        return self.linear_out(h_truncated)
def loss_fn(outputs, targets):

    colwise_mse = torch.mean(torch.square(targets - outputs), dim=(0, 1))

    loss = torch.mean(torch.sqrt(colwise_mse), dim=-1)

    return loss
def get_model_optimizer(model):

    # Differential Learning Rate

    def is_linear(name):

        return "linear" in name

    

    optimizer_grouped_parameters = [

       {'params': [param for name, param in model.named_parameters() if not is_linear(name)], 'lr': LR},

       {'params': [param for name, param in model.named_parameters() if is_linear(name)], 'lr': LR*3} 

    ]

    

    optimizer = AdamW(

        optimizer_grouped_parameters, lr=LR, weight_decay=1e-2

    )

    

    return optimizer
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
def train_loop_fn(train_loader, model, optimizer, device, scheduler, epoch=None):

    # Train

    batch_time = AverageMeter('Time', ':6.3f')

    losses = AverageMeter('Loss', ':2.4f')

    progress = ProgressMeter(

        len(train_loader),

        [batch_time, losses],

        prefix="[TRAIN] Epoch: [{}]".format(epoch)

    )

    model.train()

    end = time.time()

    for i, data in enumerate(train_loader):

        inputs = data['x']

        targets = data['y']

        inputs = inputs.to(device, dtype=torch.long)

        targets = targets.to(device, dtype=torch.float)

        

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = loss_fn(outputs, targets)

        loss.backward()

        optimizer.step()



        losses.update(loss.item(), BATCH_SIZE)

        scheduler.step()

        batch_time.update(time.time() - end)

        end = time.time()

        if i % 37 == 0 and i !=0:

            progress.display(i)
def _run():

    print('Starting Training ... ')

    

    print('  Loading Data ... ')

    train_data = preprocess_inputs(data=train)

    train_labels = np.array(train[pred_cols].values.tolist()).transpose((0, 2, 1))

    train_dataset = OpenVaccineDataset(

        train_data,

        train_labels

    )

    train_data_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        drop_last=False,

        pin_memory=True,

        num_workers=4

    )

    print('  Data Loading Completed ... ')

    

    print('  Loading Model Configurations ... ')

    num_train_steps = int(len(train_dataset)) / BATCH_SIZE

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = NeuralNet(

        embed_size=EMBEDDING_DIM, 

        num_targets=NUM_TARGETS

    )

    model = model.to(device)

    optimizer = get_model_optimizer(model)

    scheduler = CosineAnnealingLR(optimizer, T_max=num_train_steps*EPOCHS)

    print('  Model Configuration Completed ... ')

    

    print('Training Started ... ')

    for epoch in range(EPOCHS):

        train_loop_fn(

            train_data_loader,

            model,

            optimizer,

            device,

            scheduler,

            epoch

        )

        if epoch == EPOCHS-1:

            print('  Saving Model ...')

            torch.save(model.state_dict(), 'model.bin')

            print('  Model Saved ...')

    

    print('Training Completed.')
if __name__ == "__main__":

    _run()
class OpenVaccineTestDataset(Dataset):

    def __init__(self, data):

        super().__init__()

        self.data = data

    

    def __len__(self):

        return len(self.data)

    

    def __getitem__(self, idx):

        return {

            'x': self.data[idx]

        }
def test_loop_fn(test_loader, model, device):

    model.eval()

    end = time.time()

    preds = []

    for i, data in tqdm(enumerate(test_loader),total=len(test_loader)):

        inputs = data['x']

        inputs = inputs.to(device, dtype=torch.long)

        outputs = model(inputs)

        preds.append(outputs.detach().cpu().numpy())

    return preds
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



pred_len = 107

model_short = NeuralNet(embed_size=EMBEDDING_DIM, num_targets=NUM_TARGETS).to(device)

model_short.load_state_dict(

    torch.load('model.bin')

)



pred_len = 130

model_long = NeuralNet(embed_size=EMBEDDING_DIM, num_targets=NUM_TARGETS).to(device)

model_long.load_state_dict(

    torch.load('model.bin')

)



public_df = test.query("seq_length == 107").copy()

private_df = test.query("seq_length == 130").copy()



public_inputs = preprocess_inputs(public_df)

private_inputs = preprocess_inputs(private_df)



public_dataset = OpenVaccineTestDataset(public_inputs)

private_dataset = OpenVaccineTestDataset(private_inputs)



public_data_loader = DataLoader(

    public_dataset,

    shuffle=False,

    batch_size=16,

    pin_memory=False,

    drop_last=False,

    num_workers=0

)

private_data_loader = DataLoader(

    private_dataset,

    shuffle=False,

    batch_size=16,

    pin_memory=False,

    drop_last=False,

    num_workers=0

)
public_preds = np.vstack(test_loop_fn(public_data_loader, model_short, device))

private_preds = np.vstack(test_loop_fn(private_data_loader, model_long, device))
preds_ls = []



for df, preds in [(public_df, public_preds), (private_df, private_preds)]:

    for i, uid in enumerate(df.id):

        single_pred = preds[i]



        single_df = pd.DataFrame(single_pred, columns=pred_cols)

        single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]



        preds_ls.append(single_df)



preds_df = pd.concat(preds_ls)
submission = sample_df[['id_seqpos']].merge(preds_df, on=['id_seqpos'])

submission.to_csv('submission.csv', index=False)