# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import Dataset,DataLoader

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator

from ignite.metrics import Accuracy, Loss

from ignite.engine.engine import Engine, State, Events

from ignite.utils import convert_tensor

from ignite.contrib.handlers import TensorboardLogger, ProgressBar

from torch.optim import Adam

import gc

import warnings  

warnings.filterwarnings('ignore')
%%writefile preproc.py

import gc

from tqdm import tqdm

import pandas as pd

import numpy as np

import pyarrow.parquet as pq

def get_data():



    train = pq.read_table('/kaggle/input/vsb-power-line-fault-detection/train.parquet',columns = [str(i) for i in range(0,6000)])

    train = train.to_pandas().T

    target = pd.read_csv('/kaggle/input/vsb-power-line-fault-detection/metadata_train.csv')['target'].iloc[0:6000]

    gc.collect()

    return train.values,target.values
from preproc import get_data

x_train, y_train = get_data()

gc.collect()
class PowerlineDataset(Dataset):

    def __init__(self,features,labels):

        super().__init__()

        self.labels = labels

        self.features = features

        

    def __len__(self):

        return len(self.labels)

    

    def __getitem__(self,idx):

        labels = self.labels[idx]

        features = self.features[idx]

        return features, labels
class Attention(nn.Module):

    """ Applies attention mechanism on the `context` using the `query`.

    **Thank you** to IBM for their initial implementation of :class:`Attention`. Here is

    their `License

    <https://github.com/IBM/pytorch-seq2seq/blob/master/LICENSE>`__.

    Args:

        dimensions (int): Dimensionality of the query and context.

        attention_type (str, optional): How to compute the attention score:

            * dot: :math:`score(H_j,q) = H_j^T q`

            * general: :math:`score(H_j, q) = H_j^T W_a q`

    Example:

         >>> attention = Attention(256)

         >>> query = torch.randn(5, 1, 256)

         >>> context = torch.randn(5, 5, 256)

         >>> output, weights = attention(query, context)

         >>> output.size()

         torch.Size([5, 1, 256])

         >>> weights.size()

         torch.Size([5, 1, 5])

    """



    def __init__(self, dimensions, attention_type='general'):

        super(Attention, self).__init__()



        if attention_type not in ['dot', 'general']:

            raise ValueError('Invalid attention type selected.')



        self.attention_type = attention_type

        if self.attention_type == 'general':

            self.linear_in = nn.Linear(dimensions, dimensions, bias=False)



        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)

        self.softmax = nn.Softmax(dim=-1)

        self.tanh = nn.Tanh()



    def forward(self, query, context):

        """

        Args:

            query (:class:`torch.FloatTensor` [batch size, output length, dimensions]): Sequence of

                queries to query the context.

            context (:class:`torch.FloatTensor` [batch size, query length, dimensions]): Data

                overwhich to apply the attention mechanism.

        Returns:

            :class:`tuple` with `output` and `weights`:

            * **output** (:class:`torch.LongTensor` [batch size, output length, dimensions]):

              Tensor containing the attended features.

            * **weights** (:class:`torch.FloatTensor` [batch size, output length, query length]):

              Tensor containing attention weights.

        """

        batch_size, output_len, dimensions = query.size()

        query_len = context.size(1)



        if self.attention_type == "general":

            query = query.reshape(batch_size * output_len, dimensions)

            query = self.linear_in(query)

            query = query.reshape(batch_size, output_len, dimensions)



        # TODO: Include mask on PADDING_INDEX?



        # (batch_size, output_len, dimensions) * (batch_size, query_len, dimensions) ->

        # (batch_size, output_len, query_len)

        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())



        # Compute weights across every context sequence

        attention_scores = attention_scores.view(batch_size * output_len, query_len)

        attention_weights = self.softmax(attention_scores)

        attention_weights = attention_weights.view(batch_size, output_len, query_len)



        # (batch_size, output_len, query_len) * (batch_size, query_len, dimensions) ->

        # (batch_size, output_len, dimensions)

        mix = torch.bmm(attention_weights, context)



        # concat -> (batch_size * output_len, 2*dimensions)

        combined = torch.cat((mix, query), dim=2)

        combined = combined.view(batch_size * output_len, 2 * dimensions)



        # Apply linear_out on every 2nd dimension of concat

        # output -> (batch_size, output_len, dimensions)

        output = self.linear_out(combined).view(batch_size, output_len, dimensions)

        output = self.tanh(output)



        return output, attention_weights
class Dilated_Dense_Layers(nn.Module):

    def __init__(self,in_channels,out_channels,dilation_rates):

        super().__init__()

        self.dilation_rates = dilation_rates

        self.dilated_conv = nn.ModuleList()

        self.casual_conv1 = nn.Conv1d(in_channels,out_channels,kernel_size=1)

        self.casual_conv2 = nn.Conv1d(out_channels,out_channels,kernel_size=1)

        self.avgpool1d = nn.AvgPool1d(10)

        dilation_rates = [2**dilation_rate for dilation_rate in range(dilation_rates)]

        for dilation_rate in dilation_rates:

            self.dilated_conv.append(nn.Conv1d(out_channels,out_channels,kernel_size=3,padding=dilation_rate,dilation=dilation_rate))

        

    def forward(self,x):

        x = self.casual_conv1(x)

        res_x = x

        for i in range(self.dilation_rates):

            x = F.tanh(self.dilated_conv[i](x))*F.sigmoid(self.dilated_conv[i](x))

            x += res_x

        x = self.casual_conv2(x)

        x = self.avgpool1d(x)

        return x

    

class WaveLSTM(nn.Module):

    def __init__(self):

        super().__init__()

        self.out_channels = 16

        self.dilation_dense_layers1 = Dilated_Dense_Layers(1,self.out_channels,4)

        self.dilation_dense_layers2 = Dilated_Dense_Layers(self.out_channels,self.out_channels*2,3)

        self.dilation_dense_layers3 = Dilated_Dense_Layers(self.out_channels*2,self.out_channels*4,2)

        self.dilation_dense_layers4 = Dilated_Dense_Layers(self.out_channels*4,self.out_channels*8,1)

        self.LSTM = nn.GRU(800,800,bidirectional=True,batch_first=True)

        self.fc = nn.Linear(800,1)

        

    def forward(self,x):

        x = self.dilation_dense_layers1(x.unsqueeze(1).float())

        x = self.dilation_dense_layers2(x)

        x = self.dilation_dense_layers3(x)

        a,b = self.LSTM(x)

        x,_ = Attention(800)(b.permute(1,0,2)[:,0,:].unsqueeze(1),a[:,:,:800])

        x = self.fc(x)

        return F.sigmoid(x)

        
model = WaveLSTM()

print(model)
def get_dataloader():

    from sklearn.model_selection import train_test_split

    x_train1, x_test1, y_train1, y_test1 = train_test_split(x_train, y_train, test_size=0.33, random_state=42)

    train_dataset = PowerlineDataset(x_train1,y_train1)

    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=4,shuffle=True)

    valid_dataset = PowerlineDataset(x_test1,y_test1)

    valid_loader = torch.utils.data.DataLoader(valid_dataset,batch_size=4,shuffle=True)

    gc.collect()

    return train_loader,valid_loader

train_loader,valid_loader = get_dataloader()
def _prepare_batch(batch, device=None, non_blocking=False):

    """Prepare batch for training: pass to a device with options.



    """

    x, y = batch

    return (convert_tensor(x, device=device, non_blocking=non_blocking),

            convert_tensor(y, device=device, non_blocking=non_blocking))

    

def create_supervised_evaluator1(model, metrics=None,

                                device=None, non_blocking=False,

                                prepare_batch=_prepare_batch,

                                output_transform=lambda x, y, y_pred: (y_pred, y,)):

    """

    Factory function for creating an evaluator for supervised models.



    Args:

        model (`torch.nn.Module`): the model to train.

        metrics (dict of str - :class:`~ignite.metrics.Metric`): a map of metric names to Metrics.

        device (str, optional): device type specification (default: None).

            Applies to both model and batches.

        non_blocking (bool, optional): if True and this copy is between CPU and GPU, the copy may occur asynchronously

            with respect to the host. For other cases, this argument has no effect.

        prepare_batch (callable, optional): function that receives `batch`, `device`, `non_blocking` and outputs

            tuple of tensors `(batch_x, batch_y)`.

        output_transform (callable, optional): function that receives 'x', 'y', 'y_pred' and returns value

            to be assigned to engine's state.output after each iteration. Default is returning `(y_pred, y,)` which fits

            output expected by metrics. If you change it you should use `output_transform` in metrics.



    Note: `engine.state.output` for this engine is defind by `output_transform` parameter and is

        a tuple of `(batch_pred, batch_y)` by default.



    Returns:

        Engine: an evaluator engine with supervised inference function.

    """

    metrics = metrics or {}



    if device:

        model



    def _inference(engine, batch):

        model.eval()

        with torch.no_grad():

            x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)

            y_pred = model(x)

            return output_transform(x, y.float(), y_pred)



    engine = Engine(_inference)



    for name, metric in metrics.items():

        metric.attach(engine, name)



    return engine



def create_supervised_trainer1(model, optimizer, loss_fn, metrics={}, device=None):



    def _update(engine, batch):

        model.train()

        optimizer.zero_grad()

        x, y = _prepare_batch(batch, device=device)

        y_pred = model(x)

        loss = loss_fn(y_pred, y.float())

        loss.backward()

        optimizer.step()

        return loss.item(), y_pred, y.float()



    def _metrics_transform(output):

        return output[0],output[1], output[2]



    engine = Engine(_update)



    for name, metric in metrics.items():

        metric._output_transform = _metrics_transform

        metric.attach(engine, name)



    return engine

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

optimizer = Adam(model.parameters(),lr=0.0003)

criterion = nn.BCELoss()

metrics = {

    'loss': Loss(criterion),

}

trainer = create_supervised_trainer1(model.to(device), optimizer,criterion, device=device)

val_evaluator = create_supervised_evaluator1(model.to(device), metrics=metrics, device=device)

pbar = ProgressBar(bar_format='')

pbar.attach(trainer, output_transform=lambda x: {'loss': x[0]})

@trainer.on(Events.EPOCH_COMPLETED)

def compute_and_display_val_metrics(engine):

    epoch = engine.state.epoch

    metrics = val_evaluator.run(valid_loader).metrics

    print("Validation Results - Epoch: {}  Average Loss: {:.4f}"

          .format(engine.state.epoch, 

                      metrics['loss']))

@trainer.on(Events.EPOCH_COMPLETED)

def save_model(engine):

    torch.save(model.state_dict(), 'saved_model.pth')

gc.collect()
if torch.cuda.is_available():

    torch.set_default_tensor_type(torch.cuda.FloatTensor)

trainer.run(train_loader, max_epochs=5)
class TestPowerlineDataset(Dataset):

    def __init__(self,features):

        super().__init__()

        self.features = features

        

    def __len__(self):

        return len(self.features)

    

    def __getitem__(self,idx):



        features = self.features[idx]

        return features
def test_load(phase,signal_ids,batch_size):

    x_test = pq.read_table('/kaggle/input/vsb-power-line-fault-detection/test.parquet',columns =[str(signal_ids[i]) for i in range(batch_size*phase,batch_size*(phase+1))])

    x_test1 = x_test.to_pandas().T

    return x_test1.values
del train_loader

del valid_loader
from tqdm import tqdm

import pyarrow.parquet as pq

def saver():

    if torch.cuda.is_available():

        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    submission = pd.read_csv('/kaggle/input/vsb-power-line-fault-detection/sample_submission.csv')

    signal_ids = pd.read_csv('/kaggle/input/vsb-power-line-fault-detection/metadata_test.csv')['signal_id'].values

    model.load_state_dict(torch.load('saved_model.pth'))

    model.eval()

    predictions = []

    iterations = 5

    batch_size = len(signal_ids)//5

    for i in range(iterations):

        predictions = []

        print('iteraion {}'.format(i+1))

        x_test = test_load(phase=i,batch_size=batch_size,signal_ids=signal_ids)

        test_dataset = TestPowerlineDataset(x_test)

        test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=20,shuffle=False)

        with torch.no_grad():

            for idx , inputs in tqdm(enumerate(test_loader),total=len(test_loader)):

                preds = model(inputs.float()).to(device)

                predictions.append(np.squeeze(preds.cpu().detach().numpy()))

        

            predictions = np.round(np.hstack(predictions)).astype(int)

            submission[batch_size*i:batch_size*(i+1)].target = predictions

            gc.collect()

    submission.to_csv('submission.csv',index=False)
#saver()