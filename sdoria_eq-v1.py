# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

import torch.optim as optim

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# Any results you write to the current directory are saved as output.



from tqdm import tqdm



import matplotlib.pyplot as plt

%matplotlib inline



from fastai.basics import *

from fastai.callbacks import * 
PATH='../input/'

!ls '../input/'
#float_data = pd.read_csv("../input/train.csv",  nrows=10000000, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})

float_data = pd.read_csv(PATH+'train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32}).values
# Data is a 629M long time series. 

# The data is recorded in bins of 4096 samples. Withing those bins seismic data is recorded at 4MHz, 

# but there is a 12 microseconds (??? or is it 1.2 millisecond) gap between each bin, an artifact of the recording device.

# Test data consists of 150,000 long time series. We don't know where the gaps are in the test data.



# Our dataset will create items that are each based on 150,000 long chunks of data



class EQdataset (Dataset):

    def __init__(self, data, min_index =0, max_index = None, bin_density = 0.1, is_test = False):

        

        self.LENGTH_SAMPLE =150000

        self.BIN_SIZE = 4096   #4096

        self.BIN_CHUNK_SIZE = 64    #needs to be a divisor of 4096 (bin size)  #16

        self.BIN_CHUNK_COUNT = int(self.BIN_SIZE/self.BIN_CHUNK_SIZE) 

        self.min_index = min_index

        self.max_index = max_index

        if max_index is None:

            self.max_index = len(data) - 1

        

        

        self.data = data[self.min_index:self.max_index+1]

        

        self.is_test = is_test

        

        if is_test:

            self.dataset_size = 1

            self.ending_positions = [150000]

        else:

            # Pick indices of ending positions

            # dataset_size = number of bins * bin_density

            self.dataset_size = int(np.round(len(self.data)/ self.BIN_SIZE * bin_density) )

            # remember self.data index starts at 0 and not at min_index

            self.ending_positions = np.random.randint(self.LENGTH_SAMPLE, self.max_index - self.min_index , size=self.dataset_size)



        

        

            

    

    def __len__(self):

        return self.dataset_size

                 

    def __getitem__(self, idx):

        

        ending_position = self.ending_positions[idx]

        starting_position = ending_position - self.LENGTH_SAMPLE

        idx_data = self.data[starting_position: ending_position,:]

        

        

        # for each data point in the last bin, we form a 35 data point time series using data points separated by 4096 rows

        # This way we have a constant time gap between each data point in the time series

        rnn_length = int(np.floor (self.LENGTH_SAMPLE/self.BIN_SIZE)) -1   # =36

        rnn_data_indexes = np.array([self.LENGTH_SAMPLE -1 - (rnn_length -1 - n) * self.BIN_SIZE for  n in range(rnn_length)])

        

        # 256 rnn inputs of length 36 and depth 16

        

        #X = np.array([[idx_data[index-self.BIN_CHUNK_SIZE+1 -bin_chunk_num*self.BIN_CHUNK_SIZE : index+1-bin_chunk_num*self.BIN_CHUNK_SIZE][:,0] for index in rnn_data_indexes] 

                     #for bin_chunk_num in range(self.BIN_CHUNK_COUNT)])

        

        X = idx_data[self.LENGTH_SAMPLE-self.BIN_CHUNK_COUNT*rnn_length*self.BIN_CHUNK_SIZE:,0].reshape((

            rnn_length,self.BIN_CHUNK_COUNT,self.BIN_CHUNK_SIZE)).transpose(1,0,2)

        # 256 rnn output of length 36

        #y = np.array([idx_data[rnn_data_indexes - bin_chunk_num*self.BIN_CHUNK_SIZE][:,1] for bin_chunk_num in range(self.BIN_CHUNK_COUNT)])

        

        #y = idx_data[150000-self.BIN_CHUNK_COUNT*rnn_length*self.BIN_CHUNK_SIZE:,1].reshape((rnn_length,self.BIN_CHUNK_COUNT,self.BIN_CHUNK_SIZE)).transpose(1,0,2)[:,:,15] 

        

        if self.is_test:

            y =0

            

        else: 

            y = idx_data[self.LENGTH_SAMPLE -1,1]

            

        # 'meta' data

        #inspired by https://www.kaggle.com/artgor/earthquakes-fe-more-features-and-samples

        meta = []

        idx_data_series = pd.Series(idx_data[:,0])

        for windows in [10, 100, 1000]:

            

            x_roll_std = idx_data_series.rolling(windows).std().dropna().values

            meta.append(np.quantile(x_roll_std, 0.05))    

        

        

        return [torch.from_numpy(X), torch.from_numpy(np.array(meta,dtype='float32'))] , y

        

        




class EQRNN(nn.Module):

    def __init__(self, input_size = 64,meta_size=0, fc1_size = 40, hidden_size=80,num_layers=1,bidirectional=True, dropout=0.5):

        

        super().__init__()

        self.input_size = input_size

        

        self.meta_size = meta_size

        

        self.fc1_size = fc1_size

        self.hidden_size = hidden_size

        

        self.fc1 = nn.Linear(input_size, fc1_size)

        

        self.bidirectional,self.num_layers = bidirectional,num_layers

        if bidirectional: self.num_directions = 2

        else: self.num_directions = 1

            

        self.dropout = dropout

            

        self.rnn = nn.GRU(fc1_size, hidden_size,bidirectional=self.bidirectional,batch_first=True)

        

       

        self.layers2 = nn.Sequential(

            

            nn.Linear(self.num_directions * hidden_size +meta_size,128),    #32

            nn.ReLU(),

            nn.Dropout(self.dropout),   ##added

            nn.Linear(128,128),

            nn.ReLU(),

            nn.Dropout(self.dropout),

            nn.Linear(128,1),

            

            

        )

        

        

        #self.final_layers = nn.Sequential(

            

           # nn.Linear(256,128),

           # nn.ReLU(),

            #nn.Dropout(self.dropout),   ##added

           # nn.Linear(128,128),

           # nn.ReLU(),

           # nn.Dropout(self.dropout),

           # nn.Linear(128,1),

            

            

        #)

        self.final_layers = nn.Sequential(

            

            nn.Linear(4,16),

            nn.ReLU(),

            nn.Dropout(self.dropout),   ##added

            nn.Linear(16,16),

            nn.ReLU(),

            nn.Dropout(self.dropout),

            nn.Linear(16,1)

        )

            

        

        

    def forward(self,input_seq, meta):

    

        

        #cont_input_seq, cat_input_seq, meta_input_seq = input_seqs[0]

        # Note: dataloader provides vector in batch first form, whereas recurent layers need to have batch_first=True specified

        # hidden layers still have batch size in second position even with batch_first=True

        batch_size = input_seq.size(0)

        rnn_count = input_seq.size(1)

        seq_len = input_seq.size(2)

        

        

        

        

        input_seq = F.relu(self.fc1(input_seq))

        

        

        # combine first two dimensions (batchsize and rnncount) so that our input is 3D

        

        input_seq = input_seq.view(batch_size*rnn_count,seq_len, -1)

        

        

        #output of shape ( batch_size*rnn_count, seq_len, num_directions * hidden_size)

        #h_n (not needed)

        output, h_n = self.rnn(input_seq)#,h_0)

        

        

        

        

        #SpatialDropout1D in TensorFlow

        #https://discuss.pytorch.org/t/spatial-dropout-in-pytorch/21400/2

        

        # drop whole channels across all timesteps

        

        output = output.permute(0, 2, 1)    # convert to [batch,rnn_count, channels, time]

        output = F.dropout2d(output, self.dropout, training=self.training)

        output = output.permute(0,2, 1)   # back to original

        

        

        #same as GlobalMaxPool1D in TensorFlow?

        # take max over seq_len dimension

        # could also try concatenation of Average and Max Pooling 

        

        output = F.adaptive_max_pool1d(output.permute(0, 2, 1),output_size=1).permute(0, 2, 1)

        output = F.dropout(output,self.dropout,training=self.training)

        

        #output2 = F.adaptive_avg_pool1d(output.permute(0,2,1),output_size=1).permute(0, 2, 1).view(batch_size,-1)

        #output2 = F.dropout(output2,self.dropout,training=self.training)

        #output = torch.cat((output1,output2),1)

        

        # now shape (bs, rnn_count, num_directions * hidden_size)

        

        # reseparate the first two dims

        

        output = output.view(batch_size,rnn_count, -1)

        

        

        

        #output = torch.cat((output,meta.view(batch_size,1,self.meta_size).repeat(1,rnn_count,1)), 2)

        

        #add meta data to output

        #output = torch.cat((meta_input_seq.view(batch_size,-1),output),1)

        

        #shared layers for all rnn_count outputs

        output = self.layers2(output)

        

        # now shape (bs, rnn_count, num_directions * hidden_size)

        

        output = output.view(batch_size,-1)

        #output = self.final_layers(output)

        

        output = torch.mean(output,dim =1)

        

        

        #output = torch.cat((output.view(batch_size,-1),meta), 1)

        

        #output = self.final_layers(output)

        

        return output

        

        
val_max_index = int(len(float_data) * 0.2 )    #last index for validation





val_ds = EQdataset(float_data,min_index =0, max_index = val_max_index)

trn_ds = EQdataset(float_data,min_index =val_max_index +1)
trn_ds[0][0][0].shape
bs = 4096

bs2 = 4096

num_workers = 4



#train_dl = DataLoader(trn_ds, batch_size=bs,shuffle=True, num_workers=num_workers)

#val_dl = DataLoader(val_ds, batch_size=bs2,shuffle=False, num_workers=num_workers)

net = EQRNN(dropout=0.5)

net


# CUDA for PyTorch

use_cuda = torch.cuda.is_available()

device = torch.device("cuda:0" if use_cuda else "cpu")



#criterion = nn.NLLLoss(weight=torch.from_numpy(weights).to(device))



criterion =  nn.L1Loss()

#criterion = mywloss



databunch = DataBunch.create(trn_ds,val_ds, device=device, bs =32)
learn = Learner(databunch,net,callback_fns=[ShowGraph], loss_func = criterion)

learn.opt_func=AdamW

# save the best model

learn.callbacks = [SaveModelCallback(learn,monitor='val_loss',mode='min')]
#lr_find(learn)

#learn.recorder.plot()
lr=1e-2

wd = 1e-3

learn.fit(2,lr,wd=wd)
trn_ds = EQdataset(float_data,min_index =val_max_index +1)

databunch = DataBunch.create(trn_ds,val_ds, device=device, bs=32)

learn.data = databunch
learn.fit(2,lr,wd=wd)
lr=1e-3

wd = 1e-3

learn.fit(2,lr,wd=wd)
learn.fit(4,lr,wd=wd)
learn.save('EQ205')
#preds, target = learn.get_preds(ds_type=DatasetType.Valid)

# Load submission file

submission = pd.read_csv(PATH+'sample_submission.csv', index_col='seg_id', dtype={"time_to_failure": np.float32})



# Load each test data



test= []

for i, seg_id in enumerate(tqdm(submission.index)):

    

    seg = pd.read_csv(PATH+ seg_id + '.csv',dtype={'acoustic_data': np.float32})

    test_ds  = EQdataset(np.array(seg),is_test = True)

    learn.data = DataBunch.create(trn_ds,val_ds,test_ds, device=device)  

    preds, _ = learn.get_preds(ds_type=DatasetType.Test)

    submission['time_to_failure'][i] =preds

   
test_ds  = EQdataset(np.array(seg),is_test = True)

learn.data = DataBunch.create(trn_ds,val_ds,test_ds, device=device)  

preds, _ = learn.get_preds(ds_type=DatasetType.Test)
submission['time_to_failure'].min()
submission.to_csv('EQsubmission2.csv')
float_data[:,1].min()
float_data[150000][1] - float_data[0][1]