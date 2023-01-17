##########################Load Libraries  ####################################

import pandas as pd

import numpy as np

import dask.dataframe as dd

pd.set_option('display.max_columns', 500)

pd.set_option('display.max_rows', 500)

import matplotlib.pyplot as plt

import seaborn as sns

import lightgbm as lgb

from sklearn import preprocessing, metrics

from ipywidgets import widgets, interactive

import gc

import joblib

import warnings

warnings.filterwarnings('ignore')

from datetime import datetime, timedelta 

from typing import Union

from tqdm.notebook import tqdm_notebook as tqdm

from itertools import cycle

import datetime as dt

from torch.autograd import Variable

import random 

import os

from matplotlib.pyplot import figure

from fastprogress import master_bar, progress_bar

import torch

import torch.nn as nn

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler,StandardScaler

import time 

import torch.nn.functional as F

from torch.utils.data import Dataset

from sklearn.metrics import mean_squared_error

import torch 

from sklearn.model_selection import TimeSeriesSplit



from math import log

from math import exp

from scipy.stats import boxcox



 

import torch.optim as optim



%matplotlib inline



#from gensim.models import Word2Vec

#import gensim.downloader as api



pd.set_option('max_columns', 50)

plt.style.use('bmh')

color_pal = plt.rcParams['axes.prop_cycle'].by_key()['color']

color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])



 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("device is:",device)
INPUT_DIR_PATH = '../input/m5-forecasting-accuracy/'
def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics: 

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df





def read_data():

    sell_prices_df = pd.read_csv(INPUT_DIR_PATH + 'sell_prices.csv')

    sell_prices_df = reduce_mem_usage(sell_prices_df)

    print('Sell prices has {} rows and {} columns'.format(sell_prices_df.shape[0], sell_prices_df.shape[1]))



    calendar_df = pd.read_csv(INPUT_DIR_PATH + 'calendar.csv')

    calendar_df = reduce_mem_usage(calendar_df)

    print('Calendar has {} rows and {} columns'.format(calendar_df.shape[0], calendar_df.shape[1]))



    sales_train_validation_df = pd.read_csv(INPUT_DIR_PATH + 'sales_train_validation.csv')

    print('Sales train validation has {} rows and {} columns'.format(sales_train_validation_df.shape[0], sales_train_validation_df.shape[1]))



    submission_df = pd.read_csv(INPUT_DIR_PATH + 'sample_submission.csv')

    return sell_prices_df, calendar_df, sales_train_validation_df, submission_df
_,  calendar_df, sales_train_validation_df, _ = read_data()
#Create date index

date_index = calendar_df['date']

dates = date_index[0:1913]

dates_list = [dt.datetime.strptime(date, '%Y-%m-%d').date() for date in dates]
# Create a data frame for items sales per day with item ids (with Store Id) as columns names  and dates as the index 

sales_train_validation_df['item_store_id'] = sales_train_validation_df.apply(lambda x: x['item_id']+'_'+x['store_id'],axis=1)

DF_Sales = sales_train_validation_df.loc[:,'d_1':'d_1913'].T

DF_Sales.columns = sales_train_validation_df['item_store_id'].values



#Set Dates as index 

DF_Sales = pd.DataFrame(DF_Sales).set_index([dates_list])

DF_Sales.index = pd.to_datetime(DF_Sales.index)



DF_Sales.head()
#Select arbitrary index and plot the time series

index = 321

y = pd.DataFrame(DF_Sales.iloc[:,index])

TS_selected = y 

y = pd.DataFrame(y).set_index([dates_list])

y.index = pd.to_datetime(y.index)

ax = y.plot(figsize=(30, 9),color='black')

ax.set_facecolor('lightgrey')

plt.xticks(fontsize=21 )

plt.yticks(fontsize=21 )

plt.legend(fontsize=20)

plt.title(label = 'Sales Demand Selected Time Series Over Time',fontsize = 23)

plt.ylabel(ylabel = 'Sales Demand',fontsize = 21)

plt.xlabel(xlabel = 'Date',fontsize = 21)

plt.show()

SEED = 42

def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True





seed_everything(SEED)
train_size = int((len(TS_selected)-28) * 0.67)

valid_size = (len(TS_selected)-28)- train_size

print("train size is:",train_size)

print("validation size is:",valid_size)
train_data = TS_selected.iloc[0:train_size,:]

valid_data = TS_selected.iloc[train_size:train_size+valid_size,:]

print("train data shape is:",train_data.shape)

print("validation data shape is:",valid_data.shape)

train_data.head()
## Simple Difference transform  --> X(t)' = X(t) - X(t-7)

def difference(data, interval):

    return [data[i] - data[i - interval] for i in range(interval, len(data))]



np_tarin_data = train_data.iloc[:,0]

interval = 7 

transformed_train = difference(np_tarin_data, interval)



np_valid_data = valid_data.iloc[:,0]

interval = 7 

transformed_valid = difference(np_valid_data, interval)



train_data = np.array(transformed_train).reshape(len(transformed_train),1)

valid_data = np.array(transformed_valid).reshape(len(transformed_valid),1)
train_data.shape
valid_data.shape
scaler = MinMaxScaler((-1,1)).fit(train_data)

#scaler = StandardScaler().fit(train_data)

train_data_std = scaler.transform(train_data)

valid_data_std = scaler.transform(valid_data)
fig, axs = plt.subplots(2)

 

fig.suptitle('Data Distribution Before and After Data Transformation   ',fontsize = 19)

pd.DataFrame(train_data).plot(kind='hist',ax = axs[0] , alpha=.4 , figsize=[12,6], legend = False,title = ' Before Transformation',color ='red') 

pd.DataFrame(train_data_std).plot(kind='hist', ax = axs[1] ,figsize=[12,6], alpha=.4 , legend = False,title = ' After Transformation'\

                                         ,color = 'blue')

   
###  This function creates a sliding window or sequences of seq_length days and labels_length  days label ####

def sliding_windows(data, seq_length,labels_length):

    x = []

    y = []



    for i in range(len(data)-(seq_length+labels_length)):

        _x = data[i:(i+seq_length)]

        _y = data[(i+seq_length):(i+seq_length+labels_length)]

        x.append(_x)

        y.append(_y)



    return x,y
seq_length = 90

labels_length =28

train_X, train_y = sliding_windows(train_data_std, seq_length,labels_length)

print("train X  has:", len(train_X) , "series")

print("train labels  has:", len(train_y) , "series")

valid_X, valid_y = sliding_windows(valid_data_std, seq_length,labels_length)

print("validiation  X  has:", len(valid_X) , "series")

print("Validiation  labels  has:" ,len(valid_y) , "series")
plt.figure(figsize=(15,17))

fig, axs =plt.subplots(3,figsize=(12,9))



axs[0].plot(train_data_std[0:seq_length+labels_length])

axs[0].title.set_text('Original Time Series')

axs[0].set_xlim(0,seq_length+labels_length)

axs[0].set_ylim(-1,1)

axs[1].plot(train_X[0].flatten(),color="red")

axs[1].title.set_text('Train Data')

axs[1].set_xlim(0,seq_length+labels_length)

axs[1].set_ylim(-1,1)

axs[2].plot(np.pad(train_y[0].flatten(),seq_length),color='black')

axs[2].title.set_text('Labels Data')

axs[2].set_xlim(0,seq_length+labels_length)

axs[2].set_ylim(-1,1)



trainX = Variable(torch.Tensor(train_X))

trainy = Variable(torch.Tensor(train_y))



validX = Variable(torch.Tensor(valid_X))

validy= Variable(torch.Tensor(valid_y))



print ("trainX shape is:",trainX.size())

print ("trainy shape is:",trainy.size())

print ("validX shape is:",validX.size())

print ("validy shape is:",validy.size())

class Encoder(nn.Module):

    def __init__(self, seq_len, n_features, embedding_dim=64):

        super(Encoder, self).__init__()



        self.seq_len, self.n_features = seq_len, n_features

        self.embedding_dim, self.hidden_dim = embedding_dim,  embedding_dim

        self.num_layers = 3

        self.rnn1 = nn.LSTM(

          input_size=n_features,

          hidden_size=self.hidden_dim,

          num_layers=3,

          batch_first=True,

          dropout = 0.35

        )

   

    def forward(self, x):

       

        x = x.reshape((1, self.seq_len, self.n_features))

        

        h_1 = Variable(torch.zeros(

            self.num_layers, x.size(0), self.hidden_dim).to(device))

         

        

        c_1 = Variable(torch.zeros(

            self.num_layers, x.size(0), self.hidden_dim).to(device))

              

        x, (hidden, cell) = self.rnn1(x,(h_1, c_1))

        

        

        #return hidden_n.reshape((self.n_features, self.embedding_dim))

        return x, hidden , cell 
class Attention(nn.Module):

    def __init__(self, enc_hid_dim, dec_hid_dim):

        super().__init__()

        

        self.attn = nn.Linear((enc_hid_dim ) + dec_hid_dim, dec_hid_dim)

        self.v = nn.Linear(dec_hid_dim, 1, bias = False)

        

    def forward(self, hidden, encoder_outputs):

        

        #hidden = [batch size, dec hid dim]

        #encoder_outputs = [src len, batch size, enc hid dim * 2]

        

        batch_size = encoder_outputs.shape[0]

        src_len = encoder_outputs.shape[1]

        

       

        hidden = hidden[2:3,:,:]

        

        #print("hidden size is",hidden.size())

        

        

        

        #repeat decoder hidden state src_len times

        #hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        hidden = hidden.repeat(1, src_len, 1)

     

        

        #encoder_outputs = encoder_outputs.permute(1, 0, 2)

        

        #print("encode_outputs size after permute is:",encoder_outputs.size())

        

        

        #hidden = [batch size, src len, dec hid dim]

        #encoder_outputs = [batch size, src len, enc hid dim * 2]

        

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 

        

        #energy = [batch size, src len, dec hid dim]



        attention = self.v(energy).squeeze(2)

        

        #attention= [batch size, src len]

        

        

        return F.softmax(attention, dim=1)
class Decoder(nn.Module):

    def __init__(self, seq_len, input_dim=64, n_features=1):

        super(Decoder, self).__init__()



        self.seq_len, self.input_dim = seq_len, input_dim

        self.hidden_dim, self.n_features =  input_dim, n_features

        

        self.rnn1 = nn.LSTM(

          input_size=1,

          hidden_size=input_dim,

          num_layers=3,

          batch_first=True,

          dropout = 0.35

        )

        

        

      

        self.output_layer = nn.Linear(self.hidden_dim, n_features)



    def forward(self, x,input_hidden,input_cell):

       

       

        x = x.reshape((1,1,1))

        

        

     



        x, (hidden_n, cell_n) = self.rnn1(x,(input_hidden,input_cell))

    

        x = self.output_layer(x)

        return x, hidden_n, cell_n
class AttentionDecoder(nn.Module):

    def __init__(self, seq_len,attention, input_dim=64, n_features=1,encoder_hidden_state = 512):

        super(AttentionDecoder, self).__init__()



        self.seq_len, self.input_dim = seq_len, input_dim

        self.hidden_dim, self.n_features =  input_dim, n_features

        self.attention = attention 

        

        self.rnn1 = nn.LSTM(

          #input_size=1,

          input_size= encoder_hidden_state + 1,  # Encoder Hidden State + One Previous input

          hidden_size=input_dim,

          num_layers=3,

          batch_first=True,

          dropout = 0.35

        )

        

        

      

        self.output_layer = nn.Linear(self.hidden_dim * 2 , n_features)



    def forward(self, x,input_hidden,input_cell,encoder_outputs):

       

        a = self.attention(input_hidden, encoder_outputs)

        

        a = a.unsqueeze(1)

        

        #a = [batch size, 1, src len]

        

        #encoder_outputs = encoder_outputs.permute(1, 0, 2)

        

        #encoder_outputs = [batch size, src len, enc hid dim * 2]

        

      

        

        weighted = torch.bmm(a, encoder_outputs)

        

        

     

        x = x.reshape((1,1,1))

       

        

        

        rnn_input = torch.cat((x, weighted), dim = 2)

       



        #x, (hidden_n, cell_n) = self.rnn1(x,(input_hidden,input_cell))

        x, (hidden_n, cell_n) = self.rnn1(rnn_input,(input_hidden,input_cell))

        

        output = x.squeeze(0)

        weighted = weighted.squeeze(0)

        

        x = self.output_layer(torch.cat((output, weighted), dim = 1))

        return x, hidden_n, cell_n

    

      
class Seq2Seq(nn.Module):



    def __init__(self, seq_len, n_features, embedding_dim=64,output_length = 28):

        super(Seq2Seq, self).__init__()



        

        self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)

        self.attention = Attention(512,512)

        self.output_length = output_length

        self.decoder = AttentionDecoder(seq_len, self.attention, embedding_dim, n_features).to(device)

        



    def forward(self,x, prev_y):

        

        encoder_output,hidden,cell = self.encoder(x)

         

        #Prepare place holder for decoder output

        targets_ta = []

        #prev_output become the next input to the LSTM cell

        prev_output = prev_y

        

        #itearate over LSTM - according to the required output days

        for out_days in range(self.output_length) :

        

            prev_x,prev_hidden,prev_cell = self.decoder(prev_output,hidden,cell,encoder_output)

            hidden,cell = prev_hidden,prev_cell

            prev_output = prev_x

            

            targets_ta.append(prev_x.reshape(1))

           

            

        

        

        targets = torch.stack(targets_ta)



        return targets
n_features = 1

model = Seq2Seq(seq_length, n_features, 512)

model = model.to(device)
model
def init_weights(m):

    for name, param in m.named_parameters():

        nn.init.uniform_(param.data, -0.08, 0.08)

        

model.apply(init_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=4e-3,weight_decay=1e-5)

criterion = torch.nn.MSELoss().to(device) 

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 5e-3, eta_min=1e-8, last_epoch=-1)

#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,  patience=10, factor =0.5 ,min_lr=1e-7, eps=1e-08)
def train_model(model, TrainX,Trainy,ValidX,Validy,seq_length, n_epochs):

  

    history = dict(train=[], val=[])



    #best_model_wts = copy.deepcopy(model.state_dict())

    best_loss = 10000.0

    mb = master_bar(range(1, n_epochs + 1))



    for epoch in mb:

        model = model.train()



        train_losses = []

        for i in progress_bar(range(TrainX.size()[0]),parent=mb):

            seq_inp = TrainX[i,:,:].to(device)

            seq_true = Trainy[i,:,:].to(device)

           

            optimizer.zero_grad()



            

            seq_pred = model(seq_inp,seq_inp[seq_length-1:seq_length,:])

            

            

            loss = criterion(seq_pred, seq_true)



            loss.backward()

            

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            optimizer.step()



            train_losses.append(loss.item())



        val_losses = []

        model = model.eval()

        with torch.no_grad():

            for i in progress_bar(range(validX.size()[0]),parent=mb):

                seq_inp = ValidX[i,:,:].to(device)

                seq_true = Validy[i,:,:].to(device)

        

                seq_pred = model(seq_inp,seq_inp[seq_length-1:seq_length,:])

               



                loss = criterion(seq_pred, seq_true)

                val_losses.append(loss.item())



        train_loss = np.mean(train_losses)

        val_loss = np.mean(val_losses)



        history['train'].append(train_loss)

        history['val'].append(val_loss)



        if val_loss < best_loss:

            best_loss = val_loss

            torch.save(model.state_dict(), 'best_model.pt')

            print("saved best model epoch:",epoch,"val loss is:",val_loss)

        

        print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')

        scheduler.step()

    #model.load_state_dict(best_model_wts)

    return model.eval(), history
model, history = train_model(

  model,

  trainX,trainy,

  validX,validy,

  seq_length,

  n_epochs=30, ## Training only on 30 epochs to save GPU time 

    

)
TestX = np.array(TS_selected.iloc[-118:-28:,:])

Testy = np.array(TS_selected.iloc[-28:,:])

TestX = Variable(torch.Tensor(TestX))

 
TestX_Diff = difference(TestX, interval)

TestX_Diff_Norm = scaler.transform(np.array(TestX_Diff).reshape(len(TestX_Diff),1))
######Prediction###############

model.load_state_dict(torch.load('best_model.pt'))

model.eval()



with torch.no_grad():

    seq_inp = TestX.to(device)

    

        

    seq_pred = model(seq_inp,seq_inp[seq_length-1:seq_length,:])

          

data_predict = scaler.inverse_transform(seq_pred.cpu().numpy())

labels = Testy
orignal_data = np.array(TS_selected.iloc[-118:-28:,:]).flatten()

pred = data_predict.flatten()

orig_pred = np.concatenate([orignal_data,pred])

time_diff = 7 

inv_pred = np.zeros((28))

for  index in range(28):

    inv_pred[index] = orig_pred[90+index]+ orig_pred[90+index-7]

    orig_pred[90+index] = inv_pred[index]

data_predict= np.array(inv_pred)

data_predict = np.where(data_predict<0,0,data_predict)

## Add dates

df_predict = pd.DataFrame(data_predict)

df_predict = df_predict.set_index([dates_list[-28:]])

df_labels = pd.DataFrame(labels)

df_labels = df_labels.set_index([dates_list[-28:]])



# Plot 

figure(num=None, figsize=(24, 6), dpi=80, facecolor='w', edgecolor='k')

#plt.axvline(x=dates_list[train_size], c='r')

plt.plot( df_labels[0])

plt.plot(df_predict[0])

plt.legend(['Time Series','Prediction'],fontsize = 21)

plt.suptitle('Time-Series Prediction Last 28 Days',fontsize = 23)

plt.xticks(fontsize=21 )

plt.yticks(fontsize=21 )

plt.ylabel(ylabel = 'Sales Demand',fontsize = 21)

plt.xlabel(xlabel = 'Date',fontsize = 21)

plt.show()

np.sqrt(((data_predict - labels) ** 2).mean())
DF = TS_selected

colnames = DF.columns

DF = DF.rename(columns={colnames[0]:'sales'})

DF.tail()
## Simple Difference transform  --> X(t)' = X(t) - X(t-7)



def difference(data, interval):

    return [data[i] - data[i - interval] for i in range(interval, len(data))]



transformed_data = difference(DF['sales'], interval)

DF = DF.iloc[7:,:]

DF['orig_sales'] = DF['sales']

DF['sales'] = transformed_data

DF.shape
start_time = time.time()

for i in (1,7,14,28,365):

    print('Shifting:', i)

    DF['lag_'+str(i)] = DF['sales'].transform(lambda x: x.shift(i))

print('%0.2f min: Time for bulk shift' % ((time.time() - start_time) / 60))

for i in [7,14,28,60,180,365]:

    print('Rolling period:', i)

    DF['rolling_mean_'+str(i)] = DF['orig_sales'].transform(lambda x: x.shift(28).rolling(i).mean())

    DF['rolling_std_'+str(i)]  = DF['orig_sales'].transform(lambda x: x.shift(28).rolling(i).std())





print('%0.2f min: Time for loop' % ((time.time() - start_time) / 60))

DF.head()
DF = DF.replace('nan', np.nan).fillna(0)

DF.head()
DF = DF.drop('orig_sales',1)

DF_normlized = DF.copy(deep=True)

scaler = MinMaxScaler(feature_range=(-1, 1))

y_scaler = MinMaxScaler(feature_range=(-1, 1))

scaled_data = scaler.fit_transform(DF) 

y_scaler.fit_transform(DF['sales'].values.reshape(-1, 1))

DF_normlized.iloc[:,:] =  scaled_data

   



DF_normlized.head()
train_size = int((DF_normlized.shape[0]-28) * 0.67)

valid_size = (DF_normlized.shape[0]-28)- train_size

print("train size is:",train_size)

print("validation size is:",valid_size)

train_data = DF_normlized.iloc[0:train_size,:]

valid_data = DF_normlized.iloc[train_size:train_size+valid_size,:]

print("train data shape is:",train_data.shape)

print("validation data shape is:",valid_data.shape)

###  This function creates a sliding window or sequences of seq_length days and labels_length  days label ####

def sliding_windows(data, seq_length,labels_length):

    x = []

    y = []

    z = []



    for i in range(len(data)-(seq_length+labels_length)):

        _x = data.iloc[i:(i+seq_length),:]

        _y = data.iloc[(i+seq_length):(i+seq_length+labels_length),0:1]

        _z  = data.iloc[(i+seq_length):(i+seq_length+labels_length),1:]

        x.append(np.array(_x))

        y.append(np.array(_y))

        z.append(np.array(_z))



    return x,y,z
seq_length = 90

labels_length =28

train_X, train_y,train_features = sliding_windows(train_data, seq_length,labels_length)

print("train X  has:", len(train_X) , "series")

print("train labels  has:", len(train_y) , "series")

valid_X, valid_y,valid_features = sliding_windows(valid_data, seq_length,labels_length)

print("validiation  X  has:", len(valid_X) , "series")

print("Validiation  labels  has:" ,len(valid_y) , "series")
train_X[0].shape
train_y[0].shape
train_features[0].shape
trainX = Variable(torch.Tensor(train_X))

trainy = Variable(torch.Tensor(train_y))

train_features = Variable(torch.Tensor(train_features))

validX = Variable(torch.Tensor(valid_X))

validy= Variable(torch.Tensor(valid_y))

valid_features = Variable(torch.Tensor(valid_features))





print ("trainX shape is:",trainX.size())

print ("trainy shape is:",trainy.size())

print ("train features  shape is:",train_features.size())

print ("validX shape is:",validX.size())

print ("validy shape is:",validy.size())

print ("valid features  shape is:",valid_features.size())

class Encoder(nn.Module):

    def __init__(self, seq_len, n_features, embedding_dim=64):

        super(Encoder, self).__init__()



        self.seq_len, self.n_features = seq_len, n_features

        self.embedding_dim, self.hidden_dim = embedding_dim,  embedding_dim

        self.num_layers = 3

        self.rnn1 = nn.LSTM(

          input_size=n_features,

          hidden_size=self.hidden_dim,

          num_layers=3,

          batch_first=True,

          dropout = 0.35

        )

   

    def forward(self, x):

       

        x = x.reshape((1, self.seq_len, self.n_features))

        

        h_1 = Variable(torch.zeros(

            self.num_layers, x.size(0), self.hidden_dim).to(device))

         

        

        c_1 = Variable(torch.zeros(

            self.num_layers, x.size(0), self.hidden_dim).to(device))

              

        x, (hidden, cell) = self.rnn1(x,(h_1, c_1))

        

        

        #return hidden_n.reshape((self.n_features, self.embedding_dim))

        return hidden , cell 
class Decoder(nn.Module):

    def __init__(self, seq_len, input_dim=64, n_features=1):

        super(Decoder, self).__init__()



        self.seq_len, self.input_dim = seq_len, input_dim

        self.hidden_dim, self.n_features =  input_dim, n_features

        

        self.rnn1 = nn.LSTM(

          input_size=n_features,

          hidden_size=input_dim,

          num_layers=3,

          batch_first=True,

          dropout = 0.35

        )

        

        

      

        self.output_layer = nn.Linear(self.hidden_dim, n_features)



    def forward(self, x,input_hidden,input_cell):

       

       

        x = x.reshape((1,1,self.n_features ))

        #print("decode input",x.size())

             



        x, (hidden_n, cell_n) = self.rnn1(x,(input_hidden,input_cell))

    

        x = self.output_layer(x)

        return x, hidden_n, cell_n
class Seq2Seq(nn.Module):



    def __init__(self, seq_len, n_features, embedding_dim=64,output_length = 28):

        super(Seq2Seq, self).__init__()



        self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)

        self.n_features = n_features

        self.output_length = output_length

        self.decoder = Decoder(seq_len, embedding_dim, n_features).to(device)

        



    def forward(self,x, prev_y,features):

        

       

        hidden,cell = self.encoder(x)

         

        #Prepare place holder for decoder output

        targets_ta = []

        #prev_output become the next input to the LSTM cell

        dec_input = prev_y

        

        

        

       #dec_input = torch.cat([prev_output, curr_features], dim=1) 

        

        #itearate over LSTM - according to the required output days

        for out_days in range(self.output_length) :

            

          

            prev_x,prev_hidden,prev_cell = self.decoder(dec_input,hidden,cell)

            hidden,cell = prev_hidden,prev_cell

            

            prev_x = prev_x[:,:,0:1]

            #print("preve x shape is:",prev_x.size())

            #print("features shape is:",features[out_days+1].size())

            

            if out_days+1 < self.output_length :

                dec_input = torch.cat([prev_x,features[out_days+1].reshape(1,1,17)], dim=2) 

            

            targets_ta.append(prev_x.reshape(1))

           

            

        

        

        targets = torch.stack(targets_ta)



        return targets
n_features = trainX.shape[2]

model = Seq2Seq(seq_length, n_features, 512)

model = model.to(device)

model
def init_weights(m):

    for name, param in m.named_parameters():

        nn.init.uniform_(param.data, -0.08, 0.08)

        

model.apply(init_weights)
def train_model(model, TrainX,Trainy,ValidX,Validy,seq_length, n_epochs):

  

    history = dict(train=[], val=[])



    #best_model_wts = copy.deepcopy(model.state_dict())

    best_loss = 10000.0

    mb = master_bar(range(1, n_epochs + 1))



    for epoch in mb:

        model = model.train()



        train_losses = []

        for i in progress_bar(range(TrainX.size()[0]),parent=mb):

            seq_inp = TrainX[i,:,:].to(device)

            seq_true = Trainy[i,:,:].to(device)

            features = train_features[i,:,:].to(device)

           

            optimizer.zero_grad()



            

            seq_pred = model(seq_inp,seq_inp[seq_length-1:seq_length,:],features)

            

            

            loss = criterion(seq_pred, seq_true)



            loss.backward()

            

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            optimizer.step()



            train_losses.append(loss.item())



        val_losses = []

        model = model.eval()

        with torch.no_grad():

            for i in progress_bar(range(validX.size()[0]),parent=mb):

                seq_inp = ValidX[i,:,:].to(device)

                seq_true = Validy[i,:,:].to(device)

                features = valid_features[i,:,:].to(device)

        

                seq_pred = model(seq_inp,seq_inp[seq_length-1:seq_length,:],features)

               



                loss = criterion(seq_pred, seq_true)

                val_losses.append(loss.item())



        train_loss = np.mean(train_losses)

        val_loss = np.mean(val_losses)



        history['train'].append(train_loss)

        history['val'].append(val_loss)



        if val_loss < best_loss:

            best_loss = val_loss

            torch.save(model.state_dict(), 'best_model_n_features.pt')

            print("saved best model epoch:",epoch,"val loss is:",val_loss)

        

        print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')

        scheduler.step(val_loss)

    #model.load_state_dict(best_model_wts)

    return model.eval(), history
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2,weight_decay=1e-5)

criterion = torch.nn.MSELoss().to(device) 

#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 5e-3, eta_min=1e-8, last_epoch=-1)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,  patience=10, factor =0.5 ,min_lr=1e-7, eps=1e-08)
model, history = train_model(

  model,

  trainX,trainy,

  validX,validy,

  seq_length,

  n_epochs=10, #Train for few epochs as illustration, 

    

)
TestX = np.array(TS_selected.iloc[-700:-28:,:]).flatten()

Testy = np.array(TS_selected.iloc[-28:,:])





TestX.shape


def difference(data, interval):

    return [data[i] - data[i - interval] for i in range(interval, len(data))]



transformed_data = difference(TestX, interval)

DF = pd.DataFrame()

DF['sales'] = transformed_data

DF['orig_sales'] = TestX[7:]

DF.head(10)
for i in (1,7,14,28,365):

    print('Shifting:', i)

    DF['lag_'+str(i)] = DF['sales'].transform(lambda x: x.shift(i))

print('%0.2f min: Time for bulk shift' % ((time.time() - start_time) / 60))

for i in [7,14,28,60,180,365]:

    print('Rolling period:', i)

    DF['rolling_mean_'+str(i)] = DF['orig_sales'].transform(lambda x: x.shift(28).rolling(i).mean())

    DF['rolling_std_'+str(i)]  = DF['orig_sales'].transform(lambda x: x.shift(28).rolling(i).std())





print('%0.2f min: Time for loop' % ((time.time() - start_time) / 60))

DF.head()
DF = DF.drop('orig_sales', 1)





DF_normlized = DF.copy(deep=True)

scaler = MinMaxScaler(feature_range=(-1, 1))

y_scaler = MinMaxScaler(feature_range=(-1, 1))

scaled_data = scaler.fit_transform(DF) 

y_scaler.fit_transform(DF['sales'].values.reshape(-1, 1))

DF_normlized.iloc[:,:] =  scaled_data

DF_normlized = DF_normlized.iloc[-90:,:]

TestX = Variable(torch.Tensor(np.array(DF_normlized)))
TestX
######Prediction###############

model.load_state_dict(torch.load('../input/seq2seq-simple-model/best_model_n_features.pt'))

model.eval()



with torch.no_grad():

    seq_inp = TestX.to(device)

    

        

    seq_pred = model(seq_inp,seq_inp[seq_length-1:seq_length,:],seq_inp[:,1:])

          

seq_pred.shape
data_predict = y_scaler.inverse_transform(seq_pred.cpu().numpy())

labels = Testy
orignal_data = np.array(TS_selected.iloc[-118:-28:,:]).flatten()

pred = data_predict.flatten()

orig_pred = np.concatenate([orignal_data,pred])

time_diff = 7 

inv_pred = np.zeros((28))

for  index in range(28):

    inv_pred[index] = orig_pred[90+index]+ orig_pred[90+index-7]

    orig_pred[90+index] = inv_pred[index]

data_predict
data_predict= np.array(inv_pred)

#data_predict = np.where(data_predict<0,0,data_predict)

## Add dates

df_predict = pd.DataFrame(data_predict)

df_predict = df_predict.set_index([dates_list[-28:]])

df_labels = pd.DataFrame(labels)

df_labels = df_labels.set_index([dates_list[-28:]])



# Plot 

figure(num=None, figsize=(23, 6), dpi=80, facecolor='w', edgecolor='k')

#plt.axvline(x=dates_list[train_size], c='r')

plt.plot( df_labels[0])

plt.plot(df_predict[0])

plt.legend(['Time Series','Prediction'],fontsize = 21)

plt.suptitle('Time-Series Prediction 28 Days',fontsize = 23)

plt.xticks(fontsize=21 )

plt.yticks(fontsize=21 )

plt.ylabel(ylabel = 'Sales Demand',fontsize = 21)

plt.xlabel(xlabel = 'Date',fontsize = 21)

plt.show()

np.sqrt(((data_predict - labels) ** 2).mean())