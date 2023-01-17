import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import clear_output
import numpy as np
import matplotlib.pyplot as plt
device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import random
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_evaluation.csv')
date_features = pd.read_csv('/kaggle/input/m5features/date_features.csv')
sales_prices = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')
ann_features = pd.read_csv('/kaggle/input/m5features/ann_features.csv')
df['id'] = df['id'].str[:-11]
sales_prices['sell_price'] = MinMaxScaler().fit_transform(sales_prices['sell_price'].to_numpy().reshape(-1,1))
sales_prices['id'] = sales_prices['item_id'] + "_"  + sales_prices['store_id'] 
sales_prices.drop(['item_id','store_id'], axis =1, inplace = True)
date_features.head()
ann_features.head()
#std_x , mean_x ....skew_x are at the item level
#std_y , mean_y and skew_y are at the category level we can add more such encodings if needed at the store level and state level
#Help : Suggest me more aggregate properties of the time series
class MinMaxtransformer():
    ''' A class to scale the time series data for each item_id'''
    def __init__(self,d_x,d_y, info = None):
        self.d_x = d_x
        self.d_y = d_y
        if info is None :
            self.info = pd.DataFrame({'id': [],'min':[],'max':[]})
        else :
            self.info = info
    
    def fit(self, df):
        '''Will store in min and max values of the rows in a info dataframe'''
        self.info['id'] = df['id']
        self.info['max']= df.loc[:,self.d_x:self.d_y].max(axis=1)
        self.info['min']= df.loc[:,self.d_x:self.d_y].min(axis=1)
        self.info['maxdiffmin'] = self.info['max'] - self.info['min']
    
    def transform(self , df, d_x = None ,d_y = None):
        if d_x == None or d_y == None :
            d_x = self.d_x
            d_y = self.d_y
        df = pd.merge(df,self.info, on ='id', how ='left')
        for col in df.loc[:,d_x:d_y].columns:
            df[col] = (df[col] - df['min'])/(df['maxdiffmin'])
        df.drop(['min','max', 'maxdiffmin'],axis =1, inplace = True)
        return df
    
    def reverse_transform(self, df, d_x =None,d_y = None, round_ = False):
        df = pd.merge(df,self.info, on ='id', how ='left')
        if d_x == None or d_y == None :
            d_x = self.d_x
            d_y = self.d_y
        for col in df.loc[:,d_x:d_y].columns:
            df[col] = df[col] * df['maxdiffmin'] + df['min']
            if round_ :
                df[col] = round(df[col])
        df.drop(['min','max', 'maxdiffmin'],axis =1, inplace = True)
        return df
    
mmt  = MinMaxtransformer('d_1','d_1913')
mmt.fit(df)
from sklearn.model_selection import train_test_split
trainids, testids = train_test_split(df.loc[:,['id']], train_size = 0.8, random_state = 1234)
trainids = trainids['id'].to_list()
testids = testids['id'].to_list()
len(trainids)
len(testids)
class M5dataloader_v2():
    def __init__(self, ids, batch_size, estart = 'd_1', eend ='d_1913', dstart = 'd_1914', dend = 'd_1941'):
        '''IDs to be passed in list format'''
        self.ids = ids
        self.iteration = 0
        self.batch_size = batch_size
        self.estart = estart
        self.eend = eend
        self.dstart = dstart
        self.dend = dend
    
    def get_data(self, df,date_features,sales_prices, ann_features, device = device):
        start = (self.iteration * self.batch_size)% len(self.ids)
        end = start + self.batch_size
        self.iteration +=1
        if( end < len(self.ids)):
            batchidlist = self.ids[start:end]
        else :
            end = end%len(self.ids)
            batchidlist = [id for id in self.ids[start:]] + [ id for id in self.ids[:end]]
        filt = df['id'].isin(batchidlist)
        batch_data = df.loc[filt,:].drop(['item_id','dept_id','cat_id', 'store_id','state_id'], axis = 1 )
        batch_data  =  mmt.transform(batch_data,'d_1', 'd_1941')
        batch_data.set_index('id', inplace = True)
        
        
        estart = self.estart
        eend = self.eend
        dstart = self.dstart
        dend = self.dend
        # Encoder_input tensor generation
        
        encoder_data = batch_data.loc[:,estart:eend]
        encoder_data.reset_index(inplace = True)
        encoder_data = encoder_data.melt(id_vars =['id'], value_vars = encoder_data.columns.to_list()[1:],var_name ='d', value_name ='count')
        encoder_data = pd.merge(encoder_data,date_features,how = 'left',on ='d').drop(['Unnamed: 0'], axis =1)
        encoder_data = pd.merge(encoder_data,sales_prices,how = 'left',on =['id','wm_yr_wk']).drop(['date','wm_yr_wk'], axis =1)
        encoder_data['d']= encoder_data['d'].str[2:].apply(int)
        encoder_data = encoder_data.dropna(axis = 0)
        encoder_data = pd.DataFrame( { 'id': encoder_data['id'], 'd': encoder_data['d'],'Encoder_input_vector' : encoder_data.iloc[:,2:].to_numpy().tolist()})\
        .pivot(index ='id' , columns= 'd', values= 'Encoder_input_vector')
        index_vals = encoder_data.isnull().sum(axis =1).sort_values(ascending = True).index
        encoder_data = encoder_data.reindex(index_vals)
        l =[]
        for i in range(self.batch_size):
            t = torch.tensor(encoder_data.iloc[i,:].dropna().tolist()).float()
            l.append(t)
        
        encoder_data = pack_sequence(l) # pack sequence from torch.nn.utils.rnn returns a packed sequence object
        
        
        #Decoder input tensor generation
        
        decoder_data = batch_data.loc[:,dstart:dend]
        decoder_data.reset_index(inplace = True)
        decoder_data = decoder_data.melt(id_vars =['id'], value_vars = decoder_data.columns.to_list()[1:],var_name ='d', value_name ='count')
        decoder_data = pd.merge(decoder_data, date_features, how = 'left' , on = 'd').drop(['Unnamed: 0'], axis =1)
        decoder_data = pd.merge(decoder_data,sales_prices,how = 'left',on =['id','wm_yr_wk']).drop(['date','wm_yr_wk'], axis =1)
        decoder_data['d'] = decoder_data['d'].str[2:].apply(int)
        ground_truth =  decoder_data.loc[:,['id','d','count']]
        decoder_data.drop(['count'], axis = 1, inplace = True)
        decoder_data = pd.DataFrame( { 'id': decoder_data['id'], 'd': decoder_data['d'],'Decoder_input_vector' : decoder_data.iloc[:,2:].to_numpy().tolist()})\
        .pivot(index ='id' , columns= 'd', values= 'Decoder_input_vector')
        decoder_data = decoder_data.reindex(index_vals)
        decoder_data = torch.tensor(np.array(decoder_data.values.tolist())).transpose(0,1)
        decoder_data = decoder_data.float()
        
    
        ground_truth = ground_truth.pivot(index = 'id',   columns ='d', values = 'count')
        ground_truth = ground_truth.reindex(index_vals)
        ground_truth = torch.tensor(ground_truth.to_numpy().reshape(self.batch_size ,28,1)).transpose(0,1)
        ground_truth = ground_truth.float()
      
    
        
        # ann_data
        filt2 = ann_features['id'].isin(batchidlist)
        ann_data = ann_features.loc[filt2,:]
        ann_data.set_index('id', inplace = True)
        ann_data = ann_data.reindex(index_vals)
        ann_data = torch.tensor(ann_data.to_numpy().reshape(self.batch_size, ann_data.shape[1]))
        ann_data = ann_data.float()
        
        return (encoder_data.to(device), decoder_data.to(device),ground_truth.to(device), ann_data.to(device))
dataloader = M5dataloader_v2(trainids,10) # initialize it
encoder_data, decoder_data, ground_truth,ann_data = dataloader.get_data(df,date_features,sales_prices,ann_features, device ='cpu') 
# getting input dimensions of encoder , decoder and ann and testing the data loader
ann_input_size = ann_data.shape[1] 
enc_input_size= encoder_data.data.shape[1]
dec_input_size= decoder_data.shape[2]+1 # we add one to include output from previous state as input
print(ann_input_size , " ", enc_input_size, " ", dec_input_size)
#Lets look at the shapes of the tensor
print('\n ANN tensor shape :', ann_data.shape)
print('\n encoder tensor shape :', encoder_data.data.shape)
print('\n decoder tensor shape :', decoder_data.shape)
print('\n ground_truth tensor shape :', ground_truth.shape)

# Note Encoder tensor is a packed tensor
class EncoderRNN(nn.Module):
    def __init__(self, ann_input_size,enc_input_size, hidden_size, n_layers=1, dropout=0.1):
        super(EncoderRNN, self).__init__()
        
        self.ann_input_size = ann_input_size
        self.enc_input_size = enc_input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        
        
        
        self.ann_to_hidden =nn.Sequential(nn.Linear(ann_input_size,96), 
                                 nn.ReLU(),
                                 nn.Linear(96,hidden_size))
        self.gru = nn.GRU(enc_input_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True)
        
    def forward(self, ann_data, encoder_data):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        hidden = self.ann_to_hidden(ann_data)
        hidden = hidden.reshape(1, hidden.shape[0],hidden.shape[1]).repeat(self.n_layers*2,1,1)
        
        outputs, hidden = self.gru(encoder_data, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs) # unpack (back to padded)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:] # Sum bidirectional outputs
        return outputs, hidden
    
class M5_EncoderDecoder_v2(nn.Module):
    
    def __init__(self, ann_input_size,enc_input_size, dec_input_size, hidden_size, output_size = 1, verbose=False):
        super(M5_EncoderDecoder_v2, self).__init__()
        self.ann_input_size = ann_input_size
        self.enc_input_size = enc_input_size
        self.dec_input_size = dec_input_size + hidden_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        
        self.encoder_rnn_cell = EncoderRNN(ann_input_size,enc_input_size, hidden_size, n_layers=2, dropout=0.1)
        self.decoder_rnn_cell = nn.GRU(dec_input_size+hidden_size, hidden_size)
        
        self.h2o = nn.Linear(hidden_size, output_size)
        self.verbose = verbose
        self.U = nn.Linear(self.hidden_size, self.hidden_size)
        self.W = nn.Linear(self.hidden_size, self.hidden_size)
        self.V = nn.Linear(self.hidden_size,1)
        self.final = nn.ReLU()
    
        
    def forward(self, ann_data, encoder_data, decoder_data, ground_truth = None, steps = 28 , device = device):
    
        
        batch_size = decoder_data.shape[1]
        encoder_outputs,hidden = self.encoder_rnn_cell.forward(ann_data,encoder_data)
        
        U = self.U(encoder_outputs)
        initial_ground = torch.zeros(1,decoder_data.shape[1],1).to(device)
        flag = 1
        if ground_truth is None:
            ground_truth = initial_ground
            flag = 0
        else :
            ground_truth = torch.cat((initial_ground,ground_truth),0)
        
        hidden = torch.sum(hidden,dim =0).view(1,batch_size,-1)
        
        outputs = []
        for i in range(steps) :
            W  = net.W(hidden.repeat(encoder_outputs.shape[0],1,1))
            V= net.V(torch.tanh(U+W))
            alpha = F.softmax(V, dim=0)
            attn_applied = torch.bmm(alpha.T.transpose(0,1),encoder_outputs.transpose(0,1))
            if (i == 0):
                decoder_input = torch.cat((ground_truth[i,:,:].float(),decoder_data[i,:,:].float(),attn_applied.transpose(0,1)[0,:,:].float()),1)
            
            if(i > 0):
                if flag != 0 :
                    decoder_input = torch.cat((ground_truth[i,:,:].float(),decoder_data[i,:,:].float(),attn_applied.transpose(0,1)[0,:,:].float()),1)
                    
                else:
                    decoder_input = torch.cat((out[:,0].view(batch_size, 1).float(),decoder_data[i,:,:].float(),attn_applied.transpose(0,1)[0,:,:].float()),1) 
                    # no need to use i-1 as we have added a timestep inthe form of initial ground
            
            _ ,hidden = self.decoder_rnn_cell(decoder_input.view(1,decoder_input.shape[0],decoder_input.shape[1]).float(), hidden)
            out = self.h2o(hidden) 
            out = self.final(out)
            out = out.view(out.shape[1],self.output_size)
            outputs.append(out) # verify dimensions
        return outputs
def criterion_binomial_log_likehood(mu, alpha, truth):
    m = mu.reshape(-1)
    a = torch.sigmoid(alpha.reshape(-1))
    t = truth.reshape(-1)

    am = torch.clamp_min(a*m ,1e-8)

    likelihood = \
          torch.lgamma(t + 1. /a) \
        - torch.lgamma(t + 1) \
        - torch.lgamma(1. / a) \
        - 1. / a * torch.log(1 + am) \
        + t * torch.log(am/ (1 + am))

    loss = -likelihood.mean()
    return loss
def train_batch(net, batch,batch_size,opt,criterion,device, teacher_force = False):
    net.train().to(device)
    opt.zero_grad()
    encoder_data, decoder_data, ground_truth,ann_data = batch
    ground_truth = ground_truth.to(device)
    if teacher_force :
        outputs = net.forward(ann_data, encoder_data, decoder_data, ground_truth)
    else :
        outputs = net.forward(ann_data,encoder_data,decoder_data)
    
    loss = torch.zeros(1,1).float().to(device)
    for i, output in enumerate(outputs):
        loss += criterion(output[:,0], output[:,1],ground_truth[i,:,:]).float()
    loss.backward()
    opt.step()
    
    return loss/batch_size
def prediction(testids, net, df, sales_prices, date_features, ann_features,round_ = False, steps = 28,idstart = 1914):
    testloader = M5dataloader_v2(testids,len(testids))
    encoder_data, decoder_data, ground_truth,ann_data = testloader.get_data(df,date_features,sales_prices,ann_features)
    outputs = net.forward(ann_data,encoder_data,decoder_data)
    pred= pd.DataFrame({'id' : testids})
    for i, output in enumerate(outputs):
        pred['d_' + str(idstart + i)] = output[:,0].cpu().data.numpy()
    
    start = 'd_' + str(idstart)
    end = 'd_' + str(idstart + 27) 
    pred = mmt.reverse_transform(pred,start,end, round_ = round_)
    pred.set_index('id', inplace = True)
    return pred
def actual_values(testids,df,steps = 28, idstart = 1914):
    df.set_index('id',inplace = True)
    start = 'd_' + str(idstart)
    end = 'd_' + str(idstart + steps -1) 
    act = df.loc[testids,start:end]
    df.reset_index(inplace = True)
    return act
def validation_error(pred,act):
    return np.square(pred.to_numpy() - act.to_numpy()).sum()/act.to_numpy().size
def get_plots(ids, net):
    if len(ids) > 25:
        return "the number of ids in the list exceeds the limit of 25"
    
    trainhead = actual_values(ids, df,steps = 1000 , idstart = 886).T
    testvalues = actual_values(ids, df,steps = 28 , idstart = 1914).T
    predictions = prediction(ids, net, df, sales_prices, date_features, ann_features, round_ = False).T
    b = ['_encoder','_actual', '_pred']
    for i , x in enumerate([trainhead, testvalues, predictions]):
        x.columns = [x for x in map(lambda x: x + b[i],x.columns.to_list())]
        x.reset_index(inplace = True)
        x.rename(columns={'index' : 'Days'}, inplace = True)
        x['Days'] = x['Days'].str[2:].apply(int)
        
    fig, ax = plt.subplots(nrows = len(ids), ncols = 1,figsize=(25,4 * len(ids)))
    for i in range(len(ids)):
        if len(ids) == 1:
            trainhead.plot(x = 'Days',y=[i+1],ax=ax);
            testvalues.plot(x = 'Days',y=[i+1],ax=ax);
            predictions.plot(x = 'Days',y=[i+1],ax=ax);
        else :
            trainhead.plot(x = 'Days',y=[i+1],ax=ax[i]);
            testvalues.plot(x = 'Days',y=[i+1],ax=ax[i]);
            predictions.plot(x = 'Days',y=[i+1],ax=ax[i]);
def get_submission(idlist,net, batch = 100):
    
    submission = []
    for i in range(len(idlist)//batch + int(len(idlist)%batch !=0)):
        print("Iteration ", i, "/", len(idlist)//batch + int(len(idlist)%batch !=0))
        start = i * batch
        end = start + batch
        if(end > len(idlist)): 
            end = len(idlist)
        batchidlist = idlist[start:end] 
        pred = prediction(batchidlist, net, df, sales_prices, date_features, ann_features, round_ = False)
        submission.append(pred) 
    return pd.concat(submission, axis =0)
def train_setup(net,trainids,testids,validation = False,plots = False, lr = 0.01, n_batches = 5, batch_size = 140, momentum = 0.9, display_freq=1, device = device,test_batch_size = 100, teacher_upto =0 ):
    
    net = net.to(device)
    criterion = criterion_binomial_log_likehood # changed from nn.MSELoss
    opt = optim.Adam(net.parameters(), lr=lr)
    teacher_force_upto = n_batches//3
    trainloader = M5dataloader_v2(trainids,batch_size)
    loss_arr = np.zeros(n_batches + 1)
    if validation :
        valid_error = []
        valid_xaxis =[]
    for i in range(n_batches):
        batch = trainloader.get_data(df, date_features, sales_prices, ann_features) 
        loss_arr[i+1] = (loss_arr[i]*i + train_batch(net,batch,batch_size, opt, criterion, device = device, teacher_force = (i/n_batches < teacher_upto) ))/(i + 1)
        
        if i%display_freq == display_freq-1:
            if plots :
                clear_output(wait=True)
            
            if validation :
                ids = random.sample(testids,test_batch_size)
                pred = prediction(ids, net, df, sales_prices, date_features, ann_features, round_ = False)
                act = actual_values(ids, df,steps = 28 , idstart = 1914)
                valid_xaxis.append(i)
                v = validation_error(pred,act)
                valid_error.append(v)
                print('Validation error  ', v, " per observation as tested on ", test_batch_size, " random samples from test set")
            print('Epoch ',(i*batch_size)/len(trainids),' Iteration', i, 'Loss ', loss_arr[i])
            
            if plots:
                fig, axes = plt.subplots(nrows =1, ncols=2 , figsize=(20,6))
                axes[0].set_title('Train Error')
                axes[0].plot(loss_arr[2:i], '-*')
                axes[0].set_xlabel('Iteration')
                axes[0].set_ylabel('Loss')

                if validation :
                    axes[1].set_title('Validation error')
                    axes[1].plot(valid_xaxis, valid_error,'-*')
                    axes[1].set_xlabel('Iteration')
                    axes[1].set_ylabel('validation error')
                plt.show()
                print('\n\n')
        if(i%100 ==0):    
            torch.save(net.state_dict(), 'model.pt')
        if(i%500 ==0):
            filename = str(i) + 'model.pt'
            torch.save(net.state_dict(), filename)
    filename = str(i) + 'model.pt'
    torch.save(net.state_dict(), filename)    
    return (loss_arr,net)
validation = True ; plots = True ; lr = 0.01 ; n_batches = 1000 ; batch_size = 140; momentum = 0.9; display_freq=5 ; device = device ; test_batch_size = 100 ; teacher_upto = 0.5
#Training cell

net = M5_EncoderDecoder_v2(ann_input_size = 25 ,enc_input_size= 30, dec_input_size= 30, hidden_size=126,output_size= 2)
losses,net = train_setup(net,trainids,testids,validation,plots, lr, n_batches, batch_size, momentum , display_freq, device,test_batch_size, teacher_upto)


# I have already trained the model batch_size = 140 , iterations = 2000, learning_rate = 0.01 , forgot teacher enforcing guess would have learned faster
# No more GPU quota :(  sad life!! 
# the train_setup function supports plots to monitor loss and also validation feel free to play with it

#We will load my pretrained model
#net = M5_EncoderDecoder_v2(ann_input_size = 25 ,enc_input_size= 30, dec_input_size= 30, hidden_size=256,output_size = 2 )
#net.load_state_dict(torch.load('../input/m5trained-models/1500model.pt', map_location = device))
#net.eval()
#net.to(device)
#get_plots(random.sample(trainids,20),net)
#idlist = df['id'].to_list()
#submission_validation = get_submission(idlist,net)
#submission_validation.to_csv("1500submission.csv")
#cant submit this submission file coz its not in submission format I appended evaluation with 0's to it and submitted to get an WRMSE of 0.603
device

