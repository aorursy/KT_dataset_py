#!pip install iterative-stratification
import sys

sys.path.append('../input/iterative-stratification/iterative-stratification-master')

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.model_selection import KFold

from torchvision import datasets, models, transforms

from torch.utils.data import Dataset, DataLoader

from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch

from torch import nn, optim

import random , os

import numpy as np

import pandas as pd

from sklearn.metrics import log_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device
import warnings

warnings.filterwarnings('ignore')
def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

    

seed_everything(42)
df_train_features = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')

df_train_targets_scored = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')

df_test_features = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')



print('df_train_features Shape',df_train_features.shape)

print('df_train_targets_scored Shape',df_train_targets_scored.shape)

print('df_test_features Shape',df_test_features.shape)
df_train = pd.merge(df_train_features,df_train_targets_scored,on='sig_id')

df_train = df_train[df_train['cp_type']!= 'ctl_vehicle'].reset_index(drop=True)

#df_test_features = df_test_features[df_test_features['cp_type']!='ctl_vehicle'].reset_index(drop=True)



df_train.shape
## Train

df_train['cp_type'] = df_train['cp_type'].map({'trt_cp': 0, 'ctl_vehicle': 1})

df_train['cp_dose'] = df_train['cp_dose'].map({'D1': 3, 'D2': 4})

df_train['cp_time'] = df_train['cp_time'].map({24: 0, 48: 1, 72: 2})



## Test

df_test_features['cp_type'] = df_test_features['cp_type'].map({'trt_cp': 0, 'ctl_vehicle': 1})

df_test_features['cp_dose'] = df_test_features['cp_dose'].map({'D1': 3, 'D2': 4})

df_test_features['cp_time'] = df_test_features['cp_time'].map({24: 0, 48: 1, 72: 2})
x_cols = df_train_features.columns[1:]

y_cols = df_train_targets_scored.columns[1:]



print('Features',len(x_cols))

print('Labels',len(y_cols))
### For K-fold Validation

df_train['folds'] = -1

df_train = df_train.sample(frac=1).reset_index(drop=True)

splits = 5

kf = KFold(n_splits=splits,shuffle = False)

for fold,(train_index, val_idx) in enumerate(kf.split(df_train)):

    #df.iloc[train_index,:]['kfold'] = int(fold+1)

    df_train.loc[val_idx,'folds'] = int(fold)



print('Number of Unique folds in dataset',df_train['folds'].unique())
class Train_Dataset(Dataset):

    def __init__(self, dataframe,features_col,labels ):

        

        self.dataframe = dataframe

        self.features_col = features_col

        self.labels = labels

        

                

        self.x = self.dataframe[self.features_col].values

        self.y = self.dataframe[self.labels].values

        

    def __len__(self):

        return self.dataframe.shape[0]

    

    def __getitem__(self, idx):

        

        feat = torch.FloatTensor(self.x[idx])

        lab = torch.FloatTensor(self.y[idx])

        

        return feat,lab

    

class Test_Dataset(Dataset):

    def __init__(self, dataframe,features_col):

        

        self.dataframe = dataframe

        self.features_col = features_col

                

                

        self.x = self.dataframe[self.features_col].values

        

        

    def __len__(self):

        return self.dataframe.shape[0]

    

    def __getitem__(self, idx):

        

        feat = torch.FloatTensor(self.x[idx])

        

        

        return feat
class MultiHead_Attn(nn.Module):

    def __init__(self, embed_dim ,num_head ):        

        super().__init__()

        

        #self.source_input_dim = source_input_dim ## 875

        self.embed_dim = embed_dim ##1024

        self.num_head = num_head ##8

        self.head_dim = self.embed_dim // self.num_head

        

        

        assert self.embed_dim%self.num_head == 0

        

        self.q = nn.Linear(self.embed_dim , self.embed_dim )

        self.k = nn.Linear(self.embed_dim , self.embed_dim )

        self.v = nn.Linear(self.embed_dim , self.embed_dim)

        

        self.f_linear = nn.Linear(self.embed_dim, self.embed_dim)

        self.dropout = nn.Dropout(.35)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    

    def forward(self, x ):

        batch_size = x.shape[0]

        src_len    = x.shape[1]

        

        w_k = self.k(x)

        w_q = self.q(x)        

        w_v = self.v(x)

        

        w_q = w_q.view(batch_size,-1,self.head_dim)

        w_k = w_k.view(batch_size,-1,self.head_dim)

        w_v = w_v.view(batch_size,-1,self.head_dim)

         

        energy = torch.matmul( w_k.permute(0,2,1) ,w_q )

        energy = energy/self.scale

        energy = torch.softmax(energy,-1)

        

        

        f_energy = torch.matmul( self.dropout(energy) , w_v.permute(0,2,1))

        f_energy = f_energy.permute(0, 2, 1)

        f_energy = f_energy.reshape(batch_size,-1)

        out = self.f_linear(f_energy)

        

        return out
class PositionwiseFeedforwardLayer(nn.Module):

    def __init__(self, embed_dim, pf_dim):

        super().__init__()

        

        self.fc_1 = nn.Linear(embed_dim, pf_dim)

        self.fc_2 = nn.Linear(pf_dim, embed_dim)

        

        self.dropout = nn.Dropout(.35)

        

    def forward(self, x):

        

        x = self.dropout(torch.relu(self.fc_1(x)))

       

        x = self.fc_2(x)

           

        return x
class TransformerBlock(nn.Module):

    def __init__(self, embed_dim,num_heads, pf_dim):

        super().__init__()

        

        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)

        

        self.ff_layer_norm = nn.LayerNorm(embed_dim)

        

        self.attn = MultiHead_Attn( embed_dim ,num_heads )

        

        self.positionwise_feedforward = PositionwiseFeedforwardLayer(embed_dim,pf_dim)

        

        self.dropout = nn.Dropout(.35)

        

    def forward(self,src):

        

        attn = self.attn(src)

        src = self.self_attn_layer_norm(src + self.dropout(attn))

        

        _src = self.positionwise_feedforward(src)

        src = self.ff_layer_norm(src + self.dropout(_src))

        

        return src
class Encorder(nn.Module):

    def __init__(self, src_dim , embed_dim ,num_heads , pf_dim , target_dim ,depth):

        super(Encorder, self).__init__()

        

        self.embed_layer = nn.Linear( src_dim , embed_dim )

        

        self.Transformer_layers = TransformerBlock( embed_dim , num_heads ,pf_dim )

        

        tblocks = []

        

        for i in range(depth):

            tblocks.append(self.Transformer_layers)

            

        self.tblocks = nn.Sequential(*tblocks)

        

        self.dropout = nn.Dropout(.35)

        

        self.final_linear = nn.Sequential(

                                nn.utils.weight_norm(nn.Linear(embed_dim , embed_dim //2))

                                ,nn.BatchNorm1d(embed_dim //2)

                                ,nn.Dropout(.35)

                                ,nn.ReLU()

                                ,nn.Linear(embed_dim //2 , target_dim))

            

        

        

    def forward(self , x):

        

        batch_size,seq_len = x.shape

        embed = self.embed_layer(x)

        

        for layer in self.tblocks:            

            x = self.dropout(self.Transformer_layers( embed ))

            

        out = self.final_linear(x)

            

        return out

        
'''

src_dim = 875

embed_dim = 1024

num_heads = 8

pf_dim = 512

target_dim = 206

depth = 3

enc = Encorder(src_dim , embed_dim ,num_heads , pf_dim , target_dim ,depth)

'''
def train(loader, model , optimizer , loss_func ,scheduler ):

    

    model.train()

    train_running_loss = 0

    for index,(feat,label) in enumerate(loader):

        x,y = feat.to(device),label.to(device)

        

        # Zero the parameter gradients

        optimizer.zero_grad()

        pred = model(x)

        loss = loss_func(pred,y)

        loss.backward()

        optimizer.step()

        train_running_loss += loss.item()

        

    return train_running_loss/len(loader)





def valid( loader, model , loss_func ):

    

    

    valid_running_loss = 0

    pred_list = list()

    with torch.no_grad():

        model.eval()

        for val_index,(feat,label) in enumerate(loader):

            x,y = feat.to(device),label.to(device)

            

            # Zero the parameter gradients

            pred = model(x)

            pred_list.append(pred)

            loss = loss_func(pred,y)

            valid_running_loss += loss.item()

            

    return torch.cat(pred_list).sigmoid().detach().cpu().numpy(),valid_running_loss/len(loader)







def test(loader,model):

    labels = []

    with torch.no_grad():

        model.eval()

        for index,feat in enumerate(loader):

            x = feat.to(device)

            output = model(x)

            labels.append(output.sigmoid().detach().cpu().numpy())

    

    labels = np.concatenate(labels)        

    

    return labels
def train_run(seed , Batch_size , src_dim ,embed_dim , num_heads , pf_dim ,target_dim , depth , n_epoch , patience ,  learning_rate ):

    

    val_preds = np.zeros((df_train[y_cols].shape))

    test_pred = np.zeros((len(df_test_features), y_cols.shape[0]))



    for fold_num in range(splits):

        

        

        

        seed_everything(seed)

        print('seed value', seed )

        

        print('='*30,'*****','Fold',fold_num+1,'*****','='*30)

    

        trn_idx = df_train[df_train['folds'] != fold_num].index

        val_idx = df_train[df_train['folds'] == fold_num].index

        

        df_trn = df_train.loc[trn_idx].reset_index(drop=True)

        df_val = df_train.loc[val_idx].reset_index(drop=True)

        

        ### Train Dataset

        

        train_dataset = Train_Dataset(df_trn,x_cols,y_cols)

        train_loader = DataLoader(train_dataset, batch_size=Batch_size, shuffle=True)

        

        ### Valid Dataset

    

        valid_dataset = Train_Dataset(df_val,x_cols,y_cols);

        valid_loader = DataLoader(valid_dataset, batch_size=Batch_size, shuffle=False)

    

        ## Defining Model  



        model = Encorder(src_dim , embed_dim ,num_heads , pf_dim , target_dim ,depth).to(device)

        

        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print('Number of parameters',pytorch_total_params)



        ## Defining optimizer and loss function

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay= 1e-5)

        scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='max', patience=3, verbose=True, factor=0.2)

        loss_func = nn.BCEWithLogitsLoss()

        

        best_score = np.inf

        counter = 0

        

        for epoch in range(n_epoch): 

            

            print("Epoch: {}/{}.. ".format(epoch+1, n_epoch))

            

            train_loss = train(train_loader, model , optimizer , loss_func ,scheduler )

            v_out,valid_loss = valid( valid_loader, model , loss_func )

            

            print(f'\tTrain Loss: {train_loss:.5f}')

            print(f'\t Val. Loss: {valid_loss:.5f}')

            

            if valid_loss < best_score:

                

                best_score = valid_loss

                val_preds[val_idx] = v_out

                torch.save(model.state_dict(), f"fold_{fold_num}_seed_{seed}.pth")

            else:

                print('patience starts .........')

                counter +=1

                print('..counter..',counter)

                if (counter >= patience):

                    break;

            

            scheduler.step(valid_loss)

            



        

        ### ----------------- Predictions Starts here------------------###

        

        print(f'Inferenceing the test data at epoch {epoch+1} and fold {fold_num+1} and seed {seed} .....')

        

        model = Encorder(src_dim , embed_dim ,num_heads , pf_dim , target_dim ,depth).to(device)

        model.load_state_dict(torch.load(f"fold_{fold_num}_seed_{seed}.pth"))

        

        test_dataset = Test_Dataset(df_test_features,x_cols)

        test_loader = DataLoader(test_dataset, batch_size=Batch_size, shuffle=False)   

        test_pred += test(test_loader,model)/splits 

        

        print(f'Inferenceing completed for fold {fold_num+1}')

    

    return test_pred , val_preds
### Initializing variables

random_seed = [33 , 66 , 99 , 132]



Batch_size = 32

src_dim = 875

embed_dim = 1024

num_heads = 8

pf_dim = 512

target_dim = 206

depth = 8

n_epoch = 30

patience = 10

learning_rate = .0005





test_pred = np.zeros((len(df_test_features), y_cols.shape[0]))

val_preds = np.zeros((df_train[y_cols].shape))



for seed in random_seed:

    tp , vp = train_run(seed , Batch_size , src_dim ,embed_dim , num_heads , pf_dim ,target_dim , depth , n_epoch , patience ,  learning_rate )

    

    test_pred += tp/len(random_seed)

    val_preds += vp/len(random_seed)
from sklearn.metrics import log_loss

score = 0

for i in range(df_train[y_cols].shape[1]):

    _score = log_loss(df_train[y_cols].iloc[:,i], val_preds[:,i])

    score += _score / df_train[y_cols].shape[1]

print(f"oof score: {score}")
sub = pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')

sub[y_cols] = test_pred

sub.loc[df_test_features['cp_type'] == 1 , y_cols ] = 0 
sub.head(10)
sub.to_csv('submission.csv', index=False)