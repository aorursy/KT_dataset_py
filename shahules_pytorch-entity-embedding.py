import riiideducation

import dask.dataframe as dd

import  pandas as pd

import numpy as np

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split

from tqdm import tqdm

from sklearn.preprocessing import RobustScaler



import torch

import torch.nn as nn

import torch.optim as optim

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader



import warnings

import gc

warnings.filterwarnings('ignore')

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")





train= pd.read_csv('/kaggle/input/riiid-test-answer-prediction/train.csv',

                usecols=[1, 2, 3,4,7,8,9], dtype={'timestamp': 'int64', 'user_id': 'int32' ,'content_id': 'int16','content_type_id': 'int8','answered_correctly':'int8',

                                                  'prior_question_elapsed_time': 'float32','prior_question_had_explanation': 'object'}

  

                  )


train = train[train.content_type_id == False]

train = train.sort_values(['timestamp'], ascending=True)



train.drop(['timestamp','content_type_id'], axis=1,   inplace=True)



train.head(3)
results_c = train[['content_id','answered_correctly']].groupby(['content_id']).agg(['mean','std','sum','skew'])

results_c.columns = ["content_mean","content_std","content_sum","content_skew"]



results_u = train[['user_id','answered_correctly']].groupby(['user_id']).agg(['mean', 'sum','std','skew'])

results_u.columns = ["user_mean", 'user_sum','user_std','user_skew']
#reading in question df

questions_df = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/questions.csv',

                            usecols=[0,1, 3,4],

                            dtype={'question_id': 'int16',

                              'part': 'int8','bundle_id': 'int8','tags': 'str'}

                          )
questions_df = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/questions.csv',

                            usecols=[0,1, 3,4],

                            dtype={'question_id': 'int16',

                              'part': 'int8','bundle_id': 'int8','tags': 'str'}

                          )

tag = questions_df["tags"].str.split(" ", n = 10, expand = True) 

tag.columns = ['tags1','tags2','tags3','tags4','tags5','tags6']



questions_df =  pd.concat([questions_df,tag],axis=1).drop(['tags'],axis=1)

questions_df['tags1'] = pd.to_numeric(questions_df['tags1'], errors='coerce',downcast='integer').fillna(-1)

questions_df['tags2'] = pd.to_numeric(questions_df['tags2'], errors='coerce',downcast='integer').fillna(-1)

questions_df['tags3'] = pd.to_numeric(questions_df['tags3'], errors='coerce',downcast='integer').fillna(-1)

#questions_df['tags4'] = pd.to_numeric(questions_df['tags4'], errors='coerce',downcast='integer').fillna(-1)

#questions_df['tags5'] = pd.to_numeric(questions_df['tags5'], errors='coerce',downcast='integer')

#questions_df['tags6'] = pd.to_numeric(questions_df['tags6'], errors='coerce',downcast='integer')
questions_df.head(3)
cat_columns = ['prior_question_had_explanation','bundle_id','part','tags1','tags2','tags3']



cont_columns = ['prior_question_elapsed_time', "content_mean","content_std","content_sum","content_skew",

                "user_mean", 'user_sum','user_std','user_skew']



X=train.iloc[89000000:,:]

X = pd.merge(X, results_u, on=['user_id'], how="left")

X = pd.merge(X, results_c, on=['content_id'], how="left")

X = pd.merge(X, questions_df, left_on = 'content_id', right_on = 'question_id', how = 'left')



X=X[X.answered_correctly!= -1 ]

X=X.sort_values(['user_id'])

X['prior_question_had_explanation']=X['prior_question_had_explanation'].fillna('False').map({"True":True,"False":False})

X['prior_question_elapsed_time'].fillna(0,inplace=True)



for col in cont_columns:

    X[col].fillna(X[col].mode(),inplace=True)



Y = X[["answered_correctly"]]

X = X.drop(["answered_correctly"], axis=1)


features=cat_columns+cont_columns



def encode(df,cols):

    enc =  {}

    for col in cols:

        print(col)

        lbencoder = LabelEncoder()

        lb = lbencoder.fit(df[col].values)

        df[col]=lb.transform(df[col].values)

        enc[col]=lb

        

    return df,enc



X,enc_dict = encode(X,cat_columns)
scale_dict={}

fix_missing={}

for col in cont_columns:

    scaler = RobustScaler()

    scale_dict[col]=scaler.fit(X[col].values.reshape(-1,1))

    X[col] = scale_dict[col].transform(X[col].values.reshape(-1,1))

    fix_missing[col] = X[col].mode()
cat_dims = [X[col].nunique() for col in cat_columns]

cat_embs = [(dim, min(50,(dim+1)//2)) for dim in cat_dims]
cat_embs
class RidDataset(Dataset):

    def __init__(self, df,targets,cat_features,cont_features,mode='train'):

        self.mode = mode

        self.data_cont = df[cont_features].values

        self.data_cat = df[cat_features].values

        if mode=='train':

            self.targets = targets.values 

    

    def __len__(self):

        return len(self.data_cont)

    

    def __getitem__(self, idx):

        if self.mode == 'train':

            return torch.FloatTensor(self.data_cont[idx]),torch.LongTensor(self.data_cat[idx]),torch.FloatTensor(self.targets[idx])

        elif self.mode == 'test':

            return torch.FloatTensor(self.data_cont[idx]), torch.LongTensor(self.data_cat[idx]),0
class RidModel(nn.Module):

    def __init__(self,emb_dims,no_of_cont):

        super(RidModel, self).__init__()

        

        self.emb = nn.ModuleList([nn.Embedding(x,y) for x,y in emb_dims])

        

        no_of_embs = sum([y for x, y in emb_dims])

        self.no_of_embs = no_of_embs

        self.no_of_cont = no_of_cont

        

        

        self.batch_norm1 = nn.BatchNorm1d(self.no_of_cont)

        self.dropout1 = nn.Dropout(0.2)

        self.dense1 = nn.utils.weight_norm(nn.Linear(no_of_cont, 128))

        

        self.batch_norm2 = nn.BatchNorm1d(128+no_of_embs)

        self.dense2 = nn.utils.weight_norm(nn.Linear(128+no_of_embs, 32))

         

        self.batch_norm3 = nn.BatchNorm1d(32)

        self.dense3 = nn.utils.weight_norm(nn.Linear(32, 16))

        

        self.batch_norm4 = nn.BatchNorm1d(16)

        self.dense4 = nn.utils.weight_norm(nn.Linear(16, 1))

        

       

    def forward(self, cont,cat):

         

        ## cat data part

        x_cat = [emb_layer(cat[:,i]) for i,emb_layer in enumerate(self.emb)]

        x_cat = torch.cat(x_cat,1)

        x_cat = self.dropout1(x_cat)

        ##cont data

        x = self.batch_norm1(cont)

        x = self.dropout1(x)

        x = F.relu(self.dense1(x))

        

        ##concat

        x = torch.cat([x,x_cat],1)

        

        ##rest of NN

        x = self.batch_norm2(x)

        x = F.relu(self.dense2(x))

        

        x = self.batch_norm3(x)

        x = F.relu(self.dense3(x))

        

        

        x = self.batch_norm4(x)

        x = F.sigmoid(self.dense4(x))

        

        return x
X_train,X_valid,y_train,y_valid = train_test_split(X[features],Y,test_size=0.15)
del X,Y,train

gc.collect()
assert X_train.shape[0]==y_train.shape[0]

assert X_valid.shape[0]==y_valid.shape[0]

X_train.head()
nepochs=5

train_set = RidDataset(X_train,y_train,cat_columns,cont_columns,mode="train")

valid_set = RidDataset(X_valid,y_valid,cat_columns,cont_columns,mode="train")

val_auc=[]

dataloaders = {'train':DataLoader(train_set,batch_size=2**15,shuffle=True),

              "val":DataLoader(valid_set,batch_size=2**15,shuffle=True)}



model = RidModel(cat_embs,len(cont_columns)).to(DEVICE)

checkpoint_path = 'rid_model.pt'

optimizer = optim.Adam(model.parameters())

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, eps=1e-4, verbose=True)

criterion = nn.BCELoss()

best_loss = {'train':np.inf,'val':np.inf}

auc_score = {'train':0,'val':0.0}



for epoch in range(nepochs):

            epoch_loss = {'train': 0.0, 'val': 0.0}

            

            for phase in ['train', 'val']:

                if phase == 'train':

                    model.train()

                else:

                    model.eval()

                

                running_loss = 0.0

                auc=0.0

                

                for i,(x,y,z) in enumerate(dataloaders[phase]):

                    x, y, z = x.to(DEVICE), y.to(DEVICE),z.to(DEVICE)

                    optimizer.zero_grad()

                    

                    with torch.set_grad_enabled(phase=='train'):

                        preds = model(x,y)

                        loss = criterion(preds, z)

                        auc = roc_auc_score(z.detach().cpu().numpy(),preds.detach().cpu().numpy())

                        

                        if phase=='train':

                            loss.backward()

                            optimizer.step()

                    

                    running_loss += loss.item() / len(dataloaders[phase])

                    auc += auc/len(dataloaders[phase])

                

                epoch_loss[phase] = running_loss

                auc_score[phase]=auc

                

            print("Epoch {}/{}   - loss: {:5.5f}   - val_loss: {:5.5f} -- AUC {:5.4f} --val AUC {:5.4f}".format(epoch+1,

                    nepochs, epoch_loss['train'], epoch_loss['val'],auc_score['train'],auc_score['val']))

            val_auc.append(auc_score['val'])

            scheduler.step(epoch_loss['val'])

            

            if epoch_loss['val'] < best_loss['val']:

                best_loss = epoch_loss

                torch.save(model.state_dict(), checkpoint_path)

                

 



print(f'Final validation AUC Score {np.mean(val_auc):5.4f}')
env = riiideducation.make_env()

iter_test = env.iter_test()


  

model = RidModel(cat_embs,len(cont_columns)).to(DEVICE)

model.load_state_dict(torch.load(checkpoint_path))

model.eval()

    

for (test_df, sample_prediction_df) in iter_test:

    preds=[]

    

    ##preprocess

    test_df = pd.merge(test_df, results_u, on=['user_id'],  how="left")

    test_df = pd.merge(test_df, results_c, on=['content_id'],  how="left")

    test_df = test_df.loc[test_df['content_type_id'] == 0].reset_index(drop=True)

    test_df = pd.merge(test_df, questions_df, left_on = 'content_id', right_on = 'question_id', how = 'left')

    

    test_df['prior_question_elapsed_time'].fillna(0,inplace=True)

    test_df['prior_question_had_explanation'].fillna(False,inplace=True)

    

    for col in cat_columns[2:]:

        test_df[col].fillna(questions_df[col].mode(),inplace=True)



    ## cont features filling nan with mode

    for col in cont_columns:

        test_df[col].fillna(fix_missing[col],inplace=True)

    

    print(test_df[col].isna().sum())

    ## cat features encoding

    for col in cat_columns:

        test_df[col] = enc_dict[col].transform(test_df[col])

    

    ## cont features scaling

    for col in cont_columns:

        test_df[col]=scale_dict[col].transform(test_df[col].values.reshape(-1,1))



    

    #print(test_df[features].isna().sum())

    ##dataloader

    train_set = RidDataset(test_df[features],None,cat_columns,cont_columns,mode="test")

    testloader = DataLoader(train_set,batch_size=32,shuffle=False)



    ##predictions

    for i,(x,y,z) in enumerate(testloader):

        x,y = x.to(DEVICE),y.to(DEVICE)



        with torch.no_grad():

            batch_pred = model(x,y)



        preds.append(batch_pred)



    preds = torch.cat(preds, dim=0).cpu().numpy()





    ##

    test_df['answered_correctly'] =  preds

    

    env.predict(test_df.loc[test_df['content_type_id'] == 0, ['row_id', 'answered_correctly']])