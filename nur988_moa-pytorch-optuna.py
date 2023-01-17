import fastai

import optuna

import torch

import torch.nn as nn

import numpy as np

import pandas as pd

from sklearn.preprocessing import QuantileTransformer

import sys

sys.path.append('../input/iterative-stratification/iterative-stratification-master')

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
DEVICE="cuda"

EPOCHS=30

params= {'num_layers': 3, 'hidden_size': 1404, 'dropout': 0.30089577325447585, 'learning_rate': 3.764735971734488e-05}

df=pd.read_csv('../input/lish-moa/train_targets_scored.csv')

df.loc[:,'kfold']=-1

df=df.sample(frac=1).reset_index(drop=True)

targets=df.drop("sig_id",axis=1).values

mksf=MultilabelStratifiedKFold(n_splits=2)

for f,(trn,val) in enumerate(mksf.split(X=df,y=targets)):

    df.loc[val,'kfold']=int(f)

df.to_csv('train_targets_fold.csv',index=False)    
df.head()
class MoaDataset:

    def __init__(self,features,targets):

        self.features=features

        self.targets=targets

        

    def __len__(self):

        return self.features.shape[0]

    def __getitem__(self,item):

        return {

            'x':torch.tensor(self.features[item,:],dtype=torch.float),

            'y':torch.tensor(self.targets[item,:],dtype=torch.float)

        }
class TestDataset:

    def __init__(self,features):

        self.features=features

        

        

    def __len__(self):

        return self.features.shape[0]

    def __getitem__(self,item):

        return {

            'x':torch.tensor(self.features[item,:],dtype=torch.float)

        }
#df=pd.read_csv("../input/lish-moa/train_features.csv")

def ohe(df):

    ohe=pd.get_dummies(df['cp_time'])

    df=pd.concat([df,ohe],axis=1)

    ohe=pd.get_dummies(df['cp_type'])

    df=pd.concat([df,ohe],axis=1)

    ohe=pd.get_dummies(df['cp_dose'])

    df=pd.concat([df,ohe],axis=1)

    df.drop(['cp_type','cp_time','cp_dose'],axis=1,inplace=True)

    return df
class Engine:

    def __init__(self,model,optimizer,device):

        self.model=model

        self.optimizer=optimizer

        self.device=device

    @staticmethod    

    def loss_fn(targets,outputs):

        return nn.BCEWithLogitsLoss()(outputs,targets)

    def train(self,data_loader):

        self.model.train()

        final_loss=0

        for data in data_loader:

            self.optimizer.zero_grad()

            inputs=data['x'].to(self.device)

            targets=data['y'].to(self.device)

            outputs=self.model(inputs)

            loss=self.loss_fn(targets,outputs)

            loss.backward()

            self.optimizer.step()

            final_loss+=loss.item()

        return final_loss/len(data_loader)    

            

    def evaluate(self,data_loader):

        self.model.eval()

        final_loss=0

        for data in data_loader:

            #self.optimizer.zero_grad()

            inputs=data['x'].to(self.device)

            targets=data['y'].to(self.device)

            outputs=self.model(inputs)

            loss=self.loss_fn(targets,outputs)

            #loss.backward()

            #self.optimizer.step()

            final_loss+=loss.item()

        return final_loss/len(data_loader)       
class Model(nn.Module):

    def __init__(self,nfeatures,ntargets,nlayers,hidden_size,dropout):

        super().__init__()

        layers=[]

        for _ in range(nlayers-1):

            if len(layers)==0:

                layers.append(nn.BatchNorm1d(nfeatures))

                layers.append(nn.Dropout(dropout))

                layers.append(nn.Linear(nfeatures,hidden_size))

                layers.append(nn.Tanh())

                

                

                

            else:

               

                layers.append(nn.BatchNorm1d(hidden_size))

                layers.append(nn.Dropout(dropout))

                layers.append(nn.Linear(hidden_size,hidden_size))

                layers.append(nn.Tanh())

                

        if len(layers)==0:

            layers.append(nn.BatchNorm1d(nfeatures))

            layers.append(nn.Dropout(dropout))

            layers.append(nn.Linear(nfeatures,ntargets))

            

        else :

            layers.append(nn.BatchNorm1d(hidden_size))

            layers.append(nn.Dropout(dropout))

            layers.append(nn.Linear(hidden_size,ntargets))

        self.model=nn.Sequential(*layers)

            

    def forward(self,x):

        return self.model(x)


def run_training(fold,params,save_model=True):

    df=pd.read_csv("../input/lish-moa/train_features.csv")

    train_features=pd.read_csv("../input/lish-moa/train_features.csv")

    GENES=[col for col in train_features.columns if col.startswith('g-')]

    CELLS=[col for col in train_features.columns if col.startswith('c-')]

    for col in (GENES+CELLS):

        transformer=QuantileTransformer(n_quantiles=100,random_state=0,output_distribution="normal")

        vec_len=len(train_features[col].values)

        #vec_len_test=len(test_features[col].values)

        raw_vec=train_features[col].values.reshape(vec_len,1)

        transformer.fit(raw_vec)

        train_features[col]=transformer.transform(raw_vec).reshape(1,vec_len)[0]

        #test_features[col]=transformer.transform(test_features[col].values.reshape(vec_len_test,1)).reshape(1,vec_len_test)[0]

    df=ohe(train_features)

    

    targets_df=pd.read_csv("./train_targets_fold.csv")

    features_columns=df.drop('sig_id',axis=1).columns

    targets_columns=targets_df.drop(['sig_id','kfold'],axis=1).columns

    df=df.merge(targets_df,on='sig_id',how='inner')

    train_df=df[df.kfold!=fold].reset_index(drop=True)

    valid_df=df[df.kfold==fold].reset_index(drop=True)

    

    x_train=train_df[features_columns].to_numpy()

    x_valid=train_df[features_columns].to_numpy()

    y_train=train_df[targets_columns].to_numpy()

    y_valid=train_df[targets_columns].to_numpy()

    

    

    train_dataset=MoaDataset(features=x_train,targets=y_train)

    valid_dataset=MoaDataset(features=x_valid,targets=y_valid)

    

    train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=64,num_workers=8)

    valid_loader=torch.utils.data.DataLoader(valid_dataset,batch_size=64,num_workers=8)

    

    

    model=Model(

        nfeatures=x_train.shape[1],

        ntargets=y_train.shape[1],

        nlayers=params["num_layers"],

        hidden_size=params["hidden_size"],

        dropout=params["dropout"]

    

    )

    model.to(DEVICE)

    optimizer=torch.optim.Adam(model.parameters(),lr=params["learning_rate"])

    eng=Engine(model,optimizer,device=DEVICE)

    best_loss=np.inf

    

    early_stopping_iter=10

    early_stopping_counter=0

    

    for epoch in range(EPOCHS):

        train_loss=eng.train(train_loader)

        valid_loss=eng.evaluate(valid_loader)

        print(f"Fold-{fold},--EPOCH-{epoch},--TRAIN_LOSS-{train_loss},--VAL_LOSS--{valid_loss}")

        

        if best_loss>valid_loss:

            best_loss=valid_loss

            if save_model:

                torch.save(model.state_dict(),f"model_{fold}.pth")

                c=fold

        else :

            early_stopping_counter+=1

            

        if early_stopping_counter>early_stopping_iter:

            break

    return best_loss,c    
#def objective(trial):

 #   params={

  #      "num_layers":trial.suggest_int("num_layer",1,5),

   #    "hidden_size":trial.suggest_int("hidden_size",800,2048),

    #    "dropout":trial.suggest_uniform("dropout",0.1,0.7),

     #   "learning_rate":trial.suggest_loguniform("learning_rate",1e-6,1e-3)



    #}

    

   # all_losses=[]

    #for f_ in range(1):

     #  temp_loss=run_training(f_,params,save_model=False)

      # all_losses.append(temp_loss)

    #return np.mean(all_losses)    
#study=optuna.create_study(direction="minimize")

#study.optimize(objective,n_trials=100)

#print("best trial")

#trial_=study.best_trial

#print(trial_.values)

#print(trial_.params)





for i in range(1): 

    a,b=run_training(i,params,save_model=True)

    print(f"Best Score--{a}--Best Model at fold--{b}")

    

    

    

    
def inference_fn(model, dataloader, device):

    model.eval()

    preds = []

    

    for data in dataloader:

        inputs = data['x'].to(device)



        with torch.no_grad():

            outputs = model(inputs)

        

        preds.append(outputs.sigmoid().detach().cpu().numpy())

        

    preds = np.concatenate(preds)

    

    return preds

   
df=pd.read_csv("../input/lish-moa/test_features.csv")

train_features=pd.read_csv("../input/lish-moa/test_features.csv")

GENES=[col for col in train_features.columns if col.startswith('g-')]

CELLS=[col for col in train_features.columns if col.startswith('c-')]

for col in (GENES+CELLS):

    transformer=QuantileTransformer(n_quantiles=100,random_state=0,output_distribution="normal")

        #vec_len=len(train_features[col].values)

    vec_len_test=len(train_features[col].values)

    raw_vec=train_features[col].values.reshape(vec_len_test,1)

    transformer.fit(raw_vec)

        #train_features[col]=transformer.transform(raw_vec).reshape(1,vec_len)[0]

    train_features[col]=transformer.transform(train_features[col].values.reshape(vec_len_test,1)).reshape(1,vec_len_test)[0]

df=ohe(train_features)

features_columns=df.drop('sig_id',axis=1).columns

x_test=df[features_columns].to_numpy()

testdataset = TestDataset(x_test)

testloader = torch.utils.data.DataLoader(testdataset, batch_size=1024, shuffle=False)

    

model=Model(

        nfeatures=df.shape[1]-1,

        ntargets=206,

        nlayers=params["num_layers"],

        hidden_size=params["hidden_size"],

        dropout=params["dropout"]

    

    

    

    )

model.load_state_dict(torch.load(f"model_{b}.pth"))

model.to(DEVICE)

    

predictions = np.zeros((128, 206))

predictions = inference_fn(model, testloader, DEVICE)
predictions
_df=pd.read_csv("../input/lish-moa/test_features.csv")

target=pd.read_csv('../input/lish-moa/train_targets_scored.csv')

target=target.drop('sig_id',axis=1).columns

id=_df.loc[_df['cp_type'] =='ctl_vehicle', 'sig_id']



_df=pd.DataFrame(predictions,columns=list(target),index=_df['sig_id'])

_df.index[0]

for i in range(len(_df.index)):

    if _df.index[i] in(id):

        _df.iloc[_df.index[i],train_targets_scored.columns[1:]]=0

_df.to_csv('submission.csv')