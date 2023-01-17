from IPython.display import IFrame, YouTubeVideo

YouTubeVideo('ysBaZO8YmX8',width=600, height=400)
#Installing Pytorch-Tabnet

!pip install -U catalyst

!pip install pytorch-tabnet
import numpy as np

import pandas as pd

import random

import os

import seaborn as sns

from tqdm.autonotebook import tqdm

tqdm.pandas()

from scipy.stats import skew 

import pickle

import glob



#Visuals

import matplotlib.pyplot as plt



#torch

import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import Dataset,DataLoader

import catalyst

#from catalyst.data.sampler import BalanceClassSampler



#CV2

import cv2



#Importing Tabnet

from pytorch_tabnet.tab_network import TabNet



#error

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import roc_auc_score
class EarlyStopping:

    def __init__(self, patience=7, mode="max", delta=0.001,verbose=False):

        self.patience = patience

        self.counter = 0

        self.mode = mode

        self.best_score = None

        self.early_stop = False

        self.delta = delta

        self.verbose = verbose

        if self.mode == "min":

            self.val_score = np.Inf

        else:

            self.val_score = -np.Inf



    def __call__(self, epoch_score, model, model_path):



        if self.mode == "min":

            score = -1.0 * epoch_score

        else:

            score = np.copy(epoch_score)



        if self.best_score is None:

            self.best_score = score

            self.save_checkpoint(epoch_score, model, model_path)

        elif score < self.best_score + self.delta:

            self.counter += 1

            if self.verbose:

                print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))

                

            if self.counter >= self.patience:

                self.early_stop = True

        else:

            self.best_score = score

            self.save_checkpoint(epoch_score, model, model_path)

            self.counter = 0



    def save_checkpoint(self, epoch_score, model, model_path):

        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:

            if self.verbose:

                print('Validation score improved ({} --> {}). Saving model!'.format(self.val_score, epoch_score))

            torch.save(model.state_dict(), model_path)

        self.val_score = epoch_score
BATCH_SIZE = 1024 

EPOCHS = 150

LR = 0.02

seed = 2020   # seed for reproducible results

patience = 50

device = torch.device('cuda')

FOLDS = 5
def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
seed_everything(seed)
# Defining Categorical variables and their Indexes, embedding dimensions , number of classes each have

df =pd.read_csv('../input/fe-using-only-competition-data-melanoma/melanoma_folds.csv')



df.drop(['image_id','stratify_group','center','diagnosis','benign_malignant'],axis=1,inplace=True)

target = 'target'

unused_feat = ['patient_id','fold']

features = [ col for col in df.columns if col not in unused_feat+[target]] 



categorical_columns = []



for col in df.columns[df.dtypes == object]:

    

    if col not in unused_feat:

        print(col, df[col].nunique())

        

        l_enc = LabelEncoder()

        df[col] = l_enc.fit_transform(df[col].values)

        

        #SAVING LABEL _ ENC

        output = open(f'{col}_encoder.pkl', 'wb')

        pickle.dump(l_enc, output)

        output.close()

        

        categorical_columns.append(col)
class MelanomaDataset(Dataset):

    def __init__(self,features,target):

        self.features = features

        self.target = target

        

    def __len__(self):

        return len(self.features)

    

    def __getitem__(self,idx):

        return{

            'features': torch.tensor(self.features[idx],dtype=torch.float),

             'target': self.one_hot(2, self.target[idx])

        }

    

    def get_targets(self):

        return list(self.target)

    

    @staticmethod

    def one_hot(size, target):

        tensor = torch.zeros(size, dtype=torch.float32)

        tensor[target] = 1.

        return tensor
class CustomTabnet(nn.Module):

    def __init__(self, input_dim, output_dim,n_d=8, n_a=8,n_steps=3, gamma=1.3,

                cat_idxs=[], cat_dims=[], cat_emb_dim=1,n_independent=2, n_shared=2,

                momentum=0.02,mask_type="sparsemax"):

        

        super(CustomTabnet, self).__init__()

        self.tabnet = TabNet(input_dim=input_dim,output_dim=output_dim, n_d=n_d, n_a=n_a,n_steps=n_steps, gamma=gamma,

                             cat_idxs=cat_idxs, cat_dims=cat_dims, cat_emb_dim=cat_emb_dim,n_independent=n_independent,

                             n_shared=n_shared, momentum=momentum,mask_type="sparsemax")

        

        

    def forward(self, x):

        return self.tabnet(x)
class SoftMarginFocalLoss(nn.Module):

    def __init__(self, margin=0.2, gamma=2):

        super(SoftMarginFocalLoss, self).__init__()

        self.gamma = gamma

        self.margin = margin

                

        self.weight_pos = 2

        self.weight_neg = 1

    

    def forward(self, inputs, targets):

        em = np.exp(self.margin)

        

        log_pos = -F.logsigmoid(inputs)

        log_neg = -F.logsigmoid(-inputs)

        

        log_prob = targets*log_pos + (1-targets)*log_neg

        prob = torch.exp(-log_prob)

        margin = torch.log(em + (1-em)*prob)

        

        weight = targets*self.weight_pos + (1-targets)*self.weight_neg

        loss = self.margin + weight * (1 - prob) ** self.gamma * log_prob

        

        loss = loss.mean()

        

        return loss
def train_fn(dataloader,model,criterion,optimizer,device,scheduler,epoch):

    model.train()

    

    train_targets=[]

    train_outputs=[]

    

    for bi,d in enumerate(dataloader):

        features = d['features']

        target = d['target']

        

        features = features.to(device, dtype=torch.float)

        target = target.to(device, dtype=torch.float)

        

        optimizer.zero_grad()

        

        output,_ = model(features)

        

        loss = criterion(output,target)

        loss.backward()

        optimizer.step()

        

        if scheduler is not None:

            scheduler.step()

            

        output  = 1 - F.softmax(output,dim=-1).cpu().detach().numpy()[:,0]  

        

        train_targets.extend(target.cpu().detach().numpy().argmax(axis=1).astype(int).tolist())

        train_outputs.extend(output)

            

        

    return loss.item(),train_outputs,train_targets
def eval_fn(data_loader,model,criterion,device):

    

    fin_targets=[]

    fin_outputs=[]

    

    model.eval()

    with torch.no_grad():

        

        for bi, d in enumerate(data_loader):

            features = d["features"]

            target = d["target"]



            features = features.to(device, dtype=torch.float)

            target = target.to(device, dtype=torch.float)



            outputs,_ = model(features)

            

            loss_eval = criterion(outputs,target)

            

            outputs  = 1 - F.softmax(outputs,dim=-1).cpu().detach().numpy()[:,0]  

            

            fin_targets.extend(target.cpu().detach().numpy().argmax(axis=1).astype(int).tolist())

            fin_outputs.extend(outputs)

            

    return loss_eval.item(),fin_outputs,fin_targets
def print_history(fold,history,num_epochs=EPOCHS):

        plt.figure(figsize=(15,5))

        

        plt.plot(

            np.arange(num_epochs),

            history['train_history_auc'],

            '-o',

            label='Train AUC',

            color='#ff7f0e'

        )

        

        plt.plot(

            np.arange(num_epochs),

            history['val_history_auc'],

            '-o',

            label='Val AUC',

            color='#1f77b4'

        )

        

        x = np.argmax(history['val_history_auc'])

        y = np.max(history['val_history_auc'])

        

        xdist = plt.xlim()[1] - plt.xlim()[0]

        ydist = plt.ylim()[1] - plt.ylim()[0]

        

        plt.scatter(x, y, s=200, color='#1f77b4')

        

        plt.text(

            x-0.03*xdist,

            y-0.13*ydist,

            'max auc\n%.2f'%y,

            size=14

        )

        

        plt.ylabel('AUC', size=14)

        plt.xlabel('Epoch', size=14)

        

        plt.legend(loc=2)

        

        plt2 = plt.gca().twinx()

        

        plt2.plot(

            np.arange(num_epochs),

            history['train_history_loss'],

            '-o',

            label='Train Loss',

            color='#2ca02c'

        )

        

        plt2.plot(

            np.arange(num_epochs),

            history['val_history_loss'],

            '-o',

            label='Val Loss',

            color='#d62728'

        )

        

        x = np.argmin(history['val_history_loss'])

        y = np.min(history['val_history_loss'])

        

        ydist = plt.ylim()[1] - plt.ylim()[0]

        

        plt.scatter(x, y, s=200, color='#d62728')

        

        plt.text(

            x-0.03*xdist, 

            y+0.05*ydist, 

            'min loss', 

            size=14

        )

        

        plt.ylabel('Loss', size=14)

        

        plt.title(f'FOLD {fold + 1}',size=18)

        

        plt.legend(loc=3)

        plt.show()  
def run(fold):

    

    df_train = df[df.fold != fold]

    df_valid = df[df.fold == fold]

    

    # Defining DataSet

    train_dataset = MelanomaDataset(

        df_train[features].values,

        df_train[target].values

    )

  

    

    valid_dataset = MelanomaDataset(

        df_valid[features].values,

        df_valid[target].values

    )

    

    # Defining DataLoader with BalanceClass Sampler

    train_loader = DataLoader(

        train_dataset,

        #sampler=BalanceClassSampler(

         #   labels=train_dataset.get_targets(), 

          #  mode="downsampling",

        #),

        batch_size=BATCH_SIZE,

        pin_memory=True,

        drop_last=True,

        num_workers=4

    )

    

    

    valid_loader = torch.utils.data.DataLoader(

        valid_dataset,

        batch_size=BATCH_SIZE,

        num_workers=4,

        shuffle=False,

        pin_memory=True,

        drop_last=False,

    )

    

    # Defining Device

    device = torch.device("cuda")

    

    # Defining Model for specific fold

    model = CustomTabnet(input_dim = len(features), 

                         output_dim = 2,

                         n_d=32, 

                         n_a=32,

                         n_steps=4, 

                         gamma=1.6,

                         n_independent=2,

                         n_shared=2,

                         momentum=0.02,

                         mask_type="sparsemax")

    

    model.to(device)

    

    #DEfining criterion

    criterion = SoftMarginFocalLoss()

    criterion.to(device)

        

    # Defining Optimizer with weight decay to params other than bias and layer norms

    param_optimizer = list(model.named_parameters())

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    optimizer_parameters = [

        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},

        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},

            ]  

    

    optimizer = torch.optim.Adam(optimizer_parameters, lr=LR)

    

    # Defining LR SCheduler

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',

                                                           factor=0.1, patience=10, verbose=True, 

                                                           threshold=0.0001, threshold_mode='rel',

                                                           cooldown=0, min_lr=0, eps=1e-08)

    #DEfining Early Stopping Object

    es = EarlyStopping(patience=patience,verbose=False)

    

    # History dictionary to store everything

    history = {

            'train_history_loss': [],

            'train_history_auc': [],

            'val_history_loss': [],

            'val_history_auc': [],

        }

        

    # THE ENGINE LOOP    

    tk0 = tqdm(range(EPOCHS), total=EPOCHS)

    for epoch in tk0:

        train_loss,train_out,train_targets = train_fn(train_loader, model,criterion, optimizer, device,scheduler=None,epoch=epoch)

        

        val_loss,outputs, targets = eval_fn(valid_loader, model, criterion,device)

        

        train_auc = roc_auc_score(train_targets, train_out)

        auc_score = roc_auc_score(targets, outputs)

        

        scheduler.step(auc_score)

        

        tk0.set_postfix(Train_Loss=train_loss,Train_AUC_SCORE = train_auc,Valid_Loss = val_loss,Valid_AUC_SCORE = auc_score)

        

        history['train_history_loss'].append(train_loss)

        history['train_history_auc'].append(train_auc)

        history['val_history_loss'].append(val_loss)

        history['val_history_auc'].append(auc_score)



        es(auc_score,model,f'model_{fold}.bin')

        

        if es.early_stop:

            print('Maximum Patience {} Reached , Early Stopping'.format(patience))

            break   

            

    print_history(fold,history,num_epochs=epoch+1)
run(fold=0)
run(fold=1)
run(fold=2)
run(fold=3)
run(fold=4)
df_test =pd.read_csv('../input/fe-using-only-competition-data-melanoma/test.csv')

df_test['anatom_site_general_challenge'].fillna('unknown',inplace=True)

df_test['target'] = 0
# Defining Categorical variables and their Indexes, embedding dimensions , number of classes each have 

df_test.drop(['image_name'],axis=1,inplace=True)

target = 'target'

unused_feat = ['patient_id','fold','center']

features = [ col for col in df_test.columns if col not in unused_feat+[target]] 



for col in df_test.columns[df_test.dtypes == object]:

    if col not in unused_feat:

        print(col, df_test[col].nunique())

        pkl_file = open(f'{col}_encoder.pkl', 'rb')

        l_enc = pickle.load(pkl_file) 

        df_test[col] = l_enc.transform(df_test[col].values)

        pkl_file.close()
def load_model():

    

    models = []

    paths = glob.glob('model*')

    

    for path in tqdm(paths,total=len(paths)):

                

        model = CustomTabnet(input_dim = len(features), 

                         output_dim = 2,

                         n_d=32, 

                         n_a=32,

                         n_steps=4, 

                         gamma=1.6,

                         n_independent=2,

                         n_shared=2,

                         momentum=0.02,

                         mask_type="sparsemax")

        

        model.to(device)

        loader = torch.load(path)

        model.load_state_dict(loader)

        

        models.append(model)

        

    return models
models = load_model()
def make_prediction(data_loader):

    predictions = np.zeros((len(df_test),FOLDS))

    for i,model in enumerate(models):

        

        fin_outputs=[]

        

        model.eval()

        with torch.no_grad():

            

            for bi, d in enumerate(data_loader):

                features = d["features"]

                target = d["target"]



                features = features.to(device, dtype=torch.float)



                outputs,_ = model(features)



                outputs  = 1 - F.softmax(outputs,dim=-1).cpu().detach().numpy()[:,0]                

        

                fin_outputs.extend(outputs)

        

        predictions[:,i] = fin_outputs

    

    return predictions
test_dataset = MelanomaDataset(

        df_test[features].values,

        df_test[target].values

    )



test_loader = torch.utils.data.DataLoader(

        test_dataset,

        batch_size=BATCH_SIZE,

        num_workers=4,

        shuffle=False,

        pin_memory=True,

        drop_last=False,

    )
pred = make_prediction(test_loader)
pred = pred.mean(axis=-1)

pred
ss = pd.read_csv('../input/siim-isic-melanoma-classification/sample_submission.csv')
ss['target'] = pred
ss.to_csv('submission.csv',index=False)

ss.head()