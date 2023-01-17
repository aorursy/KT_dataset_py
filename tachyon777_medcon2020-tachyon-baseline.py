DEBUG = False
import os,random,copy

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.model_selection import KFold,StratifiedKFold

from sklearn.metrics import accuracy_score, roc_auc_score



import torch

import torch.nn as nn

from torch.utils.data import DataLoader, Dataset

import torch.optim as optim

import torch.nn.functional as F

from torch.optim.lr_scheduler import CosineAnnealingLR



from tqdm import tqdm_notebook as tqdm

import time

import pickle



import warnings

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 200)

%matplotlib inline

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)
def seed_everything(seed=42):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

    

seed_everything(seed=42)
train = pd.read_csv("../input/ai-medical-contest-2020/train.csv")

all_train_len = len(train)

test = pd.read_csv("../input/ai-medical-contest-2020/test.csv")

sub = pd.read_csv("../input/ai-medical-contest-2020/sample_submission.csv")



if DEBUG:

    train = train[:1000]

    test = test[:1000]

    sub = sub[:1000]
# 日時を表す列をstring型からdatetime型に変換

format='%Y-%m-%d' # 二次地表示のフォーマット, 例) 2020-09-26

cols_time = ['entry_date', 'date_symptoms']

for col in cols_time:

    train[col] = pd.to_datetime(train[col],format=format)

    test[col] = pd.to_datetime(test[col],format=format)
def onehot_encoding(train: pd.DataFrame, test: pd.DataFrame, encode_cols):

    n_train = len(train)

    train = pd.concat([train, test], sort=False).reset_index(drop=True)

    for f in encode_cols:

        try:

            if f in not_nan_cols:

                dummies = pd.get_dummies(train[f], dummy_na=False, dtype=np.uint8, prefix='OH_'+f)

            else:

                dummies = pd.get_dummies(train[f], dummy_na=True, dtype=np.uint8, prefix='OH_'+f)

        except:

            print("exception :",f)

        train = pd.concat([train,dummies], axis=1)

    test = train[n_train:].reset_index(drop=True)

    train = train[:n_train]

    return train, test
def standardization(train: pd.DataFrame, test: pd.DataFrame, encode_cols):

    n_train = len(train)

    train = pd.concat([train, test], sort=False).reset_index(drop=True)

    for f in encode_cols:

        try:

            

            train[f].fillna(train[f].mean(),inplace=True) #テストデータも平均値に含めちゃうのは議論の余地がある

            

        except:

            print("exception :",f)

    lbl = preprocessing.StandardScaler()

    train[encode_cols] = lbl.fit_transform(train[encode_cols])

    test = train[n_train:].reset_index(drop=True)

    train = train[:n_train]

    return train, test
#minmaxの方がいいかも？

def date_transform(train: pd.DataFrame, test: pd.DataFrame, encode_cols):

    n_train = len(train)

    train = pd.concat([train, test], sort=False).reset_index(drop=True)

    for f in encode_cols:

        try:

            train[f] = train[f].apply(lambda x: x.dayofyear).astype(np.uint16)

            train[f].fillna(-1,inplace=True)

        except:

            print("exception :",f)

    test = train[n_train:].reset_index(drop=True)

    train = train[:n_train]

    return train, test
#https://www.kaggle.com/osciiart/ai-medical-contest-2020-baseline

def osciiart_transform(train: pd.DataFrame, test: pd.DataFrame, encode_cols=None):

    n_train = len(train)

    train = pd.concat([train, test], sort=False).reset_index(drop=True)

    

    ######################################################################

    #既に変形してある状態で再現を行うため、上記Notebookと同じ値にはならないことに注意#

    ######################################################################

    #"age_x_entry_date",年齢と入院日を乗算

    age = train['age'].values

    age = (age - age.mean()) / age.std() # 年齢を正規化

    entry_date = train['entry_date'].values

    entry_date = (entry_date - entry_date.mean()) / entry_date.std() # 年齢を正規化

    train['age_x_entry_date'] = age * entry_date # 年齢と入院日を乗算. 2つの変数の相互作用を表現できる

    

    #"mean_of_icu_of_each_entry_date",入院日ごとのICU入室率

    col_groupby = 'entry_date' # グループ分けに用いる列名

    col_aggregate = 'icu' # 統計量を得る特徴量

    df_tmp = copy.deepcopy(train[[col_groupby,col_aggregate]])

    df_tmp[col_aggregate] = train[col_aggregate]=="Yes" # ICUに入ったかどうか

    method = 'mean' # 統計量の種類

    df_agg = df_tmp.groupby(col_groupby)[col_aggregate].agg(method).reset_index() # 集約特徴量を得る

    col_new = 'mean_of_icu_of_each_entry_date' # 特徴量名. 入院日ごとのICU入室率

    df_agg.columns = [col_groupby, col_new]

    df_tmp = pd.merge(df_tmp, df_agg, on=col_groupby, how='left')

    train[col_new] = df_tmp[col_new]

    

    

    #"entry_-_symptom_date",発症から入院までの日数

    train['entry_-_symptom_date'] = train['entry_date'] - train['date_symptoms'] # 発症から入院までの日数

    

    #"entry_date_count",入院日における、その日の入院患者数

    res = copy.deepcopy(train[["patient_id","entry_date"]])

    res2 = train.groupby('entry_date')["patient_id"].agg(len).reset_index() # 集約特徴量を得る

    res2.columns = ['entry_date','entry_date_count']

    res = pd.merge(res,res2, on='entry_date', how='left').drop('entry_date', axis=1)

    train['entry_date_count'] = res['entry_date_count']

    ###

    

    test = train[n_train:].reset_index(drop=True)

    train = train[:n_train]

    return train, test
def only_fill_na(train: pd.DataFrame, test: pd.DataFrame, encode_cols):

    n_train = len(train)

    train = pd.concat([train, test], sort=False).reset_index(drop=True)

    for f in encode_cols:

        try:

            train[f].fillna(-1,inplace=True)

        except:

            print("exception :",f)

    test = train[n_train:].reset_index(drop=True)

    train = train[:n_train]

    return train, test
train_cols = [

    "place_patient_live2", #477患者の住所(詳細)

    "age", #年齢

    "place_hospital", #33病院の所在

    "place_patient_birth", #33患者の出身地

    "type_hospital", #14患者が治療を受けた施設

    "contact_other_covid", #2他のCOVID患者と接触歴があるか

    "test_result", #3 PF,result awaitedPRC検査結果

    "pneumonia", #2 nan not in train肺炎

    "place_patient_live", #33

    "age_x_entry_date", #元データにない。年齢と入院日を乗算

    "mean_of_icu_of_each_entry_date", #元データにない。入院日ごとのICU入室率

    "place_patient_live", #33患者の住所(大別)

    "date_symptoms", #発症日

    "entry_date", #入院日 (受診日)

    "intubed", #挿管

    "patient_type", #入院or外来

    "entry_-_symptom_date", #元データにない。発症から入院までの日数

    "entry_date_count", #元データにない。入院日における、その日の入院患者数

    "chronic_renal_failure", #慢性腎不全の有無

    "diabetes", #糖尿病の有無

    "icu", #ICUで治療を受けたか

    "obesity", #肥満

    "immunosuppression", #免疫抑制

    "sex", #性別

    "other_disease", #その他の疾患

    "pregnancy", #妊娠

    "hypertension", #高血圧

    "cardiovascular", #心血管疾患

    "asthma", #ぜんそく

    "tobacco", #喫煙

    "copd", #COPD





]



standard_cols = [

    "age",

    "entry_-_symptom_date",

    "entry_date_count",

    "date_symptoms",

    "entry_date",





]



onehot_cols = [

    #"place_patient_live2",

    "place_hospital", 

    "place_patient_birth", 

    "type_hospital", 

    "contact_other_covid", 

    "test_result", 

    "pneumonia", 

    "place_patient_live",

    "intubed", 

    "patient_type", 

    "chronic_renal_failure", 

    "diabetes",

    "icu",

    "obesity",

    "immunosuppression",

    "sex",

    "other_disease",

    "pregnancy",

    "hypertension",

    "cardiovascular",

    "asthma",

    "tobacco",

    "copd",



]



#onehotでnanが含まれないもの

not_nan_cols = (

    "place_hospital",

    "type_hospital",

    "test_result",

    "place_patient_live",

    "patient_type",

    "sex",

)





date_cols = [

    "date_symptoms",

    "entry_date",



]
train, test = onehot_encoding(train, test, onehot_cols)

train, test = date_transform(train, test, date_cols)

train, test = osciiart_transform(train, test)

train, test = standardization(train, test, standard_cols)
#if "place_patient_live2" is te.

train,test = only_fill_na(train, test,["place_patient_live2"])
categorical_features_OH = [] #onehotencodingされたバイナリ値の列

for i in onehot_cols:

    for j in train[i].unique():

        categorical_features_OH.append("OH_"+i+"_"+str(j))
numerical_features = [

    "age",

    "entry_-_symptom_date",

    "entry_date_count",

    "date_symptoms",

    "entry_date",

]
class CFG:

    model_name = "medcon2020_tachyon_baseline"

    Progress_Bar = False

    max_grad_norm=1000

    gradient_accumulation_steps=1

    hidden_size=512

    dropout=0.5

    init_lr=1e-2

    weight_decay=1e-6

    batch_size=128

    n_epochs=5 if DEBUG else 10

    n_fold = 4

    num_workers = 2

    num_features=numerical_features

    cat_features=categorical_features_OH

    target_cols= "died"

    model_save_path = False
skf = StratifiedKFold(CFG.n_fold, shuffle=True, random_state=0)

train['fold'] = -1

for i, (train_idx, valid_idx) in enumerate(skf.split(train, train[CFG.target_cols])):

    train.loc[valid_idx, 'fold'] = i
class Medcon2020Dataset(Dataset):

    def __init__(self, df, num_features = CFG.num_features, cat_features = CFG.cat_features, train=True,target_cols = CFG.target_cols):

        if train:

            self.cont_values = df[num_features].values

            self.cate_values = df[cat_features].values

            self.labels = df[target_cols].values

        else:

            self.cont_values = df[num_features].values

            self.cate_values = df[cat_features].values

        self.train = train

        

    def __getitem__(self, idx):

        

        if self.train:

            cont_x = torch.FloatTensor(self.cont_values[idx])

            cate_x = torch.FloatTensor(self.cate_values[idx])

            #cate_x = torch.LongTensor(self.cate_values[idx])

            y = torch.from_numpy(np.array(self.labels[idx]))

            return cont_x, cate_x, y

        else:

            cont_x = torch.FloatTensor(self.cont_values[idx])

            cate_x = torch.FloatTensor(self.cate_values[idx])

            #cate_x = torch.LongTensor(self.cate_values[idx])

            return cont_x, cate_x

    

    def __len__(self):

        return len(self.cont_values)
class TabularNN(nn.Module):

    def __init__(self, cfg):

        super().__init__()

        self.mlp = nn.Sequential(

                          nn.Linear(len(cfg.num_features)+len(cfg.cat_features)+1, cfg.hidden_size),

                          nn.BatchNorm1d(cfg.hidden_size),

                          nn.Dropout(cfg.dropout),

                          nn.PReLU(),

                          nn.Linear(cfg.hidden_size, cfg.hidden_size),

                          nn.BatchNorm1d(cfg.hidden_size),

                          nn.Dropout(cfg.dropout),

                          nn.PReLU(),

                          nn.Linear(cfg.hidden_size,1),

                          )



    def forward(self, cont_x, cate_x):

        # no use of cate_x yet

        x = torch.cat((cont_x, cate_x), dim=1)

        x = self.mlp(x)

        return x
def training(model, iterator, optimizer, criterion, device):

    

    epoch_loss = 0

    model.train()

    bar = tqdm(iterator) if CFG.Progress_Bar else iterator

    

    for (cont_x,cate_x,y) in bar:

        optimizer.zero_grad()

        cont_x,cate_x,y = cont_x.to(device),cate_x.to(device),y.to(device)

        y_pred = model(cont_x,cate_x)

        loss = criterion(y_pred, y.unsqueeze(1))

        loss.backward()

        optimizer.step()

        loss_np = loss.detach().cpu().numpy()

        epoch_loss += loss_np

        

        if CFG.Progress_Bar:

            bar.set_description('Training loss: %.5f' % (loss_np))

        

    return epoch_loss/len(iterator)



def evaluate(model, iterator, criterion, device):

    

    epoch_loss = 0

    preds = []

    preds = np.array(preds)

    targets = []

    targets = np.array(targets)

    model.eval()

    bar = tqdm(iterator) if CFG.Progress_Bar else iterator

    

    with torch.no_grad():

        

        for (cont_x,cate_x,y) in bar:

            cont_x,cate_x,y = cont_x.to(device),cate_x.to(device),y.to(device)

            y_pred = model(cont_x,cate_x)

            loss = criterion(y_pred, y.unsqueeze(1))

            loss_np = loss.detach().cpu().numpy()

            epoch_loss += loss_np

            y_pred = torch.sigmoid(y_pred)

            preds = np.append(preds, y_pred.detach().cpu().numpy())

            targets = np.append(targets, y.detach().cpu().numpy())

            

            if CFG.Progress_Bar:

                bar.set_description('Validation loss: %.5f' % (loss_np))

                

    val_acc = accuracy_score(targets, np.round(preds))

    try:

       val_roc = roc_auc_score(targets, preds)

    except ValueError:

       val_roc = -1

    

    return epoch_loss/len(iterator),val_acc,val_roc
def fit_model(model, model_name, train_iterator, valid_iterator, optimizer, loss_criterion, device, epochs):

    """ Fits a dataset to model"""

    best_valid_loss = float('inf')

    

    train_losses = []

    valid_losses = []

    valid_roc_scores = []

    

    for epoch in range(epochs):

        scheduler.step(epoch)

    

        train_loss = training(model, train_iterator, optimizer, loss_criterion, device)

        valid_loss,valid_acc_score, valid_roc_score= evaluate(model, valid_iterator, loss_criterion, device)

        

        train_losses.append(train_loss)

        valid_losses.append(valid_loss)

        valid_roc_scores.append(valid_roc_score)



        if valid_loss <= best_valid_loss:

            best_valid_loss = valid_loss

            if CFG.model_save_path:

                torch.save(model.state_dict(), os.path.join(model_save_path,f'{model_name}.pt'))

            else:

                torch.save(model.state_dict(), f'{model_name}_best.pt')

            best_model = copy.deepcopy(model)

        

        #schedulerの処理 cosineannealingは別

        #if scheduler != None:

        #    scheduler.step(valid_loss)

        

        print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.4f} | Val. Loss: {valid_loss:.4f} ')

        print(f'Val. ACC Score: {valid_acc_score:.3f} | Val. Metric Score: {valid_roc_score:.4f}')

        #print(f'Val. Loss: {valid_loss:.3f} | Val. ACC Score: {valid_acc_score:.3f} | Val. Metric Score: {valid_roc_score:.4f}')

        #print(f'lr:{optimizer.param_groups[0]["lr"]:.7f}')

        if CFG.model_save_path:

                torch.save(model.state_dict(), os.path.join(model_save_path,f'{model_name}_final.pt'))

        else:

            torch.save(model.state_dict(), f'{model_name}_final.pt')

    

    return train_losses, valid_losses, valid_roc_scores,best_model
tr_loss=[]

val_loss=[]

val_roc=[]

models = []

best_models = []

for fold in range(1 if DEBUG else CFG.n_fold): #n_fold

    print(f"Fitting on Fold {fold+1}")

    #Make Train and Valid DataFrame from fold

    train_fold = train[train['fold'] != fold].reset_index(drop=True)

    valid_fold = train[train['fold'] == fold].reset_index(drop=True)

    

    #target encoding

    target_E_cols = ["place_patient_live2"]

    for col in target_E_cols:

        train_fold[col + "_te"] = -1

        valid_fold[col + "_te"] = -1

        data_tmp = pd.DataFrame({col:train_fold[col],"target":train_fold[CFG.target_cols]})

        target_mean = data_tmp.groupby(col)["target"].mean() #平均値の算出

        valid_fold.loc[:,col + "_te"] = valid_fold[col].map(target_mean) #評価用→trainの平均で格納

        

        #学習データの変換後の値を格納する配列を準備

        tmp = np.repeat(np.nan,train_fold.shape[0]) #他foldに出てこない値があるとnanのままになるのでデフォで0を入れとくしかない。

        

        #target encodingにおいて、自分の値を参照してはいけないので、更にfoldに分ける

        kf_encoding = KFold(n_splits=4,shuffle=False)

        for idx_1,idx_2 in kf_encoding.split(train_fold):

            target_mean = data_tmp.iloc[idx_1].groupby(col)["target"].mean()

            tmp[idx_2] = train_fold[col].iloc[idx_2].map(target_mean)

        

        train_fold.loc[:,col + "_te"] = tmp 

    train_fold["place_patient_live2_te"].fillna(0.,inplace=True) #他foldに出てこない値があるとnanのままになるのでデフォで0を入れとくしかない。

    valid_fold["place_patient_live2_te"].fillna(0.,inplace=True)

    #Build and load Dataset

    train_data = Medcon2020Dataset(train_fold,num_features = CFG.num_features + ["place_patient_live2_te"], cat_features = CFG.cat_features)

    valid_data = Medcon2020Dataset(valid_fold,num_features = CFG.num_features + ["place_patient_live2_te"], cat_features = CFG.cat_features)

    

    train_iterator = DataLoader(train_data, shuffle=True, batch_size=CFG.batch_size, num_workers=CFG.num_workers)

    valid_iterator = DataLoader(valid_data, shuffle=False, batch_size=CFG.batch_size, num_workers=CFG.num_workers)

    

    #Initialize model, loss and optimizer

    model = TabularNN(CFG).to(device)

    loss_criterion = nn.BCEWithLogitsLoss()

    opt=optim.Adam(model.parameters(), lr=CFG.init_lr, betas=(0.9,0.999))

    scheduler = CosineAnnealingLR(opt, CFG.n_epochs)

    

    name = CFG.model_name + "_f" + str(fold)

    

    temp_tr_loss, temp_val_loss,temp_val_roc,best_model = fit_model(model, name, train_iterator, valid_iterator, opt, loss_criterion, device, epochs=CFG.n_epochs)

    

    tr_loss.append(temp_tr_loss)

    val_loss.append(temp_val_loss)

    val_roc.append(temp_val_roc)

    models.append(model)

    best_models.append(best_model)
for i in range(len(tr_loss)):

    fig,ax = plt.subplots(nrows=1, ncols=2, figsize=(20,5))

    ax[0].plot(tr_loss[i])

    ax[0].set_title('Training and Validation Loss')

    ax[0].plot(val_loss[i])

    ax[0].set_xlabel('Epoch')



    ax[1].plot(val_roc[i])

    ax[1].set_title('Val ROC Score')

    ax[1].set_xlabel('Epoch')





    ax[0].legend();

    ax[1].legend();
#if target E

for col in target_E_cols:

    test[col + "_te"] = -1

    data_tmp = pd.DataFrame({col:train[col],"target":train[CFG.target_cols]})

    target_mean = data_tmp.groupby(col)["target"].mean() #平均値の算出

    test.loc[:,col + "_te"] = test[col].map(target_mean) #評価用→trainの平均で格納

test["place_patient_live2_te"].fillna(0.,inplace=True)
test_dataset = Medcon2020Dataset(test,num_features = CFG.num_features + ["place_patient_live2_te"], cat_features = CFG.cat_features,train=False)
def get_predictions(model, iterator, device):

    

    preds = np.array([0.]*len(test))

    model.eval()

    bar = tqdm(iterator) if CFG.Progress_Bar else iterator

    

    with torch.no_grad():

        res = np.array([])

        for (cont_x,cate_x) in bar:

            cont_x,cate_x = cont_x.to(device),cate_x.to(device)

            y_pred = model(cont_x,cate_x)

            y_pred = torch.sigmoid(y_pred)

            res = np.append(res, y_pred.detach().cpu().numpy())



        preds += res

    return preds
prediction = np.array([0.]*len(test))

for i in range(len(models)):

    test_iterator = DataLoader(dataset=test_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers)

    preds = get_predictions(models[i], test_iterator, device)

    prediction += preds

prediction /= len(models)
sub[CFG.target_cols] = prediction

sub.to_csv('submission_final.csv', index=False)

sub.head()
prediction = np.array([0.]*len(test))

for i in range(len(best_models)):

    test_iterator = DataLoader(dataset=test_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers)

    preds = get_predictions(best_models[i], test_iterator, device)

    prediction += preds

prediction /= len(best_models)

sub[CFG.target_cols] = prediction

sub.to_csv('submission_best.csv', index=False)

sub.head()