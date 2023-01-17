import os

import sys

import gc

import glob

import time

from os import listdir

import tqdm

from typing import Dict



from sklearn.preprocessing import StandardScaler

from sklearn.metrics import roc_auc_score



import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use("fivethirtyeight")



import plotly.express as px

import plotly.graph_objs as go

from plotly.offline import iplot

import plotly.figure_factory as ff



#supress warnings

import warnings

warnings.filterwarnings("ignore")



from colorama import Fore, Back, Style

y_ = Fore.YELLOW

r_ = Fore.RED

g_ = Fore.GREEN

b_ = Fore.BLUE

m_ = Fore.MAGENTA

c_ = Fore.CYAN

sr_ = Style.RESET_ALL
folder_path = '../input/riiid-test-answer-prediction/'

train_csv = folder_path + 'train.csv'

test_csv =  folder_path + 'example_test.csv'

lec_csv  =  folder_path + 'lectures.csv'

que_csv =   folder_path + 'questions.csv'

sample_csv =    folder_path + 'example_sample_submission.csv'



dtype = {'row_id':'int64',

         'timestemp':'int64',

        'user_id':'int32',

        'content_id':'int16',

        'content_type_id':'int8',

        'task_container_id':'int16',

        'user_answer':'int8',

        'answered_correctly':'int8',

        'prior_question_elapsed_time':'float32',

        'prior_question_had_explanation':'boolean'}



train_data = pd.read_csv(train_csv,dtype=dtype,nrows=10**6)

test_data = pd.read_csv(test_csv)

lec_data = pd.read_csv(lec_csv)

que_data = pd.read_csv(que_csv)

sample = pd.read_csv(sample_csv)
print(f"{y_}Number of rows in train data: {r_}{train_data.shape[0]}\n{y_}Number of columns in train data: {r_}{train_data.shape[1]}")

print(f"{g_}Number of rows in test data: {r_}{test_data.shape[0]}\n{g_}Number of columns in test data: {r_}{test_data.shape[1]}")

print(f"{c_}Number of rows in lecture data: {r_}{lec_data.shape[0]}\n{c_}Number of columns in lecture data: {r_}{lec_data.shape[1]}")

print(f"{m_}Number of rows in question data: {r_}{que_data.shape[0]}\n{m_}Number of columns in question data: {r_}{que_data.shape[1]}")

print(f"{b_}Number of rows in submission data: {r_}{sample.shape[0]}\n{b_}Number of columns in submission data:{r_}{sample.shape[1]}")
train_data.head().style.applymap(lambda x:"background-color:lightgreen")
train_data.info()
#looking for null values

train_data.isna().sum()
lec_data.head()
que_data.head()
def countplot(column):

    plt.figure(dpi=100)

    sns.countplot(train_data[column])

    plt.show()
countplot('user_answer')
countplot('answered_correctly')
countplot('content_type_id')
countplot("prior_question_had_explanation")
plt.figure(dpi=100)

sns.distplot(train_data[~train_data["prior_question_elapsed_time"].isna()]["prior_question_elapsed_time"],color='yellow')

plt.show()
def distribution1(column,color,n=40):

    df = train_data[column].value_counts().reset_index()

    df.columns = [column,'count']

    df[column] = df[column].astype(str) + '-'

    df = df.sort_values(['count'],ascending=False)



    plt.figure(figsize=(15,10))

    plt.subplot(121)

    sns.distplot(df['count'],color=color)



    plt.subplot(122)

    sns.barplot(x='count',y=column,data=df[:n],orient='h')

    plt.show()
distribution1("user_id","purple") 
distribution1("content_id","red")
distribution1("task_container_id","green",n=50)
answered_correctly = train_data.groupby(['user_id'])['answered_correctly'].agg(['sum','count']).reset_index()

answered_correctly = answered_correctly[answered_correctly['count']>=10]

answered_correctly['user_id'] = answered_correctly['user_id'].astype(str) + "_"

answered_correctly['percentage'] = (answered_correctly['sum'] / answered_correctly['count']) * 100

answered_correctly = answered_correctly.sort_values(['percentage'],ascending=False)



plt.figure(figsize=(7,10))

sns.barplot(x='percentage',y='user_id',data=answered_correctly[:50],orient='h');
df_correct_user_answers = train_data[train_data['answered_correctly']==1]['user_answer']

df_incorrect_user_answers = train_data[train_data['answered_correctly']==0]['user_answer']



plt.figure(figsize=(15,7))

plt.subplot(121)

sns.countplot(df_correct_user_answers)

plt.title("Correctly answered user answers")

plt.subplot(122)

sns.countplot(df_incorrect_user_answers)

plt.title("Incorrectl answered user answers");
sorted_user_id_timestamp = train_data.sort_values(['user_id','timestamp'])

train_data["time_required_to_answer"] = sorted_user_id_timestamp.groupby('user_id')['prior_question_elapsed_time'].shift(periods=-1)

responce_time_correct = train_data[train_data['answered_correctly']==1].groupby('user_answer')['time_required_to_answer'].mean()

responce_time_incorrect = train_data[train_data['answered_correctly']==0].groupby('user_answer')['time_required_to_answer'].mean()



plt.figure(figsize=(15,7))

plt.subplot(121)

sns.barplot(responce_time_incorrect.index,responce_time_correct.values)

plt.title("Responce time for correctly answered answers")

plt.subplot(122)

sns.barplot(responce_time_correct.index,responce_time_correct.values)

plt.title("Responce time for incorrectly answered answers");
train_data["timespend"]=train_data.groupby('user_id')["timestamp"].transform(lambda x: x.max() - x.min())
plt.figure(dpi=100)

plt.hist(train_data.timespend,color='red')

plt.xlabel("timespend");
train_data = train_data.sort_values("timestamp").reset_index(drop=True)

train_data['interaction_count'] = 1

train_data['interaction_count'] = train_data.groupby("user_id")['interaction_count'].transform('cumsum')

train_data['correct_answers_till_now'] = train_data.groupby('user_id')['answered_correctly'].transform('cumsum')

train_data['accuracy_per_timestamp'] = train_data['correct_answers_till_now']*100 / train_data['interaction_count']



f = plt.figure(figsize=(7,7))

sns.set_style(style="whitegrid")

sns.despine(f, left=True, bottom=True)

sns.scatterplot(x='timestamp',y='accuracy_per_timestamp',data=train_data,hue='content_type_id');

plt.xlabel("accuracy of user")

plt.ylabel("timestamp");
top_user = train_data[train_data.user_id == 7171715]

top_user = pd.merge(top_user,que_data,left_on='content_id',right_on='question_id',how='left')

top_user = pd.merge(top_user,lec_data,left_on='content_id',right_on='lecture_id',how='left')

top_user.head()
print("number of question and lecture attented by user: ",top_user.content_id.nunique())

print("number of questions attented by user; ",top_user.question_id.nunique())

print("number of lectures attented by user: ",top_user[top_user.content_type_id==1].content_id.nunique())

print("number of bundles attented by user: ",top_user.bundle_id.nunique())
sns.set_style(style="darkgrid")

plt.figure(figsize=(15,7))

plt.subplot(121)

sns.lineplot(x='timestamp',y='accuracy_per_timestamp',data=top_user[:100],color='green')

plt.subplot(122)

sns.lineplot(x='timestamp',y='accuracy_per_timestamp',data=top_user[100:],color='green');
dtype = {

    'row_id': 'int64',

    'timestamp': 'int64',

    'user_id': 'int32',

    'content_id': 'int16',

    'answered_correctly': 'int8',

    'prior_question_elapsed_time': 'float16',

    'prior_question_had_explanation': 'boolean'

}



train_data = pd.read_csv(

    '/kaggle/input/riiid-test-answer-prediction/train.csv',

    usecols = dtype.keys(),

    dtype=dtype, 

    index_col = 0,

    nrows = 10**7

)
# feature_data = train_data.iloc[:int(0.9 * len(train_data))]

# train_data = train_data.iloc[int(0.9 * len(train_data)):]



train_data = train_data.sort_values("timestamp").reset_index(drop=True)

train_data['time_required_to_answer'] = train_data.groupby('user_id')['prior_question_elapsed_time'].shift(-1)

train_data['question_has_explanation'] = train_data.groupby('user_id')['prior_question_had_explanation'].shift(-1)



tag = que_data["tags"].str.split(" ", n = 10, expand = True) 

tag.columns = ['tags1','tags2','tags3','tags4','tags5','tags6']



que_data =  pd.concat([que_data,tag],axis=1).drop(['tags'],axis=1)

que_data['tags1'] = pd.to_numeric(que_data['tags1'], errors='coerce',downcast='integer').fillna(-1)

que_data['tags2'] = pd.to_numeric(que_data['tags2'], errors='coerce',downcast='integer').fillna(-1)

que_data['tags3'] = pd.to_numeric(que_data['tags3'], errors='coerce',downcast='integer').fillna(-1)



train_data = pd.merge(train_data,que_data,left_on='content_id',right_on='question_id',how='left')

train_data['timespend'] = train_data.groupby("user_id")["timestamp"].transform(lambda x: (x.max() - x.min())/1000)

train_answered_question = train_data[train_data['answered_correctly']!=-1]



grouped_by_user_id = train_answered_question.groupby("user_id")

df1 = grouped_by_user_id.agg({'answered_correctly':['mean','count','std','median']}).copy()

df1.columns =  ['mean_user_accuracy', 'questions_answered', 'std_user_accuracy', 'median_user_accuracy']



del grouped_by_user_id

gc.collect()
grouped_by_content_id = train_answered_question.groupby("content_id")

df2 = grouped_by_content_id.agg({'answered_correctly':['mean','count','std','median']}).copy()

df2.columns =  ['mean_accuracy', 'questions_asked', 'std_accuracy', 'median_accuracy']



# df3 = grouped_by_content_id.agg({'timespend':['mean','std','median']}).copy()

# df3.columns =  ['mean_time', 'std_time', 'median_time']



del grouped_by_content_id

del train_answered_question

# del feature_data

gc.collect()
features = [

    #numerical columns

    'mean_user_accuracy', 

    'questions_answered',

    'std_user_accuracy', 

    'median_user_accuracy',

    'mean_accuracy', 

    'questions_asked',

    'std_accuracy', 

    'median_accuracy',

    'prior_question_elapsed_time', 

    'time_required_to_answer',

    #categorical columns

    'prior_question_had_explanation',

    'question_has_explanation',

    'timespend',

    'bundle_id',

    'tags1',

    'tags2',

    'tags3',

#     'mean_time',

#     'std_time',

#     'median_time',

]

target_column = 'answered_correctly'
train_data = train_data[train_data[target_column] != -1]

train_data = train_data.merge(df1, how='left', on='user_id')

train_data = train_data.merge(df2, how='left', on='content_id')

# train_data = train_data.merge(df3, how='left', on='content_id')



train_data['prior_question_had_explanation'] = train_data['prior_question_had_explanation'].fillna(value = False).astype(bool)

train_data['question_has_explanation'] = train_data['question_has_explanation'].fillna(value = False).astype(bool)



train_data = train_data.fillna(value = -1)



target = train_data[target_column].values

train_data = train_data[features]

train_data = train_data.replace([np.inf, -np.inf], np.nan)

train_data = train_data.fillna(-1)
train_data.head()
scaler = StandardScaler()

train_data = scaler.fit_transform(train_data)
import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.metrics import roc_auc_score
class Model(nn.Module):

    def __init__(self,input_size,output_size):

        super(Model,self).__init__()

        self.batch_norm1 = nn.BatchNorm1d(input_size)

        self.dropout1 = nn.Dropout(0.3)

        self.linear1 = nn.utils.weight_norm(nn.Linear(input_size,128))

        

        self.batch_norm2 = nn.BatchNorm1d(128)

        self.dropout2 = nn.Dropout(0.2)

        self.linear2 = nn.utils.weight_norm(nn.Linear(128,32))

        

        self.batch_norm3 = nn.BatchNorm1d(32)

        self.dropout3 = nn.Dropout(0.2)

        self.linear3 = nn.utils.weight_norm(nn.Linear(32,output_size))

        

    def forward(self,xb):

        x = self.batch_norm1(xb)

        x = self.dropout1(x)

        x = F.leaky_relu(self.linear1(x))

        

        x = self.batch_norm2(x)

        x = self.dropout2(x)

        x = F.leaky_relu(self.linear2(x))

        

        x = self.batch_norm3(x)

        x = self.dropout3(x)

        return self.linear3(x)





# class Model(nn.Module):

#     def __init__(self,input_dim,output_dim):

#         super(Model,self).__init__()

#         self.layer1 = nn.Linear(input_dim,100)

#         self.layer2 = nn.Linear(100,100)

#         self.layer3 = nn.Linear(100,output_dim)

            

#     def forward(self,xb):

#         x1 =  F.leaky_relu(self.layer1(xb))

#         x1 =  F.leaky_relu(self.layer2(x1))

#         return self.layer3(x1)

config = {

    "epochs":15,

    "train_batch_size":50_000,

    "valid_batch_size":50_000,

    "test_batch_size":50_000,

    "nfolds":3,

    "learning_rate":0.001,

}
def run(plot_losses=True):

  

    def train_loop(train_loader,model,loss_fn,device,optimizer,lr_scheduler=None):

        model.train()

        total_loss = 0

        for i, (inputs, targets) in enumerate(train_loader):

            inputs = inputs.to(device)

            targets = targets.to(device)

            

            optimizer.zero_grad()

            outputs = model(inputs)



            loss = loss_fn(outputs,targets)

            loss.backward()

                

            total_loss += loss.item()



            optimizer.step()

            if lr_scheduler != None:

                lr_scheduler.step(loss.item())

                    

        total_loss /= len(train_loader)

        return total_loss

    

    def valid_loop(valid_loader,model,loss_fn,device):

        model.eval()

        total_loss = 0

        predictions = list()

        

        for i, (inputs, targets) in enumerate(valid_loader):

            inputs = inputs.to(device)

            targets = targets.to(device)

            outputs = model(inputs)                 



            loss = loss_fn(outputs,targets)

            predictions.extend(outputs.sigmoid().detach().cpu().numpy())

            

            total_loss += loss.item()

        total_loss /= len(valid_loader)

            

        return total_loss,np.array(predictions)    

    



    kfold = StratifiedKFold(n_splits=config['nfolds'])

    

    #for storing losses of every fold

    fold_train_losses = list()

    fold_valid_losses = list()

    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"{device} is used")

    

    def loss_fn(outputs,targets):

        targets = targets.view(-1,1)

        return nn.BCEWithLogitsLoss()(outputs,targets)

    

    #kfold

    for k , (train_idx,valid_idx) in enumerate(kfold.split(train_data,target)):

      

        x_train,x_valid,y_train,y_valid = train_data[train_idx,:],train_data[valid_idx,:],target[train_idx],target[valid_idx]

        

        input_dim = x_train.shape[1]

        output_dim = 1



        model = Model(input_dim,output_dim)

        model.to(device)

        

        train_tensor = torch.tensor(x_train,dtype=torch.float)

        y_train_tensor = torch.tensor(y_train,dtype=torch.float)



        train_ds = TensorDataset(train_tensor,y_train_tensor)

        train_dl = DataLoader(train_ds,

                             batch_size = config["train_batch_size"],

                             shuffle=True,

                              num_workers = 4,

                              pin_memory=True

                             )



        valid_tensor = torch.tensor(x_valid,dtype=torch.float)

        y_valid_tensor = torch.tensor(y_valid,dtype=torch.float)



        valid_ds = TensorDataset(valid_tensor,y_valid_tensor)

        valid_dl = DataLoader(valid_ds,

                             batch_size =config["valid_batch_size"],

                             shuffle=False,

                              num_workers = 4,

                              pin_memory=True,

                             )

        

        optimizer = optim.Adam(model.parameters(),lr=config['learning_rate'])

        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, eps=1e-4, verbose=True)



        print(f"Fold {k}")

        best_loss = 999

        

        train_losses = list()

        valid_losses = list()

        start = time.time()

        for i in range(config["epochs"]):

            train_loss = train_loop(train_dl,model,loss_fn,device,optimizer,lr_scheduler=lr_scheduler)

            valid_loss,predictions = valid_loop(valid_dl,model,loss_fn,device)

            

            train_losses.append(train_loss)

            valid_losses.append(valid_loss)

            end = time.time()

            epoch_time = end - start

            start = end

            

            score = roc_auc_score(y_valid,predictions)

                          

            print(f"epoch:{i} Training loss:{train_loss} | Validation loss:{valid_loss} | Score: {score:.4f} | epoch time {epoch_time:.2f} ")

            

            if valid_loss <= best_loss:

                print(f"{g_}Validation loss Decreased from {best_loss} to {valid_loss}{sr_}")

                best_loss = valid_loss

                torch.save(model.state_dict(),f'model{k}.bin')

                

        fold_train_losses.append(train_losses)

        fold_valid_losses.append(valid_losses)

        

        

    if plot_losses == True:

        plt.figure(figsize=(20,14))

        for i, (t,v) in enumerate(zip(fold_train_losses,fold_valid_losses)):

            plt.subplot(2,5,i+1)

            plt.title(f"Fold {i}")

            plt.plot(t,label="train_loss")

            plt.plot(v,label="valid_loss")

            plt.legend()

        plt.show()   
run()
def inference(test):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    all_prediction = np.zeros((test.shape[0],1))

    

    for i in range(config["nfolds"]):

        

        input_dim = test.shape[1]

        output_dim = 1

        model = Model(input_dim,output_dim)

        model.load_state_dict(torch.load(f"model{i}.bin"))

        

        predictions = list()

        model.to(device)

        test_tensor = torch.tensor(test,dtype=torch.float)

        test_dl = DataLoader(test_tensor,

                        batch_size=config["test_batch_size"],

                        shuffle=False)

    

        with torch.no_grad():

            for i, inputs in enumerate(test_dl):

                inputs = inputs.to(device, dtype=torch.float)

                outputs= model(inputs) 

                predictions.extend(outputs.sigmoid().cpu().detach().numpy())



        all_prediction += np.array(predictions)/config['nfolds']

        

    return all_prediction
import riiideducation

env = riiideducation.make_env()
iter_test = env.iter_test()
for (test_data,sample_prediction_df) in iter_test:

    test_data = pd.merge(test_data,que_data,left_on='content_id',right_on='question_id',how='left')

    test_data['timespend'] = test_data.groupby("user_id")['timestamp'].transform(lambda x: x.max() - x.min())

    test_data['time_required_to_answer'] = test_data.groupby('user_id')['prior_question_elapsed_time'].shift(-1)

    test_data['question_has_explanation'] = test_data.groupby('user_id')['prior_question_had_explanation'].shift(-1)

    test_data = test_data.merge(df1,how='left',on='user_id')

    test_data = test_data.merge(df2,how='left',on='content_id')

#     test_data = test_data.merge(df3,how='left',on='content_id')



    test_data['prior_question_had_explanation'] = test_data['prior_question_had_explanation'].fillna(value = False).astype(bool)

    test_data['question_has_explanation'] = test_data['question_has_explanation'].fillna(value = False).astype(bool)



    test_data.fillna(value = -1, inplace = True)

    test_transform = scaler.transform(test_data[features])

    test_data['answered_correctly'] = inference(test_transform)

    env.predict(test_data.loc[test_data['content_type_id']==0,['row_id','answered_correctly']])
sub = pd.read_csv("./submission.csv")

sub.shape