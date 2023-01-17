# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import sys

import torch

import torch.nn as nn

import torch.optim as optim

from torch.utils.data import Dataset, DataLoader



from sklearn.preprocessing import StandardScaler    

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.naive_bayes import GaussianNB

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn import svm

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

from sklearn import metrics





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Read procurement data

df_procurement = pd.read_csv('/kaggle/input/prozorro-public-procurement-dataset/Competitive_procurements.csv')

df_procurement.head()
df_procurement.info()

#Some columns format are not correct
df_procurement['lot_announce_date'] = pd.to_datetime(df_procurement['lot_announce_date'])
df_procurement.isnull().sum()
df_procurement.describe()
df_procurement.nunique()
#Handle Missing values

df = df_procurement[['organizer_name','organizer_code']].drop_duplicates().reset_index(drop=True)

df[df['organizer_code'].isna()]
df[df['organizer_name']==df.at[70, 'organizer_name']]

df_procurement['organizer_code']=df_procurement['organizer_code'].fillna(20517231.0)

df_procurement.isnull().sum()

#Finding: There is only one NaN in original data which is not a missing value
df = df_procurement[['participant_name','participant_code']].drop_duplicates().reset_index(drop=True)

df[df['participant_code'].isna()]
df_procurement['participant_code'] = df_procurement['participant_code'].fillna(df_procurement['participant_name'])
df_procurement.isnull().sum()

#All missing values are handled
#Number of procurements, organizers and participant over the years

df = pd.concat([df_procurement.groupby('lot_announce_year').lot_id.nunique(),

          df_procurement.groupby('lot_announce_year').organizer_code.nunique(),

          df_procurement.groupby('lot_announce_year').participant_code.nunique()], axis=1)

df.plot.bar()

#Findings: Number of procurements, organizers and participants increased from 2015 to 2017. Then number of procurements starts to decrease but organizers and participants remains same.
#What changes over the years.

##Top 5 lot_cpv each year under each lot_procur_type.

df = pd.concat([df_procurement.groupby('lot_announce_year').lot_procur_type.nunique(),

          df_procurement.groupby('lot_announce_year').lot_cpv.nunique(),

          df_procurement.groupby('lot_announce_year').lot_cpv_2_digs.nunique(),

               df_procurement.groupby('lot_announce_year').lot_cpv_4_digs.nunique()], axis=1)

df.plot.bar()

#sns.pairplot(df_procurement, x_vars=['lot_initial_value'], y_vars=['lot_final_value'], kind='reg')

#df_suppliers = pd.read_csv('/kaggle/input/prozorro-public-procurement-dataset/Suppliers.csv')#

#df_suppliers.shape

#df_suppliers.head()
df_procurement.columns
df=df_procurement[['lot_announce_date','organizer_code','lot_cpv','lot_cpv_4_digs', 'lot_cpv_2_digs']]

df=df.drop_duplicates()

df['temp']=1

df['Org_lot_cpv_cumsum'] = df.groupby(['organizer_code','lot_cpv','lot_cpv_4_digs', 'lot_cpv_2_digs'])['temp'].cumsum()

df=df.drop(['temp'],axis=True)

df_procurement=df_procurement.merge(df,on=['lot_announce_date','organizer_code','lot_cpv','lot_cpv_4_digs', 'lot_cpv_2_digs'])



df=df_procurement[['lot_announce_date','organizer_code','lot_procur_type']]

df=df.drop_duplicates()

df['temp']=1

df['Org_lot_procur_type_cumsum'] = df.groupby(['organizer_code','lot_procur_type'])['temp'].cumsum()

df=df.drop(['temp'],axis=True)

df_procurement=df_procurement.merge(df,on=['lot_announce_date','organizer_code','lot_procur_type'])
df_procurement['temp']=1

df_procurement['Org_Par_Connection'] = df_procurement.groupby(['organizer_code','participant_code'])['temp'].cumsum()

df_procurement['Org_Par_lot_cpv_cumsum'] = df_procurement.groupby(['organizer_code','participant_code','lot_cpv','lot_cpv_4_digs', 'lot_cpv_2_digs'])['temp'].cumsum()

df_procurement['no_of_participant']=df_procurement.groupby(['organizer_code','lot_id'])['temp'].transform("count")

df_procurement = df_procurement.drop(['temp'], axis=1)

df_procurement['Org_Par_supplier_cnt'] = df_procurement.groupby(['organizer_code','participant_code'])['supplier_dummy'].cumsum()

df_procurement['business_value'] = df_procurement['supplier_dummy']*df_procurement['lot_final_value']

df_procurement['Org_Par_business_value'] = df_procurement.groupby(['organizer_code','participant_code'])['business_value'].cumsum()

df_procurement = df_procurement.drop(['business_value'], axis=1)

df_procurement['Org_Par_region'] = (df_procurement['organizer_region']== df_procurement['participant_region']).astype(int)

df_procurement.describe()

#Findings: Very few participant become frequent supplier to organizers over time

#Findings: Very few participant made strong connections with Organizer over time
df = df_procurement.groupby(['organizer_code','participant_code'])['lot_announce_date'].first().reset_index()

df.columns = ['organizer_code','participant_code', 'joining_date']

df_procurement = pd.merge(df_procurement,df,on=['organizer_code','participant_code'])

df_procurement['no_days_of_connectison']=(df_procurement['lot_announce_date']-df_procurement['joining_date']) / np.timedelta64(1, 'D')

#df_procurement['no_of_participant']=df_procurement.groupby(['organizer_code','lot_id','participant_code']).sum()

#df_procurement['no_days_of_time_pass'] = (df_procurement['lot_announce_date']-df_procurement['lot_announce_date'].min()) / np.timedelta64(1, 'D')

df_procurement.head(6)
df_procurement.columns
#1. Shift Org_Par_supplier_cnt and Org_Par_business_value one row down within group inorder to avoid seeing future

df_procurement['Org_Par_supplier_cnt_shift']=df_procurement.groupby(['organizer_code','participant_code'])['Org_Par_supplier_cnt'].shift()

df_procurement['Org_Par_business_value_shift']=df_procurement.groupby(['organizer_code','participant_code'])['Org_Par_business_value'].shift()

df_procurement['Org_Par_supplier_cnt_shift']=df_procurement['Org_Par_supplier_cnt_shift'].fillna(0)

df_procurement['Org_Par_business_value_shift']=df_procurement['Org_Par_business_value_shift'].fillna(0)
sns.countplot(x=df_procurement['supplier_dummy'])
#X=df_procurement[['lot_initial_value','Org_lot_cpv_cumsum',

#       'Org_lot_procur_type_cumsum', 'Org_Par_Connection',

#       'Org_Par_lot_cpv_cumsum', 'no_of_participant', 'Org_Par_supplier_cnt',

#       'Org_Par_business_value', 'Org_Par_region',

#       'no_days_of_connectison']]

#Y=df_procurement['supplier_dummy']

#X['lot_initial_value']=((X['lot_initial_value']-X['lot_initial_value'].min())/(X['lot_initial_value'].max()-X['lot_initial_value'].min()))

#X['Org_Par_business_value']=((X['Org_Par_business_value']-X['Org_Par_business_value'].min())/(X['Org_Par_business_value'].max()-X['Org_Par_business_value'].min()))

#X.head()
X=df_procurement[['lot_procur_type','Org_lot_cpv_cumsum',

       'Org_lot_procur_type_cumsum', 'Org_Par_Connection',

       'Org_Par_lot_cpv_cumsum', 'no_of_participant', 'Org_Par_supplier_cnt',

       'Org_Par_business_value', 'Org_Par_region',

       'no_days_of_connectison']]

Y=df_procurement['supplier_dummy']

X['Org_Par_business_value'] = pd.cut(X['Org_Par_business_value'], bins=10)

X.head()

X=pd.get_dummies(X)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=10)
lr = LogisticRegression()

lr.fit(X_test, Y_test)

y_prob=lr.predict_proba(X_test)[:,1]

y_pred=np.where(y_prob>0.5,1,0)

confusion_matrix = metrics.confusion_matrix(Y_test, y_pred)

print(confusion_matrix)

auc_roc = metrics.roc_auc_score(Y_test, y_pred)

print(auc_roc)
#LDA

lda = LinearDiscriminantAnalysis()

lda.fit(X_test, Y_test)

y_pred=lda.predict(X_test)

confusion_matrix = metrics.confusion_matrix(Y_test, y_pred)

print(confusion_matrix)
SVM = svm.LinearSVC()

SVM.fit(X_test, Y_test)

y_pred=SVM.predict(X_test)

confusion_matrix = metrics.confusion_matrix(Y_test, y_pred)

confusion_matrix
RF = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)

RF.fit(X_test, Y_test)

y_pred=RF.predict(X_test)

confusion_matrix = metrics.confusion_matrix(Y_test, y_pred)

print(confusion_matrix)

print(classification_report(Y_test, y_pred))
RF = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)

RF.fit(X_test, Y_test)

y_pred=RF.predict(X_test)

confusion_matrix = metrics.confusion_matrix(Y_test, y_pred)

print(confusion_matrix)

print(classification_report(Y_test, y_pred))
NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100, 20), random_state=1)

NN.fit(X_test, Y_test)

y_pred=NN.predict(X_test)

confusion_matrix = metrics.confusion_matrix(Y_test, y_pred)

confusion_matrix
EPOCHS = 10

BATCH_SIZE = 64

LEARNING_RATE = 0.001





#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=69)

y_train = Y_train

y_test = Y_test



## train data

class trainData(Dataset):

    

    def __init__(self, X_data, y_data):

        self.X_data = X_data

        self.y_data = y_data

        

    def __getitem__(self, index):

        return self.X_data[index], self.y_data[index]

        

    def __len__ (self):

        return len(self.X_data)





train_data = trainData(torch.Tensor(X_train.values), torch.FloatTensor(y_train.values))

## test data    

class testData(Dataset):

    

    def __init__(self, X_data):

        self.X_data = X_data

        

    def __getitem__(self, index):

        return self.X_data[index]

        

    def __len__ (self):

        return len(self.X_data)

    



test_data = testData(torch.FloatTensor(X_test.values))



train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_loader = DataLoader(dataset=test_data, batch_size=1)



class binaryClassification(nn.Module):

    def __init__(self):

        super(binaryClassification, self).__init__()

        # Number of input features is 10.

        self.layer_1 = nn.Linear(22, 64) #10,22

        self.layer_2 = nn.Linear(64, 64)

        self.layer_out = nn.Linear(64, 1) 

        

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(p=0.1)

        self.batchnorm1 = nn.BatchNorm1d(64)

        self.batchnorm2 = nn.BatchNorm1d(64)

        

    def forward(self, inputs):

        x = self.relu(self.layer_1(inputs))

        x = self.batchnorm1(x)

        x = self.relu(self.layer_2(x))

        x = self.batchnorm2(x)

        x = self.dropout(x)

        x = self.layer_out(x)

        

        return x

    

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)



model = binaryClassification()

model.to(device)

print(model)

criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)



def binary_acc(y_pred, y_test):

    y_pred_tag = torch.round(torch.sigmoid(y_pred))



    correct_results_sum = (y_pred_tag == y_test).sum().float()

    acc = correct_results_sum/y_test.shape[0]

    acc = torch.round(acc * 100)

    

    return acc





model.train()

for e in range(1, EPOCHS+1):

    epoch_loss = 0

    epoch_acc = 0

    for X_batch, y_batch in train_loader:

        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()

        

        y_pred = model(X_batch)

        #print(y_pred)

        #print(y_batch.unsqueeze(1))

        

        loss = criterion(y_pred, y_batch.unsqueeze(1))

        acc = binary_acc(y_pred, y_batch.unsqueeze(1))

        

        loss.backward()

        optimizer.step()

        

        epoch_loss += loss.item()

        epoch_acc += acc.item()

        



    print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')

    

    

    

y_pred_list = []

model.eval()

with torch.no_grad():

    for X_batch in test_loader:

        X_batch = X_batch.to(device)

        y_test_pred = model(X_batch)

        y_test_pred = torch.sigmoid(y_test_pred)

        y_pred_tag = torch.round(y_test_pred)

        y_pred_list.append(y_pred_tag.cpu().numpy())



y_pred_list = [a.squeeze().tolist() for a in y_pred_list]



confusion_matrix(y_test, y_pred_list)
print(classification_report(y_test, y_pred_list))