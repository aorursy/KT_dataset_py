import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv("../input/data-mining-assignment-2/train.csv")

test = pd.read_csv("../input/data-mining-assignment-2/test.csv")
train.isna().sum()
train.Class.describe()
categorical = []

for i in train.columns[1:]:

    if(train.dtypes[i]=='object'):

        categorical.append(i)

        

categorical
for i in categorical:

    print(i,":",train[i].unique())
train['col11'].replace({

    'No':0,

    'Yes':1

    },inplace=True)

test['col11'].replace({

    'No':0,

    'Yes':1

    },inplace=True)
train['col44'].replace({

    'No':0,

    'Yes':1

    },inplace=True)

test['col44'].replace({

    'No':0,

    'Yes':1

    },inplace=True)
train['col37'].replace({

    'Male':0,

    'Female':1

    },inplace=True)

test['col37'].replace({

    'Male':0,

    'Female':1

    },inplace=True)
train['col56'].replace({

    'Low':0,

    'Medium':1,

    'High':2

    },inplace=True)

test['col56'].replace({

    'Low':0,

    'Medium':1,

    'High':2

    },inplace=True)
train_onehot = train.copy()

train_onehot = pd.get_dummies(train_onehot, columns=['col2'], prefix = ['col2'])

train_onehot.head()
test_onehot = test.copy()

test_onehot = pd.get_dummies(test_onehot,columns=['col2'],prefix=['col2'])

test_onehot.head()
corr = train_onehot.corr()

corr
col_rem_corr = []

for i in train_onehot.columns[1:]:

    if(corr['Class'][i] < 0.1):

        col_rem_corr.append(i)

        

len(col_rem_corr)
train_rem = train_onehot.drop(col_rem_corr,1)

test_rem = test_onehot.drop(col_rem_corr,1)
y_train = train_rem['Class']

x_train = train_rem.drop(['Class','ID'],axis=1)

x_test = test_rem.drop(['ID'],axis=1)
#Scaling

from sklearn import preprocessing



min_max_scaler = preprocessing.StandardScaler()

scaler = min_max_scaler.fit(x_train)

train_scaled = scaler.transform(x_train)

train_scaled = pd.DataFrame(train_scaled)

test_scaled = scaler.transform(x_test)

test_scaled = pd.DataFrame(test_scaled)
#Random Forest

from sklearn.ensemble import RandomForestClassifier



train_acc =[]

for i in range(1,20):

    for j in range(20,200,20):

        rf = RandomForestClassifier(n_estimators=j,max_depth=i,random_state=1)

        rf.fit(x_train,y_train)

        acc_train = rf.score(x_train,y_train)

        train_acc.append([i,j,acc_train])
train_acc
rfc = RandomForestClassifier(n_estimators=100,max_depth = 8,random_state=1)

rfc.fit(x_train,y_train)

y_pred_rf = rfc.predict(x_test)
out = [[test['ID'][i],y_pred_rf[i]] for i in range(300)]

out_df = pd.DataFrame(data=out,columns=['ID','Class'])

out_df.to_csv(r'out_1_11.csv',index=False)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

 csv = df.to_csv(index=False)

 b64 = base64.b64encode(csv.encode())

 payload = b64.decode()

 html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

 html = html.format(payload=payload,title=title,filename=filename)

 return HTML(html)

create_download_link(out_df)
