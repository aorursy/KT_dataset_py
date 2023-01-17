import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
data_orig=pd.read_csv("../input/dm-dataset-a2/train.csv",sep=',')

data=data_orig

data_test=pd.read_csv("../input/dm-dataset-a2/test_1.csv",sep=',')

data_test_cpy=data_test
data1=data.drop(['ID','Worker Class','Enrolled','MLU','Reason','Area','MSA','REG','MOVE','Live','PREV','Teen','Fill'],1)

data_t1=data_test.drop(['ID','Worker Class','Enrolled','MLU','Reason','Area','MSA','REG','MOVE','Live','PREV','Teen','Fill'],1)
data1=data1.drop(['MIC','MOC','State','Detailed','COB FATHER', 'COB MOTHER', 'COB SELF'],1)

data_t1=data_t1.drop(['MIC','MOC','State','Detailed','COB FATHER', 'COB MOTHER', 'COB SELF'],1)
data1['Hispanic']=data1['Hispanic'].replace('?','HA')

data_t1['Hispanic']=data_t1['Hispanic'].replace('?','HA')
data1.info()
data_t1.info()
data2=pd.get_dummies(data1, columns=['Schooling','Married_Life','Cast','Hispanic','Sex','Full/Part','Tax Status','Summary','Citizen'])

data_t2=pd.get_dummies(data_t1, columns=['Schooling','Married_Life','Cast','Hispanic','Sex','Full/Part','Tax Status','Summary','Citizen'])
data2.info()
data_t2.info()
X=data2.drop(['Class'],1)
from sklearn import preprocessing

#Performing Min_Max Normalization

min_max_scaler = preprocessing.MinMaxScaler()

np_scaled = min_max_scaler.fit_transform(X)

X_sc = pd.DataFrame(np_scaled)

X_sc.columns=X.columns

X_sc.info()
#Performing Min_Max Normalization

min_max_scaler1 = preprocessing.MinMaxScaler()

np_scaled1 = min_max_scaler.fit_transform(data_t2)

X_test = pd.DataFrame(np_scaled1)

X_test.columns=data_t2.columns

X_test.info()
Y=data2['Class']
from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(X_sc, Y, test_size=0.20, random_state=42)
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)

X_sm, Y_sm = sm.fit_sample(X_sc, Y)



sm2= SMOTE(random_state=42)

X_train_sm, Y_train_sm= sm2.fit_sample(X_train, Y_train)
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score, roc_curve, recall_score, classification_report
from sklearn.linear_model import LogisticRegression

lg = LogisticRegression(solver = 'liblinear', C = 4, random_state = 42)

lg.fit(X_train_sm,Y_train_sm)

Y_pred_LR = lg.predict(X_val)

print( roc_auc_score(Y_val, Y_pred_LR) )
from sklearn.linear_model import LogisticRegression 

lg1 = LogisticRegression(solver = 'liblinear', C = 4, random_state = 42) 

lg.fit(X_sm,Y_sm) 

Y_test=lg.predict(X_test) 

result=pd.DataFrame(Y_test).astype(int) 

result.columns=['Class'] 

result['Class'].value_counts()
id_col=data_test['ID']

result_final=pd.concat([id_col,result],axis=1).reindex()

result_final.columns=['ID','Class']

result_final
result_final.to_csv('2016A7PS0098G_4.csv',index=False)

result_final.info()
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

create_download_link(result_final)