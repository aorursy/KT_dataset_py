import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import sys
data_orig = pd.read_csv("../input/dm2-hershal/train.csv", sep=',')

data_orig_2 = pd.read_csv("../input/dm2-hershal/test_1.csv",sep=',')

data = data_orig

data2 = data_orig_2
data = data.replace({'?':None})

data = data.drop(['ID'],1)

data2 = data2.replace({'?':None})

data2 = data2.drop(['ID'],1)
# listcol= ['Hispanic','COB FATHER','COB MOTHER','COB SELF']

# for i in listcol:

#     data[i] = data[i].fillna(data[i].mode()[0])

#     data2[i] = data2[i].fillna(data2[i].mode()[0])

for i in data.columns.tolist():

    data[i] = data[i].fillna(data[i].mode()[0])

    if i !="Class":

        data2[i] = data2[i].fillna(data2[i].mode()[0])
for i in data.columns.tolist():

    if data[i].isnull().any():

        data = data.drop([i],1)

        if i !="Class":

            data2 = data2.drop([i],1)
# droplist = ['Weaks','COB SELF','COB MOTHER','COB FATHER','Detailed','Schooling']

# # 'Hispanic','Married_Life','Cast','Full/Part','Tax Status','Summary','Citizen'

# for i in droplist:

#     print(str(len(data[i].unique()))+" "+str(len(data2[i].unique()))+" "+i)

# for i in droplist:

#     data = data.drop([i],1)

#     data2 = data2.drop([i],1)

data = data.drop(['Detailed'],1)

data2 = data2.drop(['Detailed'],1)
data.info()
for i in data.columns.tolist():

    if data[i].dtype == 'object':

        data = pd.get_dummies(data,columns=[i])

        data2 = pd.get_dummies(data2,columns=[i])
from sklearn.model_selection import train_test_split

y=data['Class']

x=data.drop(['Class'],axis=1)

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1, random_state=42)
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42, ratio = 1)

x_train_res, y_train_res = sm.fit_sample(x_train, y_train)

x_full,y_full = sm.fit_sample(x,y)

# x_full,y_full = x,y

# x_train_res, y_train_res = x_train, y_train

zero = 0

one = 0

for i in y_train_res:

    if i == 0:

        zero += 1

    else:

        one +=1

print(str(zero)+" "+str(one))
#plp
from sklearn import preprocessing

#Performing Min_Max Normalization

min_max_scaler = preprocessing.MinMaxScaler()

np_scaled = min_max_scaler.fit_transform(x_train_res)

x_train_res = pd.DataFrame(np_scaled)

np_scaled_val = min_max_scaler.transform(x_val)

x_val = pd.DataFrame(np_scaled_val)

min_max_scaler2 = preprocessing.MinMaxScaler()

min_max_scaler3 = preprocessing.MinMaxScaler()

np_scaled2 = min_max_scaler2.fit_transform(data2)

x_test = pd.DataFrame(np_scaled2)

np_scaled3 = min_max_scaler3.fit_transform(x_full)

x_full = pd.DataFrame(np_scaled3)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,roc_auc_score
i = 5

lg = LogisticRegression(C = i,random_state = 42,solver='liblinear')

lg.fit(x_train_res,y_train_res)

lg.score(x_val,y_val)

print(str(i)+" "+str(roc_auc_score(y_val,lg.predict(x_val)))+"\r"),
y_pred_LR = lg.predict(x_val)

print(confusion_matrix(y_val, y_pred_LR))
print(classification_report(y_val, y_pred_LR))
roc_auc_score(y_val,y_pred_LR)
y_val.value_counts()
lg_F = LogisticRegression(C = 5,random_state = 42,solver='liblinear')

lg_F.fit(x_full,y_full)
res1 = pd.DataFrame(lg_F.predict(x_test))

# res1 = res1.replace({0:1,1:0})

res2 = pd.DataFrame(data_orig_2['ID'])

final = pd.concat([res2,res1], axis=1).reindex()

final.columns = ['ID','Class']

final.head()
final['Class'].value_counts()
final.to_csv('sub15_lr_os100.csv', index = False)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html='<a download="{filename}" href="data:text/csv;base64,{payload}"target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(final)