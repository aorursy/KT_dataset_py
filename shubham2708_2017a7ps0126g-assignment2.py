# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_validate

from sklearn.metrics import make_scorer

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import f1_score

from sklearn.metrics import roc_auc_score





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/data-mining-assignment-2/train.csv')

df2 = pd.read_csv('../input/data-mining-assignment-2/test.csv')

lala = pd.read_csv('../input/data-mining-assignment-2/test.csv')

data = df

temp = df2
df = df.drop(columns=['ID'])

df2 = df2.drop(columns=['ID'])

data = data.drop(columns=['ID'])

temp = temp.drop(columns=['ID'])
le = LabelEncoder()

for i in df.columns:

    if(df[i].dtype == object):

        df[i] = df[i].astype('str')

        df[i] = le.fit_transform(df[i])

        

for i in df2.columns:

    if(df2[i].dtype == object):

        df2[i] = df2[i].astype('str')

        df2[i] = le.fit_transform(df2[i])
df_train = df.drop(['Class'], axis=1)

df_test = df2

shifts = []

for col in df_train.columns:

    X_train = df_train[[col]]

    X_test = df_test[[col]]

    X_train["target"] = 0

    X_test["target"] = 1

    X_tmp = pd.concat((X_train, X_test),ignore_index=True).drop(['target'], axis=1)

    y_tmp= pd.concat((X_train.target, X_test.target),ignore_index=True)

 

    X_train_tmp, X_test_tmp, y_train_tmp, y_test_tmp = train_test_split(X_tmp,y_tmp,test_size=0.25,random_state=1)

    rf = RandomForestClassifier(n_estimators=50,n_jobs=-1,max_features=1,min_samples_leaf=5,max_depth=5,random_state=1)

    rf.fit(X_train_tmp, y_train_tmp)

    y_pred_tmp = rf.predict_proba(X_test_tmp)[:, 1]

    score = roc_auc_score(y_test_tmp, y_pred_tmp)

    shifts.append((max(np.mean(score), 1 - np.mean(score)) - 0.5) * 2)

print(shifts)
remove_lst_shift = []

for i in range(len(shifts)):

    if shifts[i] == 1.0:

         remove_lst_shift.append('col'+str(i))

remove_lst_shift
y=data['Class']

X=df.drop(['Class'],axis=1)
xyz = pd.DataFrame()

for i in X:

    xyz[i] = df2[i]
scaler=StandardScaler()

X_N=scaler.fit(X).transform(X)

X_N=pd.DataFrame(X_N,columns=X.columns)



scaler=StandardScaler()

df2=scaler.fit(xyz).transform(xyz)

df2=pd.DataFrame(df2,columns=xyz.columns)
X_train, X_test, y_train, y_test = train_test_split(X_N, y, test_size=0.20, random_state=0)
# mdep_lst = []

# for i in range(5,15):

#     mdep_lst.append(i)

    

# # n_est_lst = []

# # for i in range(40,101):

# #     n_est_lst.append(i)



# msplit_lst = []

# for i in range(2,10):

#     msplit_lst.append(i)



# rand_st = []

# for i in range(0, 51, 10):

#     rand_st.append(i)





# rf_temp = RandomForestClassifier(n_estimators = 100)        #Initialize the classifier object



# parameters = {'min_samples_split': msplit_lst, 'max_depth': mdep_lst, 'random_state':rand_st}    #Dictionary of parameters



# scorer = make_scorer(f1_score, average = 'micro')         #Initialize the scorer using make_scorer



# grid_obj = GridSearchCV(rf_temp, parameters, scoring=scorer)         #Initialize a GridSearchCV object with above parameters,scorer and classifier



# grid_fit = grid_obj.fit(X_train, y_train)        #Fit the gridsearch object with X_train,y_train



# best_rf = grid_fit.best_estimator_         #Get the best estimator. For this, check documentation of GridSearchCV object



# print(grid_fit.best_params_)
# best params : 'min_samples_split': 2, 'max_depth': 14, 'random_state':23



best_rf = RandomForestClassifier(n_estimators=100, max_depth=14, min_samples_split=2,random_state=23)

best_rf.fit(X_train, y_train)

best_rf.score(X_test,y_test) 
importances = best_rf.feature_importances_

imp_lst = []

for i in range(0, len(importances)):

    imp_lst.append(['col'+str(i) , importances[i]])

sorted_imp_lst = sorted(imp_lst, key = lambda x: x[1])

remove_lst = []

for i in range(0, 53):

    flag = 0

#     print(sorted_imp_lst[i][0])

    if sorted_imp_lst[i][0] == 'col59':

        print(sorted_imp_lst[i][0])

#     for j in data[sorted_imp_lst[i][0]]:

#         for k in temp[sorted_imp_lst[i][0]]:

#             if k == j:

#                 flag = 1

#                 break

#         if flag == 1:

#             break

#     if flag == 0:

    if sorted_imp_lst[i][0] in remove_lst_shift:

        remove_lst.append(sorted_imp_lst[i][0])

print(remove_lst)
new_X = pd.DataFrame()

new_df2 = pd.DataFrame()

for i in range(0, len(sorted_imp_lst)):

    if sorted_imp_lst[i][0] not in remove_lst:

        new_X[sorted_imp_lst[i][0]] = data[sorted_imp_lst[i][0]]

        new_df2[sorted_imp_lst[i][0]] = temp[sorted_imp_lst[i][0]]

len(new_X.columns)
le = LabelEncoder()

for i in new_X.columns:

    if(new_X[i].dtype == object):

        new_X[i] = new_X[i].astype('str')

        new_X[i] = le.fit_transform(new_X[i])

        

for i in new_df2.columns:

    if(new_df2[i].dtype == object):

        new_df2[i] = new_df2[i].astype('str')

        new_df2[i] = le.fit_transform(new_df2[i])
scaler=StandardScaler()

neww_X=scaler.fit(new_X).transform(new_X)

neww_X=pd.DataFrame(neww_X,columns=new_X.columns)



scaler=StandardScaler()

neww_df2=scaler.fit(new_df2).transform(new_df2)

neww_df2=pd.DataFrame(neww_df2,columns=new_df2.columns)
X_train, X_test, y_train, y_test = train_test_split(neww_X, y, test_size=0.20, random_state=0)
best_rf = RandomForestClassifier(n_estimators=100, max_depth=14, min_samples_split=2, random_state=23)

best_rf.fit(X_train, y_train)

best_rf.score(X_test,y_test) 
submit_pred = best_rf.predict(neww_df2)
final = pd.DataFrame({'ID': lala['ID'][:]})

final['Class'] = submit_pred[:]
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "pred.csv"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(final)