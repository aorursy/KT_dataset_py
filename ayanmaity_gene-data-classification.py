# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/gene-expression/data_set_ALL_AML_train.csv')
train_df.head()
train_df = train_df.drop(['Gene Description','Gene Accession Number'],axis=1)
call_cols = ['call']+['call.'+str(i) for i in range(1,38)]
train_df = train_df.drop(call_cols,axis=1)
train_df = train_df.T
actual_df = pd.read_csv('../input/gene-expression/actual.csv')
train_df['cancer'] = list(actual_df['cancer'])[:38]
train_df_ALL = train_df[[col for col in train_df.columns if col!='cancer']][train_df['cancer']=='ALL']
train_df_AML = train_df[[col for col in train_df.columns if col!='cancer']][train_df['cancer']=='AML']
train_df_ALL.shape
test_df = pd.read_csv('../input/gene-expression/data_set_ALL_AML_independent.csv')
test_df = test_df.drop(['Gene Description','Gene Accession Number'],axis=1)
call_cols = ['call']+['call.'+str(i) for i in range(1,34)]
test_df = test_df.drop(call_cols,axis=1)
test_df = test_df.T
test_df['cancer'] = list(actual_df['cancer'])[38:]
total_df = train_df.append(test_df)
total_df.shape
w_mi_df = pd.read_csv('../input/w-micsv/w_mi.csv')
v_mi = list(w_mi_df['keys'])
total_df1 = total_df.copy()
total_df.columns
y = total_df['cancer'].values
total_df = total_df.drop([v for v in list(total_df.columns) if v not in v_mi],axis=1)
from sklearn.preprocessing import MinMaxScaler as scaler
min_scaler = scaler()
total_norm = min_scaler.fit_transform(total_df.as_matrix())
total_norm = total_df.as_matrix()
total_norm.shape
X_train = total_norm[:38]
y_train = y[:38]
X_test = total_norm[38:]
y_test = y[38:]
y.shape
X_all = []
X_aml = []
for i in range(X_train.shape[0]):
    if y_train[i]=='ALL':
        X_all.append(X_train[i])
    else :
        X_aml.append(X_train[i])
X_all = np.array(X_all)
X_aml = np.array(X_aml)
mu_all = np.mean(X_all,axis=0)
mu_aml = np.mean(X_aml,axis=0)
std_all = np.std(X_all,axis=0) 
std_aml = np.std(X_aml,axis=0) 
y_test
def similarity(t,mu_all,mu_aml,std_all,std_aml):
    sim_all = np.exp(-np.power(t-mu_all,2)/(2*std_all*std_all))
    sim_aml = np.exp(-np.power(t-mu_aml,2)/(2*std_aml*std_aml))
    return sim_all,sim_aml

w_mi_dic = {w_mi_df['keys'][i]:w_mi_df['values'][i] for i in range(w_mi_df.shape[0])}
w_mi = []
for v in list(total_df.columns):
    w_mi.append(w_mi_dic[v])
w_mi = np.array(w_mi)

acc=0
y_pred=[]
for ind in range(38):
    t1 = X_train[ind]
    sim_all1,sim_aml1 = np.zeros([7129,]),np.zeros([7129,])
    sim_all1,sim_aml1 = similarity(t1,mu_all,mu_aml,std_all,std_aml)
    S_all1 = np.sum(sim_all1*np.exp(w_mi))
    S_aml1 = np.sum(sim_aml1*np.exp(w_mi))
    if S_all1>=S_aml1:
        y_pred.append('ALL')
    else:
        y_pred.append('AML')
np.sum(y_pred==y_train)/38
y_pred=[]
for ind in range(34):
    t1 = X_test[ind]
    sim_all1,sim_aml1 = np.zeros([7129,]),np.zeros([7129,])
    sim_all1,sim_aml1 = similarity(t1,mu_all,mu_aml,std_all,std_aml)
    S_all1 = np.sum(sim_all1*np.exp(w_mi))
    S_aml1 = np.sum(sim_aml1*np.exp(w_mi))
    if S_all1>=S_aml1:
        y_pred.append('ALL')
    else:
        y_pred.append('AML')
np.sum(y_pred==y_test)/34
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
total_df1 = total_df1.drop(['cancer'],axis=1)
X_full = total_df1.as_matrix()
from sklearn.preprocessing import MinMaxScaler as scaler
min_scaler = scaler()
X_full = min_scaler.fit_transform(X_full)
X_f_train = X_full[:38]
X_f_test = X_full[38:]
X_f_train.shape
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(verbose=1)
mlp.fit(X_f_train,y_train)
mlp.score(X_f_train,y_train),mlp.score(X_f_test,y_test)
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_f_train,y_train)
nb.score(X_f_train,y_train),nb.score(X_f_test,y_test)
