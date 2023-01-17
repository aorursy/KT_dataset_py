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
df = pd.read_csv('../input/breast-cancer/breast_cancer.txt')
df.head()
from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer(missing_values=-1)
data = df.as_matrix()
data[data=='?'] = '-1'
data = data.astype(int)
data = my_imputer.fit_transform(data)
data
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(data[:,:9], data[:,9], test_size=0.33)
from sklearn.neural_network import MLPClassifier
clf_mlp1 = MLPClassifier(hidden_layer_sizes=(15,),activation='logistic',learning_rate_init=0.01,max_iter=300,verbose=0)
train_x = data[:,:9]
train_y = data[:,9]
clf_mlp1.fit(train_x,train_y)
score1 = clf_mlp1.score(test_x,test_y)
score1
clf_mlp2 = MLPClassifier(hidden_layer_sizes=(15,8),activation='relu',learning_rate_init=0.01,max_iter=300,verbose=0)
clf_mlp2.fit(train_x,train_y)
score2 = clf_mlp2.score(test_x,test_y)
score2
clf_mlp3 = MLPClassifier(hidden_layer_sizes=(15,10,5),activation='relu',learning_rate_init=0.01,max_iter=300,verbose=0)
clf_mlp3.fit(test_x,test_y)
score3 = clf_mlp3.score(train_x,train_y)
score3
df_hv = pd.read_csv('../input/house-votes/house-votes-84.csv')
data_hv = df_hv.as_matrix()
train_x_hv, test_x_hv, train_y_hv, test_y_hv = train_test_split(data_hv[:,:16], data_hv[:,16], test_size=0.33)
clf_hv = MLPClassifier(hidden_layer_sizes=(25,),activation='logistic',learning_rate_init=0.01,max_iter=300,verbose=0)
clf_hv.fit(train_x_hv,train_y_hv)
score_hv = clf_hv.score(test_x_hv,test_y_hv)
score_hv
df_hp = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
df_hp.head()
df_hp = df_hp.drop(['Id'],axis=1)
cols = list(df_hp)
my_imputer = SimpleImputer(strategy='most_frequent')
df_hp_new = pd.DataFrame(my_imputer.fit_transform(df_hp),columns=cols)
df_hp.head()
df_hp_new.head()
for c in cols:
    if df_hp[c].dtype=='object':
        for u in df_hp_new[c].unique(): 
            df_hp_new[c+str('_')+u] = list((df_hp_new[c]==u)*1)
        df_hp_new = df_hp_new.drop([c],axis=1)
y_all = df_hp_new['SalePrice'].values
X_all = df_hp_new.drop(['SalePrice'],axis=1).as_matrix().astype(int)
X_all.shape
from sklearn.model_selection import train_test_split
train_x_hp, test_x_hp, train_y_hp, test_y_hp = train_test_split( X_all, y_all, test_size=0.1, random_state=5)
from sklearn.model_selection import cross_validate
from sklearn.neural_network import MLPRegressor
res_list = []
data_hp = np.zeros([1460,289])
data_hp[:,:288] = np.copy(X_all)
data_hp[:,288] = np.copy(y_all)
for i in range(5):
    clf_hp1 = MLPRegressor(hidden_layer_sizes=(100,),learning_rate_init=0.01,max_iter=1000,verbose=0)
    train_x_hp = data_hp[:,:288]
    train_y_hp = data_hp[:,288]
    cv_res = cross_validate(clf_hp1,train_x_hp,train_y_hp,cv=10)
    res_list.append(np.mean(np.array(cv_res['test_score'])))
    np.random.shuffle(data)
    print(str(i)+" done")
mlp_bc1 = np.mean(res_list)
mlp_bc_std1 = np.std(np.array(res_list))
print('Average accuracy : '+str(mlp_bc1)+' Standard Deviation : '+str(mlp_bc_std1))
res_list = []
data_hp = np.zeros([1460,289])
data_hp[:,:288] = np.copy(X_all)
data_hp[:,288] = np.copy(y_all)
for i in range(5):
    clf_hp1 = MLPRegressor(hidden_layer_sizes=(100,50),learning_rate_init=0.01,max_iter=1000,verbose=0)
    train_x_hp = data_hp[:,:288]
    train_y_hp = data_hp[:,288]
    cv_res = cross_validate(clf_hp1,train_x_hp,train_y_hp,cv=10)
    res_list.append(np.mean(np.array(cv_res['test_score'])))
    np.random.shuffle(data)
    print(str(i)+" done")
mlp_bc2 = np.mean(res_list)
mlp_bc_std2 = np.std(np.array(res_list))
print('Average accuracy : '+str(mlp_bc2)+' Standard Deviation : '+str(mlp_bc_std2))
import matplotlib.pyplot as plt
plt.bar(x=['MLP1','MLP2'],height=[mlp_bc1,mlp_bc2],color=['b','g'])
plt.show()
from sklearn.linear_model import Perceptron
clf_slp = Perceptron(max_iter=300)
clf_slp.fit(train_x,train_y)
score4 = clf_slp.score(test_x,test_y)
score4
from sklearn.linear_model import Perceptron
clf_slp1 = Perceptron(max_iter=300)
clf_slp1.fit(train_x_hv,train_y_hv)
score4 = clf_slp1.score(test_x_hv,test_y_hv)
score4
