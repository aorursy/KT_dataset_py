# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df1 = pd.read_csv('/kaggle/input/cat-in-the-dat/train.csv')

df2 = pd.read_csv('/kaggle/input/cat-in-the-dat/test.csv')

y = df1['target'].values

df = pd.concat([df1,df2],axis = 0,sort = False)
df.drop(['id','target'],inplace = True,axis = 1)
#%%bash

#pip install category-encoders
df1.columns
df_train = df.iloc[:df1.shape[0],:]

df_test = df.iloc[df1.shape[0]:,:]

from category_encoders import TargetEncoder

ec = TargetEncoder(cols=['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4', 'nom_0', 'nom_1',

       'nom_2', 'nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9',

       'ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5', 'day', 'month'])

x_opt = ec.fit_transform(df_train.iloc[:,:],y)

x_test = ec.transform(df_test.iloc[:,:])
x_opt.shape,x_test.shape,y.shape
x_test.head()
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_opt_sc = sc.fit_transform(x_opt)

x_test_sc = sc.fit_transform(x_test)
from sklearn.model_selection import StratifiedKFold

skd = StratifiedKFold(n_splits=10,random_state=17)
from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(hidden_layer_sizes=(23,10,5),activation='relu',early_stopping=True,batch_size=256,learning_rate='invscaling',random_state=17)
for train,test in skd.split(x_opt_sc,y):

    x_train,x_t = x_opt_sc[train],x_opt_sc[test]

    y_train,y_t = y[train],y[test]
%%time

clf.fit(x_train,y_train)
pred = clf.predict_proba(x_t)
from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score

print(roc_auc_score(y_t,pred[:,1]))

print(roc_auc_score(y,clf.predict_proba(x_opt)[:,1]))

print(confusion_matrix(y_t,clf.predict(x_t)))

confusion_matrix(y,clf.predict(x_opt_sc))
df3 = pd.DataFrame()

df3['id'] = df2['id']

df3['target'] = clf.predict_proba(x_test_sc)[:,1]
df3.to_csv('submit2.csv',index=False)
df3