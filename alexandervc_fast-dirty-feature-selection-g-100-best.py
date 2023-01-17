# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session




import time 

import matplotlib.pyplot as plt

import seaborn as sns



df = pd.read_csv('/kaggle/input/lish-moa/train_features.csv',index_col = 0)  

df
df_test = pd.read_csv('/kaggle/input/lish-moa/test_features.csv',index_col = 0)  

df_test
y = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv',index_col = 0 )

y
y_additional = pd.read_csv('/kaggle/input/lish-moa/train_targets_nonscored.csv',index_col = 0 )

y_additional
y_sum = y.sum(axis = 1)

y_sum.value_counts()
y_additional.sum(axis = 1).value_counts()
mode_which_part_to_process = 'full'

if mode_which_part_to_process == 'full':

    # consider only gene expression part 

    list_features_names = [c for c in df.columns if ('c-' in c) or ('g-' in c)]

    X = df[list_features_names ].values

if mode_which_part_to_process == 'genes':

    # consider only gene expression part 

    list_features_names =[c for c in df.columns if 'g-' in c]

    X = df[list_features_names ].values

if mode_which_part_to_process == 'c':

    # consider only gene expression part 

    list_features_names =[c for c in df.columns if 'c-' in c]

    X = df[list_features_names ].values



print(len([c for c in df.columns if 'g-' in c] ), 'genes count ')

X_original_save = X.copy()

print(X.shape)
y_01 = (y_sum > 0 ).astype(float)

y_01.value_counts()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y_01, test_size=0.33, random_state=42)

X_train.shape, X_test.shape, y_train.shape, y_test.shape
from sklearn.metrics import roc_auc_score

df_features_stat = pd.DataFrame()

for i in range(X.shape[1]):

    v = X[:,i]

    r = roc_auc_score(y_01, v )

    df_features_stat.loc[list_features_names[i],'rocauc'] = r

    df_features_stat.loc[list_features_names[i],'rocauc Abs'] = np.abs(r-0.5) + 0.5

    

    

df_features_stat.sort_values(by = 'rocauc Abs', ascending=False,  inplace = True) 

plt.figure(figsize = (25,6))

plt.plot( df_features_stat['rocauc Abs'].values,'o')

plt.title('Features rocauc Abs')

plt.grid()

plt.show()

df_features_stat#.head(10)
from sklearn import linear_model

from sklearn.metrics import roc_auc_score



start = time.time()

clf = linear_model.LogisticRegression(penalty='l1', solver='liblinear',

                                      tol=1e-6, max_iter=int(1e6),

                                      warm_start=True ) #,

                                      # intercept_scaling=10000.)

    

coefs_ = []

for c in [0.001, .002, 0.003, 0.005, 0.008,  0.01, 0.1 , 1, 1e10]:

    clf.set_params(C=c)

    clf.fit(X_train, y_train)

    coefs_.append(clf.coef_.ravel().copy())

    print("This took %0.3fs" % (time.time() - start))

    p = clf.predict_proba(X_train)[:,1]

    r_train = roc_auc_score(y_train, p )

    p = clf.predict_proba(X_test)[:,1]

    r = roc_auc_score(y_test, p )

    print('c=',c, 'rocauc test:', np.round(r,3) , 'Number of features selected:', (clf.coef_.ravel() !=  0).sum() , 'rocauc train:', np.round(r_train,3)) 

    if (clf.coef_.ravel() !=  0).sum()  < 100:

        print( np.array(list_features_names)[ (clf.coef_.ravel() !=  0) ] )

        corr_matr = np.corrcoef(X[:, (clf.coef_.ravel() !=  0) ] .T)

        print(np.triu(corr_matr,1).max(), np.triu(corr_matr,1).min() )  

        

    print()

# print(coefs_)
m1 = ( (coefs_[-1] !=  0) * (~(coefs_[-2] !=  0) ) )#.sum()

np.array(list_features_names)[ m1 ] , m1.sum()
m2 = ( (coefs_[-1] !=  0) * (~(coefs_[-3] !=  0) ) )#.sum()

np.array(list_features_names)[ m2 ] , m2.sum()