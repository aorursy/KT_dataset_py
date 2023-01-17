import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import catboost as cb
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 600)

import os
print(os.listdir("../input"))
testing = pd.read_csv('../input/poker-hand-testing.txt',names=['S1','C1','S2','C2','S3','C3','S4','C4','S5','C5','hand'])
training = pd.read_csv('../input/poker-hand-training-true.data',names=['S1','C1','S2','C2','S3','C3','S4','C4','S5','C5','hand'])
testing.head()
training.head()
print(training.shape)
print(testing.shape)
X = training.drop(['hand'],axis=1)
y = training.hand
Xte = testing.drop(['hand'],axis=1)
yte = testing.hand
#Catboost
#Random seeds are fixed at 1234 to make scores reproducible
def CVandTest(X,y,Xte,yte):
    #80/20 train test split cross-validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    cat = cb.CatBoostClassifier(
        loss_function='MultiClassOneVsAll',
        random_seed = 1234
    )
    cat_model = cat.fit(X_train,y_train,verbose=False) 

    cat_predA = cat_model.predict(X_test, prediction_type='Class')
    cat_predLL = cat_model.predict(X_test, prediction_type='Probability')

    print("CV accuracy: {}".format(accuracy_score(y_test,cat_predA)))
    print("CV logloss: {}".format(log_loss(y_test,cat_predLL)))
    
    #Training with all X,y data, testing with Xte,yte
    cat_model = cat.fit(X,y,verbose=False)

    cat_predAt = cat_model.predict(Xte, prediction_type='Class')
    cat_predLLt = cat_model.predict(Xte, prediction_type='Probability')

    print("Test accuracy: {}".format(accuracy_score(yte,cat_predAt)))
    print("Test logloss: {}".format(log_loss(yte,cat_predLLt)))
    
    return (cat_predA, cat_predLL, cat_predAt, cat_predLLt)
(cat_predA, cat_predLL, cat_predAt, cat_predLLt) = CVandTest(X,y,Xte,yte)
#Logarithmic Histogram
plt.hist((np.reshape(cat_predAt,(yte.shape[0],)),yte),bins=[0,1,2,3,4,5,6,7,8,9,10],log=True)
plt.legend(labels=('preds','test'))
plt.show()
#Combine training and testing to create features for both together at the same time
all_data = pd.concat([training, testing]).reset_index(drop=True)
#Number of same suites and ranks sparse matrix creation using bincount2D_vectorized
#https://stackoverflow.com/questions/46256279/bin-elements-per-row-vectorized-2d-bincount-for-numpy
def bincount2D_vectorized(a):    
    N = a.max()+1
    a_offs = a + np.arange(a.shape[0])[:,None]*N
    return np.bincount(a_offs.ravel(), minlength=a.shape[0]*N).reshape(-1,N)

S = all_data.iloc[:,[0,2,4,6,8]].astype(int)
S = pd.DataFrame(bincount2D_vectorized(S.values),columns=['suit0','suit1','suit2','suit3','suit4'])
all_data = pd.merge(all_data, S, how='left', left_index=True, right_index=True).drop(['suit0'],axis=1)
#bincount starts counting from 0, but our suits start at 1.  Dropping suit0.

R = all_data.iloc[:,np.arange(1,10,2)].astype(int)
cols = ['rank{}'.format(x) for x in range(0,14,1)]
R = pd.DataFrame(bincount2D_vectorized(R.values),columns=cols)
all_data = pd.merge(all_data, R, how='left', left_index=True, right_index=True).drop(['rank0'],axis=1)
#bincount starts counting from 0, but our ranks start at 1.  Dropping rank0.

all_data.head()
#Splitting back to train/test
X = all_data.iloc[:25010,:].drop(['hand'],axis=1)
Xte = all_data.iloc[25010:,:].drop(['hand'],axis=1)
(cat_predA, cat_predLL, cat_predAt, cat_predLLt) = CVandTest(X,y,Xte,yte)
#Logarithmic Histogram
plt.hist((np.reshape(cat_predAt,(yte.shape[0],)),yte),bins=[0,1,2,3,4,5,6,7,8,9,10],log=True)
plt.legend(labels=('preds','test'))
plt.show()
#Number of ranks with x's is another bincount problem
R = all_data.loc[:,['rank{}'.format(n) for n in range(1,14,1)]].astype(int)
R = pd.DataFrame(bincount2D_vectorized(R.values),columns=['rankCount{}'.format(n) for n in range(0,5,1)])
all_data = pd.merge(all_data, R, how='left', left_index=True, right_index=True).drop(['rankCount0'],axis=1)

all_data.head()
#Differences between consecutive ranks: rank2 - rank1, rank3 - rank2, ..., rank13 - rank12. And rank1 - rank13
all_data['diff1_13'] = all_data['rank1'] - all_data['rank13']
for i in range(2,14,1):
    all_data['diff{}_{}'.format(i,i-1)] = all_data['rank{}'.format(i)] - all_data['rank{}'.format(i-1)]

all_data.tail()
#Splitting back to train/test
X = all_data.iloc[:25010,:].drop(['hand'],axis=1)
Xte = all_data.iloc[25010:,:].drop(['hand'],axis=1)
(cat_predA, cat_predLL, cat_predAt, cat_predLLt) = CVandTest(X,y,Xte,yte)
#Logarithmic Histogram
plt.hist((np.reshape(cat_predAt,(yte.shape[0],)),yte),bins=[0,1,2,3,4,5,6,7,8,9,10],log=True)
plt.legend(labels=('preds','test'))
plt.show()