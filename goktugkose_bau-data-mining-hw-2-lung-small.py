import os

import warnings

import numpy as np 

import pandas as pd 

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from mlxtend.feature_selection import SequentialFeatureSelector

from sklearn.metrics import confusion_matrix,accuracy_score,r2_score,precision_score,recall_score,f1_score

from scipy.io import loadmat

print(os.listdir("../input/"))

warnings.filterwarnings("ignore")
mat = loadmat('../input/lung_small.mat')

df = pd.DataFrame(np.hstack((mat['X'],mat['Y'])))
df.head()
print('There are', df.shape[0], 'rows and', df.shape[1], 'columns in the dataset.')
X = df.iloc[:,:-1]

y = df.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0,test_size=0.2)
lr = LogisticRegression()

forward = SequentialFeatureSelector(lr,

                                    k_features = 5,

                                    forward = True,

                                    floating = False,

                                    scoring = 'accuracy',

                                    n_jobs=4,

                                    cv = 0)

backward = SequentialFeatureSelector(lr,

                                    k_features = 5,

                                    forward = False,

                                    floating = False,

                                    scoring = 'accuracy',

                                    n_jobs=4,

                                    cv = 0)
forward.fit(X_train,y_train)

forward_cols = list(forward.k_feature_idx_)

lr.fit(X_train.loc[:,forward_cols],y_train)
print('Sequential Forward Feature Selection (k=5):')

print(forward.k_feature_idx_)

print('CV score: ')

print(forward.k_score_)
pred_train_forward= lr.predict(X_train.loc[:,forward_cols])

print('Training accuracy on selected features: %.3f' % accuracy_score(y_train, pred_train_forward ))

print('Training precision on selected features: %.3f' % precision_score(y_train,pred_train_forward ,average='micro'))

print('Training recall on selected features: %.3f' % recall_score(y_train,pred_train_forward ,average='micro'))

print('Training f1_score on selected features: %.3f' % f1_score(y_train,pred_train_forward ,average='micro'))
pred_valid_forward= lr.predict(X_test.loc[:,forward_cols])

print('Validation accuracy on selected features: %.3f' % accuracy_score(y_test, pred_valid_forward))

print('Validation precision on selected features: %.3f' % precision_score(y_test,pred_valid_forward,average='micro'))

print('Validation recall on selected features: %.3f' % recall_score(y_test,pred_valid_forward,average='micro'))

print('Validation f1_score on selected features: %.3f' % f1_score(y_test,pred_valid_forward,average='micro'))
backward.fit(X_train,y_train)

backward_cols = list(backward.k_feature_idx_)

lr.fit(X_train.loc[:,backward_cols],y_train)
print('Sequential Backward Feature Selection (k=5):')

print(backward.k_feature_idx_)

print('CV score: ')

print(backward.k_score_)
pred_train_backward= lr.predict(X_train.loc[:,backward_cols])

print('Training accuracy on selected features: %.3f' % accuracy_score(y_train, pred_train_backward ))

print('Training precision on selected features: %.3f' % precision_score(y_train,pred_train_backward ,average='micro'))

print('Training recall on selected features: %.3f' % recall_score(y_train,pred_train_backward ,average='micro'))

print('Training f1_score on selected features: %.3f' % f1_score(y_train,pred_train_backward ,average='micro'))
pred_valid_backward= lr.predict(X_test.loc[:,backward_cols])

print('Validation accuracy on selected features: %.3f' % accuracy_score(y_test, pred_valid_backward))

print('Validation precision on selected features: %.3f' % precision_score(y_test,pred_valid_backward,average='micro'))

print('Validation recall on selected features: %.3f' % recall_score(y_test,pred_valid_backward,average='micro'))

print('Validation f1_score on selected features: %.3f' % f1_score(y_test,pred_valid_backward,average='micro'))