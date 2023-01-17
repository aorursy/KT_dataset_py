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
import numpy as np

import pandas as pd

import os

train=pd.read_csv('Desktop/Training/Kaggle/train.csv')

test=pd.read_csv('Desktop/Training/Kaggle/test.csv')





train.head(n=10).T



print('Train:', train.shape)

print('Test:', test.shape)



train['target'].value_counts()  #unbalanced data







import gc



from sklearn.model_selection import  train_test_split



from sklearn.metrics import roc_curve, roc_auc_score, auc, confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression



target=train['target']



train_input=train.drop(['target','ID_code'],axis=1)

test_input=test.drop(['ID_code'],axis=1)



features=list(train_input.columns)



X_train, X_test, Y_train, Y_test=train_test_split(train_input,target,test_size=0.2, random_state=1000)



print('Train:',X_train.shape)

print('Test:',X_test.shape)



logist=LogisticRegression(C=0.001, class_weight='balanced')



logist.fit(X_train, Y_train)



logist_pred=logist.predict_proba(X_test)[:,1]



logist_pred



def performance(Y_test, logist_pred):

    logist_pred_var=[0 if i<0.5 else 1 for i in logist_pred]

    print('Confusion matrix:')

    print(confusion_matrix(Y_test, logist_pred_var))

    fpr,tpr,thresholds=roc_curve(Y_test,logist_pred,pos_label=1)

    print('AUC:')

    print(auc(fpr,tpr))

    

performance(Y_test,logist_pred)



logist_pred_test=logist.predict_proba(test_input)[:,1]



submit=test[['ID_code']]

submit['target']=logist_pred_test



submit.head()



submit.to_csv('Downloads/sample_submission.csv',index=False)

df = pd.read_csv('Downloads/sample_submission.csv')



df['pred_probability'] = df.target>0.5

df['pred_probability'] = (df['pred_probability'] ==True).astype(int)



df.to_csv('Downloads/sample_submission1.csv',index=False)