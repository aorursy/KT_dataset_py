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
input_df=pd.read_csv('../input/creditcard.csv')
pd.isnull(input_df).sum()
input_df.describe()
X=input_df.iloc[:,1:30]

Y=input_df.iloc[:,30]
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1)

print('length of X_train =',len(X_train))

print('length of X_test =',len(X_test))
#X_val,X_test,Y_val,Y_test=train_test_split(X_test,Y_test,test_size=0.5)

#print('length of X_test =',len(X_test))
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

X_train=scaler.fit_transform(X_train)

#X_val=scaler.transform(X_val)

X_test=scaler.transform(X_test)
cov_matrix=np.cov(X_train,rowvar=False)
from sklearn.naive_bayes import GaussianNB

gnb=GaussianNB()

gnb.fit(X_train,Y_train)
Y_pred=gnb.predict_proba(X_test)

print(gnb.classes_)

Y_pred=Y_pred[:,1]

from sklearn.metrics import roc_auc_score

rs=roc_auc_score(Y_test,Y_pred)
print(rs)
p_df=pd.DataFrame(Y_pred)

p_df.rename(columns={0:'target'},inplace=True)

#p_df['target']=p_df['target'].astype(float)

output=pd.DataFrame({'ID_code':test_df['ID_code'],'target':p_df['target']})

print(output)

output.to_csv('out.csv',index=False)