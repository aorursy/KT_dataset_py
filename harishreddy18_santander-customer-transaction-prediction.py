#importing required packages

import numpy as np

import pandas as pd
#importing data

train=r'../input/train.csv'

test=r'../input/test.csv'

train_df=pd.read_csv(train)

test_df=pd.read_csv(test)

#dropping 1st two columns to get features

X_train=train_df.iloc[:,2:].values

Y_train=train_df.iloc[:,1:2].values

Y_train=Y_train.ravel()

X_rtest=test_df.iloc[:,1:].values
#scaling and splitting the training data

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_rtest = sc_X.transform(X_rtest)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

from imblearn.over_sampling import SMOTE

sm=SMOTE(random_state=42)

X_train, Y_train = sm.fit_resample(X_train, Y_train)
from lightgbm.sklearn import LGBMClassifier

lgbm=LGBMClassifier(n_estimators=100000,num_leaves=13)

lgbm.fit(X_train,Y_train,eval_set=(X_test,Y_test),early_stopping_rounds=3000,eval_metric='accuracy',verbose=1000)
#print(X_rtest.shape)

Y_pred=lgbm.predict_proba(X_rtest)

#Y_pred=np.greater(Y_pred,0.5,dtype=np.float64)

p_df=pd.DataFrame(Y_pred)

p_df.rename(columns={0:'target'},inplace=True)

#p_df['target']=p_df['target'].astype(float)

output=pd.DataFrame({'ID_code':test_df['ID_code'],'target':p_df['target']})

print(output)

output.to_csv('out.csv',index=False)