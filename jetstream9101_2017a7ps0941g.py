import numpy as np

import pandas as pd

from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier
dataset = pd.read_csv('/kaggle/input/eval-lab-2-f464/train.csv')

dataset_test = pd.read_csv('/kaggle/input/eval-lab-2-f464/test.csv')
dataset.head()
dataset.info()
X_train = dataset.drop(columns=['id','class'])

X_train = X_train[['chem_0','chem_1','chem_4','chem_5','chem_6','attribute']]

y_train = dataset['class']



X_test = dataset_test.drop(columns=['id'])

X_test = X_test[['chem_0','chem_1','chem_4','chem_5','chem_6','attribute']]

dataset.corr()
rfc = RandomForestClassifier(n_estimators=2000)

rfc.fit(X_train,y_train)



y_pred = rfc.predict(X_test)



df_final = pd.DataFrame({'id':dataset_test['id'], 'class':np.round(y_pred)})

df_final.to_csv('submission.csv',index=False)