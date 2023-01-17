import numpy as np

import pandas as pd

import sklearn

import seaborn as sns
data = pd.read_csv("/kaggle/input/eval-lab-2-f464/train.csv")

data.head()
test_data=pd.read_csv('/kaggle/input/eval-lab-2-f464/test.csv')

test_data.head()
data.describe()

data.columns

cols = [ 'chem_1', 'chem_2', 'chem_4', 'chem_5','chem_6', 'attribute']

X = data[cols]

y = data['class']

#from sklearn.model_selection import train_test_split

#X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size = 0.33)

X_test = test_data[cols]
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

X_train_sc = scaler.fit_transform(X)

X_test_sc = scaler.transform(X_test)
from xgboost import XGBClassifier

xg = XGBClassifier()

xg.fit(X_train_sc,y)



y_pred = xg.predict(X_test_sc)
from sklearn.ensemble import RandomForestClassifier

rand_for = RandomForestClassifier(n_estimators=3000)



rand_for.fit(X_train_sc,y)

y_pred = rand_for.predict(X_test_sc)


from xgboost import XGBClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import VotingClassifier, RandomForestClassifier



estimators = [('rf', RandomForestClassifier(n_estimators=3000)), ('det', DecisionTreeClassifier()), ('xgb', XGBClassifier(n_estimators=3000))]



# soft_voter = VotingClassifier(estimators=estimators, voting='soft').fit(X_train,y_train)

hard_voter = VotingClassifier(estimators=estimators, voting='hard').fit(X_train_sc,y)

ypred = hard_voter.predict(X_test_sc);
y_pred
y_pred=np.rint(y_pred)

len(y_pred)
final= pd.DataFrame(test_data['id'])
y_pred = y_pred.astype(int)
final['class'] = y_pred.astype(int)
final.head()
final.to_csv('predicted.csv',encoding='utf-8',index=False)
y_pred