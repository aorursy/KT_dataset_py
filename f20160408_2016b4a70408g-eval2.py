import numpy as np

import pandas as pd

import warnings

warnings.filterwarnings("ignore")

from matplotlib import pyplot as plt

%matplotlib inline

import seaborn as sns
df = pd.read_csv('/kaggle/input/eval-lab-2-f464/train.csv')

df.info()
df['class'].value_counts()
df.head()
df.isnull().sum()
sns.heatmap(df.corr())
cols_to_drop = ['id','chem_2','chem_3','chem_5','chem_7']

df_final = df.drop(cols_to_drop, axis=1)
X = df_final.drop('class',axis=1)

y = df_final['class']



print (X.shape, y.shape)
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
#Submission 1



#RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier



clf = RandomForestClassifier(n_estimators = 3000, random_state = 15).fit(X_train,y_train)

y_pred = clf.predict(X_val)
from sklearn.metrics import accuracy_score

acc = accuracy_score(y_val,y_pred)



print(acc)
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

X_train = scaler.fit_transform(X_train)

X_val = scaler.fit_transform(X_val)
# Submission 2



from sklearn.ensemble import VotingClassifier

from sklearn.ensemble import BaggingClassifier

from xgboost import XGBClassifier



estimators = [('rf', RandomForestClassifier()), ('bag', BaggingClassifier()), ('xgb', XGBClassifier())]



hard_voter = VotingClassifier(estimators=estimators, voting='hard').fit(X_train,y_train)

y_pred = hard_voter.predict(X_val)

y_pred = y_pred.astype(int)

acc = accuracy_score(y_val,y_pred)



print(acc)
df_test = pd.read_csv('/kaggle/input/eval-lab-2-f464/test.csv')

df_test.head()
cols_to_drop = ['id','chem_2','chem_3','chem_5','chem_7']

id_test = df_test['id']

X_test = df_test.drop(cols_to_drop, axis=1)
X = scaler.fit_transform(X)

X_test = scaler.fit_transform(X_test)
from sklearn.ensemble import VotingClassifier

from sklearn.ensemble import BaggingClassifier

from xgboost import XGBClassifier



estimators = [('rf', RandomForestClassifier()), ('bag', BaggingClassifier()), ('xgb', XGBClassifier())]



hard_voter = VotingClassifier(estimators=estimators, voting='hard').fit(X,y)

y_test = hard_voter.predict(X_test)

y_test = y_test.astype(int)
prediction = pd.DataFrame(id_test)

prediction['class'] = y_test

prediction.head()
prediction.to_csv('pred1.csv', index=False)