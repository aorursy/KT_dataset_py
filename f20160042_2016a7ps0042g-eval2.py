import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
train = pd.read_csv('/kaggle/input/eval-lab-2-f464/train.csv')

test = pd.read_csv('/kaggle/input/eval-lab-2-f464/test.csv')
train.head()
train.info()
train['class'].unique()
train.isnull().sum()
train.duplicated().sum()
test.info()
import seaborn as sns

f, ax = plt.subplots(figsize=(10, 8))

corr = train.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True, ax=ax, annot = True);
train=train.drop(['id'],axis=1)

train.info()
test=test.drop(['id'],axis=1)

test.info()
y=train['class']

X=train.drop(['class'],axis=1)

X.head()
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)
from sklearn.ensemble import ExtraTreesClassifier
score_train_RF = []

score_test_RF = []



for i in range(1,18,1):

    rf = ExtraTreesClassifier(n_estimators=i, random_state = 42)

    rf.fit(X_train, y_train)

    sc_train = rf.score(X_train,y_train)

    score_train_RF.append(sc_train)

    sc_test = rf.score(X_val,y_val)

    score_test_RF.append(sc_test)
plt.figure(figsize=(10,6))

train_score,=plt.plot(range(1,18,1),score_train_RF,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='green', markersize=5)

test_score,=plt.plot(range(1,18,1),score_test_RF,color='red',linestyle='dashed',  marker='o',

         markerfacecolor='blue', markersize=5)

plt.legend( [train_score,test_score],["Train Score","Test Score"])

plt.title('Fig4. Score vs. No. of Trees')

plt.xlabel('No. of Trees')

plt.ylabel('Score')
rf = ExtraTreesClassifier(n_estimators=14, random_state = 42)

rf.fit(X_train, y_train)

rf.score(X_val,y_val)
rf.fit(X,y)

y_pred = rf.predict(test)
final_test=pd.read_csv('/kaggle/input/eval-lab-2-f464/test.csv')

X_test_id = final_test['id']

final = pd.concat([X_test_id,pd.DataFrame(y_pred)],axis=1)

final.columns=['id','class']

final.to_csv('sub7.csv',index=False)