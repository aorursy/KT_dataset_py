import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df_train = pd.read_csv("../input/train.csv")
df_train.head()
df_test = pd.read_csv("../input/test.csv")

df_origin_test = df_test.copy()
df_train.head()
plt.figure(figsize = (30,10))

sns.heatmap(df_train.isna(),yticklabels=False,cbar=False,cmap='viridis')
pd.DataFrame(df_train.isna().sum(),columns=["Count"])[pd.DataFrame(df_train.isna().sum(),columns=["Count"])["Count"]>0]
df_train.fillna((df_train.mean()), inplace=True,axis=0)
df_test.fillna((df_test.mean()), inplace=True,axis=0)
plt.figure(figsize = (30,10))

sns.heatmap(df_train.isna(),yticklabels=False,cbar=False,cmap='viridis')
sns.set_style('whitegrid')

sns.countplot(x='parentesco1',data=df_train,palette='RdBu_r')
df_train["parentesco1"].value_counts()
df_categorial_train = df_train.select_dtypes(include=['object']).head()
df_categorial_train.head()
df_train.dependency.replace(to_replace=dict(yes=1, no=0), inplace=True)
df_test.dependency.replace(to_replace=dict(yes=1, no=0), inplace=True)
df_train.edjefe.replace(to_replace=dict(yes=1, no=0), inplace=True)
df_test.edjefe.replace(to_replace=dict(yes=1, no=0), inplace=True)
df_train.edjefa.replace(to_replace=dict(yes=1, no=0), inplace=True)
df_test.edjefa.replace(to_replace=dict(yes=1, no=0), inplace=True)
df_train.pop('Id')

df_train.pop('idhogar')
df_test.pop('Id')

df_test.pop('idhogar')
df_train.dependency.value_counts()
df_train.dependency=df_train.dependency.astype('float64')

df_train.dependency.dtype
df_test.dependency=df_test.dependency.astype('float64')
df_train.edjefe=df_train.edjefe.astype('int')

df_train.edjefa=df_train.edjefa.astype('int')
df_test.edjefe=df_test.edjefe.astype('int')

df_test.edjefa=df_test.edjefa.astype('int')
df_train.select_dtypes(include=['object']).count()
sns.set_style('whitegrid')

sns.countplot(x='Target',data=df_train,palette='RdBu_r')
y_train = df_train.pop("Target")
X_train = df_train
df_test.head()
from sklearn.ensemble import RandomForestClassifier

arr=[2,4,8,16,32,64,128,168,200,220]

train_score_randomfor=[]

test_score_randomfor=[]

for i in arr:

    rfreg = RandomForestClassifier(n_estimators = i)

    rfreg = rfreg.fit(X_train, y_train)

    train_score_randomfor.append(rfreg.score(X_train , y_train))
from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(arr, np.array(train_score_randomfor)*100, 'b', label="Train AUC")

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('AUC score')

plt.xlabel('n_estimators')

plt.show()
rfreg = RandomForestClassifier(n_estimators = 30)

rfreg = rfreg.fit(X_train, y_train)

rfreg.score(X_train , y_train)
predict = rfreg.predict(df_test)
pd.Series(predict).value_counts()
predict
df_submission = pd.DataFrame({"Id":df_origin_test.Id,"Target":predict})
df_submission.to_csv("Submission_Finale.csv",index=False)