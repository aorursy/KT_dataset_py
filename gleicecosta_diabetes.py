# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import scikitplot as skplt



from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv ('../input/diabetes.csv')
df.shape
df.info()
df.head(10).T
df.tail().T
df.describe().T
df.iloc[:,:-1].hist(bins=20, figsize=(20,10), grid=False, edgecolor='black', alpha=0.5, color='green')
def correlacao(df, size=10):

    corr= df.corr()

    fig, ax = plt.subplots(figsize=(12,10))

    ax.matshow(corr)

    plt.xticks(range(len(corr.columns)),corr.columns)

    plt.yticks(range(len(corr.columns)),corr.columns)

correlacao(df)
df.corr()
# Verificando a quantidade de linhas que foram preenchidas com 0

print('# Quantidade de linhas preenchidas com 0 na coluna Pregnancies: {0}'.format (len(df.loc[df['Pregnancies']==0])))

print('# Quantidade de linhas preenchidas com 0 na coluna Glucose: {0}'.format (len(df.loc[df['Glucose']==0])))

print('# Quantidade de linhas preenchidas com 0 na coluna BloodPressure: {0}'.format (len(df.loc[df['BloodPressure']==0])))

print('# Quantidade de linhas preenchidas com 0 na coluna SkinThickness: {0}'.format (len(df.loc[df['SkinThickness']==0])))

print('# Quantidade de linhas preenchidas com 0 na coluna Insulin: {0}'.format (len(df.loc[df['Insulin']==0])))

print('# Quantidade de linhas preenchidas com 0 na coluna BMI: {0}'.format (len(df.loc[df['BMI']==0])))

print('# Quantidade de linhas preenchidas com 0 na coluna DiabetesPedigreeFunction: {0}'.format (len(df.loc[df['DiabetesPedigreeFunction']==0])))

print('# Quantidade de linhas preenchidas com 0 na coluna Age: {0}'.format (len(df.loc[df['Age']==0])))
#df = df(deep = True)

df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df[['Glucose',

    'BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)



## Mostrando a contagem de Nans

print(df.isnull().sum())
df['Glucose'].fillna(df['Glucose'].mean(), inplace = True)

df['BloodPressure'].fillna(df['BloodPressure'].mean(), inplace = True)

df['SkinThickness'].fillna(df['SkinThickness'].median(), inplace = True)

df['Insulin'].fillna(df['Insulin'].median(), inplace = True)

df['BMI'].fillna(df['BMI'].median(), inplace = True)
num_verd = len(df.loc[df['Outcome']==True])

num_fals = len(df.loc[df['Outcome']==False])

num_verd, num_fals
# Dividindo o DataFrame

train, test = train_test_split(df, test_size=0.20, random_state=42, stratify=df['Outcome'])

train, valid = train_test_split(train, test_size=0.20, random_state=42)
train.shape, valid.shape, test.shape
feats = [c for c in df.columns if c not in ['Outcome']]
df.head()
from sklearn.preprocessing import Imputer
# Regressão Logística

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C = 0.7, random_state=42)

lr.fit(train[feats], train['Outcome'])
preds_lr = lr.predict(valid[feats])

accuracy_score(valid['Outcome'],preds_lr)
accuracy_score(test['Outcome'],lr.predict(test[feats]))
# Modelo Random Forest

rf = RandomForestClassifier(n_estimators=200, min_samples_split=5, max_depth=4, random_state=42)

rf.fit(train[feats], train['Outcome'])

preds = rf.predict(valid[feats])
accuracy_score(valid['Outcome'],preds)
accuracy_score(test['Outcome'],rf.predict(test[feats]))
accuracy_train = []

accuracy_test = []

for x in range(10):

    if x != 0:

        rf = RandomForestClassifier(n_estimators=200, min_samples_split=5, max_depth=x, random_state=42)

        rf.fit(train[feats], train['Outcome'])

        accuracy_train.append(accuracy_score(valid['Outcome'],rf.predict(valid[feats])))

accuracy_train
pd.Series(accuracy_train).plot.line()
# Modelo GBM - GradientBoostingClassifier

accuracy_trainG = []

for x in range(10):

    if x != 0:

        gbm = GradientBoostingClassifier(n_estimators=200, learning_rate=1.0, max_depth=x, random_state=42)

        gbm.fit(train[feats], train['Outcome'])

        accuracy_trainG.append(accuracy_score(valid['Outcome'], gbm.predict(valid[feats])))

pd.Series(accuracy_trainG).plot.line()
preds_gbm = gbm.predict(valid[feats])

accuracy_score(valid['Outcome'],preds_gbm)
accuracy_score(test['Outcome'],gbm.predict(test[feats]))
# Modelo XGBoost

accuracy_trainX = []

for x in range(10):

    if x != 0:

        xgb = XGBClassifier(n_estimators=200, learning_rate=x/100, random_state=42)

        xgb.fit(train[feats], train['Outcome'])

        accuracy_trainX.append(accuracy_score(valid['Outcome'], xgb.predict(valid[feats])))

pd.Series(accuracy_trainX).plot.line()
preds_xgb = xgb.predict(valid[feats])

accuracy_score(valid['Outcome'],preds_xgb)
accuracy_score(test['Outcome'],xgb.predict(test[feats]))
accuracy_train = []

accuracy_trainG = []

accuracy_trainX = []

for y in range(4):

    if y != 0:

        for x in range (10):

            if x != 0:

                rf = RandomForestClassifier(n_estimators=y*100, min_samples_split=5, max_depth=x, random_state=42)

                rf.fit(train[feats], train['Outcome'])

                accuracy_train.append(accuracy_score(valid['Outcome'],rf.predict(valid[feats])))

                gbm = GradientBoostingClassifier(n_estimators=y*100, learning_rate=1.0, max_depth=x, random_state=42)

                gbm.fit(train[feats], train['Outcome'])

                accuracy_trainG.append(accuracy_score(valid['Outcome'], gbm.predict(valid[feats])))

                xgb = XGBClassifier(n_estimators=y*100, learning_rate=x/100, random_state=42)

                xgb.fit(train[feats], train['Outcome'])

                accuracy_trainX.append(accuracy_score(valid['Outcome'], xgb.predict(valid[feats])))



pd.Series(accuracy_train).plot.line()

plt.grid(True)

plt.show()
pd.Series(accuracy_trainG).plot.line()

plt.grid(True)

plt.show()
pd.Series(accuracy_trainX).plot.line()

plt.grid(True)

plt.show()
# Feature Importance com RF

x=4

y=2

rf = RandomForestClassifier(n_estimators=y*100, min_samples_split=5, max_depth=x, random_state=42)

rf.fit(train[feats], train['Outcome'])

pd.Series(rf.feature_importances_, index=feats).sort_values().plot.barh()
# Feature Importance com GBM

x=4

y=2

gbm = GradientBoostingClassifier(n_estimators=y*100, learning_rate=1.0, max_depth=x, random_state=42)

gbm.fit(train[feats], train['Outcome'])

pd.Series(gbm.feature_importances_, index=feats).sort_values().plot.barh()
# Feature Importance com XGB

x=3

y=3

xgb = XGBClassifier(n_estimators=y*100, learning_rate=x/100, random_state=42)

xgb.fit(train[feats], train['Outcome'])

pd.Series(xgb.feature_importances_, index=feats).sort_values().plot.barh()
skplt.metrics.plot_confusion_matrix(valid['Outcome'], preds)