import os

print((os.listdir('../input/')))
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np
df_train = pd.read_csv('../input/webclubrecruitment2019/TRAIN_DATA.csv')

df_test = pd.read_csv('../input/webclubrecruitment2019/TEST_DATA.csv')
df_train.describe()
test_index=df_test['Unnamed: 0']
train_X = df_train.loc[:, 'V1':'V16']

train_y = df_train.loc[:, 'Class']
df_train["Class"].value_counts()
colors = ["green", "red"]

labels ="Good", "Bad"



plt.suptitle('Loan Status', fontsize=20)



df_train["Class"].value_counts().plot.pie(explode=[0,0.1], autopct='%1.2f%%', shadow=True, colors=colors, 

                                          labels=labels, fontsize=12, startangle=50)
df_test = df_test.loc[:, 'V1':'V16']
df = pd.concat([train_X,df_test])
df.info()
import matplotlib.pyplot as plt

import seaborn as sns
plt.figure(figsize = (16, 10))

sns.heatmap(df.corr(), annot = True, cmap="YlGnBu")

plt.show()
df = pd.get_dummies(df,prefix='Category_',columns = ["V2","V3","V4","V5","V7","V8","V9","V11","V16"],drop_first = True)
df.shape
sns.distplot(df['V1'])

plt.show()
#from sklearn.preprocessing import StandardScaler

#sc = StandardScaler()

#df["V1"] = pd.DataFrame(sc.fit_transform(pd.DataFrame(df["V1"])))
#np.log(df["V1"])

#sns.distplot(np.log(df["V1"]))

#plt.show()
X = df[0:30000]

X_final = df[30000:45210]
pd.crosstab(X["V14"],train_y)
a = np.array(df['V14'].values.tolist())

df['V14'] = np.where(a > -1, 1, a).tolist()
df["V14"].describe()
pd.crosstab(X["V15"],train_y)
a = np.array(df['V15'].values.tolist())

df['V15'] = np.where(a > 0, 1, a).tolist()
df["V15"].describe()
df["V12"].describe()
pd.set_option("max_columns",200)
pd.crosstab(X["V12"],train_y)
sns.distplot(df['V12'])

plt.show()
(df["V12"]>1000).sum()
pd.crosstab(X["V10"],train_y)
pd.crosstab(X["V6"],train_y)
pd.crosstab(X["V13"],train_y)
X = df[0:30000]
X_final = df[30000:45210]
from sklearn import metrics

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV
# param_grid = {

#  'max_depth': [4,8,10],

#  'min_samples_leaf': range(100, 400, 200),

#  'min_samples_split': range(200, 500, 200),

#  'n_estimators': range(10,100, 10),

#  'max_features': [5, 10,15,20]

# }

# rf = RandomForestClassifier()

# grid_search = GridSearchCV(estimator = rf, param_grid = param_grid,

#  cv = 3, n_jobs = -1,verbose = 1)

# grid_search.fit(X,train_y)

# print('We can get accuracyof',grid_search.best_score_,'using',grid_search.best_params_)

rf = RandomForestClassifier(n_estimators=40,max_depth =5,min_samples_split = 200)

rf.fit(X,train_y)
pred  = rf.predict_proba(X_final)
result=pd.DataFrame()

result['Id'] = test_index

result['PredictedValue'] = pd.DataFrame(pred[:,1])

result.head()
result.to_csv('output.csv', index=False)