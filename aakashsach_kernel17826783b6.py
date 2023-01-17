# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns

from sklearn import preprocessing
df=pd.read_csv('/kaggle/input/tdac-wine/Test_Data.csv')
df.head()
df.type.unique()
df.drop("index",axis=1,inplace=True)
df.describe()
df.info()
df.isna().sum()
sns.countplot(df['type'])
plt.figure(figsize=(14,14))

ax=sns.heatmap(df.corr(),annot=True)

bottom, top = ax.get_ylim()

ax.set_ylim(bottom + 0.5, top - 0.5)
target=df['type']

df.drop('type',axis=1,inplace=True)
min_max_scaler=preprocessing.MinMaxScaler()

scaled_vals=min_max_scaler.fit_transform(df.values)

preprocessed_df=pd.DataFrame(scaled_vals,columns=df.columns)

preprocessed_df.head()
from sklearn.manifold import TSNE

data=TSNE(n_components=2, early_exaggeration=2.0).fit_transform(preprocessed_df)

data=pd.DataFrame(data)

data.head()
type0=data[target==0]

type1=data[target==1]

fig,ax=plt.subplots(1,1,figsize=(12, 12))

type0.plot.scatter(0,1, color='red', ax=ax, label='Type 0')

type1.plot.scatter(0,1, color='blue', ax=ax, label='Type 1')
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

parameters_logit= [{'C':[0.01,0.1,0.2,0.5],'solver':['liblinear'],'penalty':['l1','l2'],'max_iter':[1000]}]

grid_search_model=GridSearchCV(estimator=LogisticRegression(), param_grid=parameters_logit,scoring='accuracy',cv=10)
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(preprocessed_df,target)

print(X_train.shape)

print(X_test.shape)
grid_search_model.fit(X_train, Y_train)
train_prediction=grid_search_model.predict(X_train)

from sklearn.metrics import classification_report

print(classification_report(Y_train, train_prediction))
test_prediction=grid_search_model.predict(X_test)

print(classification_report(Y_test, test_prediction))
val_df=pd.read_csv('/kaggle/input/tdac-wine/Val_Data.csv')
val_scaled_vals=min_max_scaler.fit_transform(val_df.values)

val_preprocessed_df=pd.DataFrame(val_scaled_vals,columns=val_df.columns)

val_preprocessed_df.head()
submission=pd.DataFrame(val_df['Index'])

val_preprocessed_df.drop('Index',axis=1,inplace=True)

final_prediction=grid_search_model.predict(val_preprocessed_df.values)

submission['type']=final_prediction

submission.rename(columns={"Index":"ID"},inplace=True)
submission.head()
sns.countplot(submission['type'])
submission[submission['type']==0].count()
submission.to_csv('Submission.csv',index=False)