# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingClassifier





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/hmeq-data/hmeq.csv')

df2 = df.copy()
display(df.sample(10).T)

display(df.shape)
df.info()
df.describe().T
def missing_values(data_frame):

    total = data_frame.isnull().count()

    missing = data_frame.isnull().sum()

    missing_percent = missing/total * 100

    display(missing_percent)

    

missing_values(df)



import matplotlib.pyplot as plt

%matplotlib inline

df.hist(figsize=(20,10))
numeric_feats = [c for c in df.columns if df[c].dtype != 'object' and c not in ['BAD']]

df_numeric_feats = df[numeric_feats]
df_numeric_feats.hist(figsize=(20,8), bins=30)

plt.tight_layout() 
jobs = df['JOB'].dropna().unique()

plt.figure(figsize=(14,15))

c=1

for i in jobs:

    plt.subplot(7,1,c)

    plt.title(i)

    df[df['JOB'] == i]['VALUE'].hist(bins=20)

    c+=1

plt.tight_layout() 
df.groupby('JOB')['YOJ'].mean()
value_mean_by_job = df.groupby('JOB')['VALUE'].mean()

value_mean_by_job
qtd1 = df.groupby(['BAD'])['JOB'].value_counts()



qtd1
plt.figure(figsize=(12,6))

plt.subplot(1,2,1)

fig=sns.boxplot(x='LOAN', data=df, orient='v')

fig.set_title('BoxPlot de Empréstimo (LOAN)')

fig.set_ylabel('Valores de Empréstimos')
import sklearn



from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

import scikitplot as skplt

from sklearn.model_selection import cross_val_score
df_rf = pd.get_dummies(df, df.dtypes[(df.dtypes==np.object) | (df.dtypes=='category')].index.values, drop_first=True)

df_rf.head()
train, test = train_test_split(df_rf, test_size=0.20, random_state=42)
train, valid = train_test_split(train, test_size=0.20, random_state=42)



train.shape, valid.shape, test.shape
df_rf.info()
rf = RandomForestClassifier(n_jobs=-1, oob_score=True, n_estimators = 150, random_state=150)
feats = [c for c in df_rf.columns if c not in ['BAD']]

feats
df_rf['BAD'] = pd.to_numeric(df['BAD'])
missing_values(df)
rf.fit(train[feats],train['BAD'])
preds = rf.predict(valid[feats])
accuracy_score(valid['BAD'], preds_val)
preds_test = rf.predict(test[feats])



preds_test
skplt.metrics.plot_confusion_matrix(test['BAD'], preds_test)