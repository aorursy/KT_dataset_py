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
%reload_ext autoreload

%autoreload 2

%matplotlib inline



# library for deep learning api interface

from fastai import *

from fastai.tabular import *



# library for visualization

import seaborn as sns

sns.set()
df_train = pd.read_csv('../input/titanic/train.csv')

df_test = pd.read_csv('../input/titanic/test.csv')
df_train.info()
df_train.head()
df_test.info()
df_test.head()
df_train.isnull().sum()
df_test.isnull().sum()
plt.subplots(figsize=(20,15))

sns.heatmap(df_train.corr(), cmap='coolwarm', annot=True)
sns.pairplot(data=df_train, hue='Survived', diag_kind='kde')

plt.show()
sns.catplot(x='Pclass', y='Embarked', hue='Survived', col='Sex', data=df_train, kind='violin')
sns.catplot(x='Parch', y='SibSp', hue='Survived', col='Sex', data=df_train, kind='bar')
sns.boxplot(data=df_train, x='Survived', y='Age')
df_train['Parch'].value_counts()
df_train['SibSp'].value_counts()
# df_train['Surname'] = df_train.Name.str.split(',').apply(lambda x:x[0])

# df_train['Surname'].isnull().any()
df_train.Cabin
for df in (df_train, df_test):

    df['Deck'] = df.Cabin.str[0]

    df.loc[df.Cabin.isnull(), 'Deck'] = 'N'

    df['HasCabin'] = 1

    df.loc[df.Cabin.isnull(), 'HasCabin'] = 0

    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    df['IsAlone'] = 0

    df.loc[df.FamilySize==1, 'IsAlone'] = 1

    df['IsChild'] = 0

    df.loc[df.Age<=12, 'IsChild'] = 1

    df['IsTeen'] = 0

    df.loc[(df.Age>12) & (df.Age<=19), 'IsTeen'] = 1

    df['IsAdult'] = 0

    df.loc[(df.Age>19) & (df.Age<=50), 'IsAdult'] = 1

    df['IsOld'] = 0

    df.loc[df.Age>50, 'IsOld'] = 1

    
df_train.head()
df_train['Embarked'].value_counts()
df_test['Embarked'].value_counts()
df_train['Age'].fillna(value=df_train['Age'].mean(), inplace=True)

df_train['Embarked'].fillna(value='S', inplace=True)

df_train.fillna(value={'Deck':'C'}, inplace=True)

df_train.isnull().sum()
df_test.fillna({'Fare': df_test['Fare'].mean()}, inplace=True)

df_test.fillna({'Age': df_test['Age'].mean()}, inplace=True)

df_test.fillna({'Embarked':'S'}, inplace=True)

df_test.fillna({'Deck':'C'}, inplace=True)

df_test.isnull().sum()
# df_train["Survived"][df_train["Survived"]==1] = "Survived"

# df_train["Survived"][df_train["Survived"]==0] = "Died"
sns.barplot(x='Sex', y='Survived', data=df_train)
sns.barplot(x='Sex', y='Age', hue='Survived', data=df_train)
sns.barplot(x='Embarked', y='Pclass', hue='Survived', data=df_train)
sns.barplot(x='Sex', y='HasCabin', hue='Survived', data=df_train)
sns.violinplot(y='Pclass', x='IsAlone', hue='Survived', data=df_train)
cat_names = ['Sex', 'Pclass', 'IsAlone', 'FamilySize', 'HasCabin', 'Deck', 'SibSp', 'Parch', 'Embarked', 'IsChild', 'IsTeen', 'IsAdult', 'IsOld']

cont_names = ['Age', 'Fare']

dep_var = 'Survived'

procs = [Categorify, Normalize]

# 
np.random.seed(2)
test_db = TabularList.from_df(df_test, cat_names=cat_names, cont_names=cont_names, procs=procs)
train_db = (TabularList.from_df(df_train, cat_names=cat_names, cont_names=cont_names, procs=procs)

            .split_by_idx(list(range(200)))

            .label_from_df(cols=dep_var)

            .add_test(test_db, label=0)

            .databunch())
train_db.show_batch(10)
learn = tabular_learner(train_db, layers=[200], metrics=accuracy, emb_drop=0.001)
learn.lr_find()

learn.recorder.plot()
learn.fit(5, 1e-2)
learn.recorder.plot_losses()

learn.recorder.plot_metrics()

learn.recorder.plot_lr()
# learn.save('Titanic')
predictions, _ = learn.get_preds(DatasetType.Test)

pred=np.argmax(predictions, 1)

pred.shape
pd.read_csv('../input/titanic/gender_submission.csv')
submission_df = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': pred})
submission_df.to_csv('submission.csv', index=False)
from IPython.display import HTML

import base64



def create_download_link( df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv()

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = f'<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    return HTML(html)



create_download_link(submission_df)