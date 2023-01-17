# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
%reload_ext autoreload

%autoreload 2

%matplotlib inline

from fastai import *

from fastai.vision import *

from fastai.tabular import *
train = pd.read_csv('../input/train.csv')

train.head()

train.count()
test = pd.read_csv("../input/test.csv")

test.isnull().sum()
test_id = test['PassengerId'] 



test["Fare"] = test["Fare"].fillna(value =test["Fare"].mean())
train.nunique()
train_title = [i.split(",")[1].split(".")[0].strip() for i in train["Name"]]

train["Title"] = pd.Series(train_title)



test_title = [i.split(",")[1].split(".")[0].strip() for i in test["Name"]]

test["Title"] = pd.Series(test_title)
train["Family_size"] = train["SibSp"] + train["Parch"] + 1

test["Family_size"] = test["SibSp"] + test["Parch"] + 1
dep_var = 'Survived'

#cat_names = data.select_dtypes(exclude=['int', 'float']).columns

# cat_names = ['Sex','SibSp', 'Pclass', 'Title', 'Parch', 'Family_size']

# cont_names = data.select_dtypes([np.number]).columns

# cont_names = ['Age', 'Fare']



# cont_names, cat_names = cont_cat_split(df=train, max_card=20, dep_var='Survived')

cat_names = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Title', 'Family_size'] # 'Embarked','Cabin',

cont_names = ['Age', 'Fare']





print("Categorical columns are : ", cat_names)

print('Continuous numerical columns are :', cont_names)

procs = [FillMissing, Categorify, Normalize]

test = TabularList.from_df(test, cat_names=cat_names, cont_names=cont_names)



data = (TabularList.from_df(train, path='.', cat_names=cat_names, cont_names=cont_names, procs=procs)

                        .random_split_by_pct(valid_pct=0.2, seed=43)

                        .label_from_df(cols = dep_var)

                        .add_test(test, label=0)

                        .databunch())
data.show_batch(rows=10)

learn = tabular_learner(data, layers=[1000, 200, 15], metrics=accuracy, emb_drop=0.1, callback_fns=ShowGraph)

learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(15, max_lr=slice(1e-03))

learn.model
learn.recorder.plot_losses()
predictions, *_ = learn.get_preds(DatasetType.Test)

labels = np.argmax(predictions, 1)
submission = pd.DataFrame({'PassengerId': test_id, 'Survived': labels})

submission.to_csv('submission.csv', index=False)

submission.head()
