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

print(os.listdir("../input"))
train_df = pd.read_csv('../input/heart-attack-prediction/data.csv')





train_df.head()



train_df.isna().sum()




train_df=train_df.drop(

['slope',

  'ca',

  'thal',

  'chol',

],axis=1)

train_df = train_df[train_df.fbs != "?"]

train_df = train_df[train_df.restecg != "?"]

train_df = train_df[train_df.thalach != "?"]

train_df = train_df[train_df.exang != "?"]

train_df = train_df[train_df.oldpeak != "?"]
train_df.head()




classe = train_df['num       ']

atributos = train_df.drop('num       ', axis=1)

atributos.head()
from sklearn.model_selection import train_test_split

atributos_train, atributos_test, class_train, class_test = train_test_split(atributos, classe, test_size = 0.25 )



atributos_train.describe()




from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=3, random_state =0)

model = dtree.fit(atributos_train, class_train)




from sklearn.metrics import accuracy_score

classe_pred = model.predict(atributos_test)

acc = accuracy_score(class_test, classe_pred)

print("My Decision Tree acc is {}".format(acc))


