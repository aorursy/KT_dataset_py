# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

train.tail()
features_name = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']

train_data = train[features_name]

train_data.head()

sex_dict = {'male': 0, 'female': 1}

embark_dict = {'C': 3, 'Q': 1, 'S': 2}

train_data['Sex'] = train_data['Sex'].apply(lambda x: embark_dict[x] if type(x)==str else 0)

train_data['Embarked'] = train_data['Embarked'].apply(lambda x: embark_dict[x])

train_data['Cabin'] = train_data['Cabin'].apply(lambda x: 0 if type(x)==str else 1)

train_data.head()

#a = train_data.loc[:, "Cabin"]

#a = train_data.loc[0, "Cabin"]
