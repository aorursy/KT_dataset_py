# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn import preprocessing



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
titanic_train_df = pd.read_csv("../input/train.csv", index_col = 0, usecols=["PassengerId", "Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"])

titanic_train_df.head()
label_encoder = preprocessing.LabelEncoder()

titanic_train_df['Sex'] = label_encoder.fit_transform(titanic_train_df['Sex'])

titanic_train_df.dropna(inplace=True)

titanic_train_df.head()
X = titanic_train_df.loc[:, titanic_train_df.columns != 'Survived'].as_matrix()

y = titanic_train_df['Survived'].values

len(X)