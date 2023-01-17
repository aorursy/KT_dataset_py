# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

import matplotlib.pyplot as plt

from IPython.display import display

import csv as csv





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", ".."]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train.csv', header = 0, index_col = 'PassengerId')

df_test = pd.read_csv('../input/test.csv', header = 0, index_col = 'PassengerId')

df = pd.concat([df_train, df_test], keys=["train", "test"])

#df = pd.read_csv("../input/train.csv")

# convert gender to boolean

df["Sex"] = df["Sex"].apply(lambda x: x == "male") # 1 => male

display(df)
# clean_df

df.loc[df['Fare'].isnull(), 'Fare'] = df['Fare'].mode()[0]

clean_df = df.copy()

del clean_df["Name"]

del clean_df["Ticket"]

clean_df["Cabin"].describe()

# for now drop Cabin too

del clean_df["Cabin"]

#display(clean_df)
# deal with nan age

clean_df.isnull().sum()

# mean male age

#display(clean_df["Sex"])

# clean_df[clean_df["Sex"]]["Age"].describe()
male_age = clean_df["Age"].mean()

# female_age = clean_df[clean_df["Sex"]==False]["Age"].mean()

clean_df["Age"].fillna(value=male_age, inplace=True)

# clean_df[clean_df["Sex"]==False]["Age"].fillna(value=female_age, inplace=True)

clean_df = clean_df[clean_df["Embarked"].notnull()]

clean_df.isnull().sum()
df_embark = pd.get_dummies(clean_df["Embarked"])

df_new = clean_df.join(df_embark)

del df_new["Embarked"]

df_Pclass = pd.get_dummies(df_new["Pclass"],prefix='Pclass')

df_new = df_new.join(df_Pclass)

del df_new["Pclass"]

df_new
clean_df.dtypes

df_train = df_new.ix['train']

print(df_train.shape)

df_test  = df_new.ix['test']
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import LabelEncoder, LabelBinarizer

from sklearn_pandas import DataFrameMapper



print( 'Training')

# features = ['Sex',  'SibSp', 'Age','Embarked', 'Pclass', 'Fare']



# pipeline = Pipeline([('featurize', featurize(features)), ('forest', RandomForestClassifier())])
y = df_train['Survived'].as_matrix()

X = df_train.drop('Survived',axis=1).as_matrix()

forest = RandomForestClassifier(n_estimators = 1000)

forest = forest.fit(X,y)

cross_val_score(forest, X, y, cv=5)
print('Predicting')

test_data = df_test.drop('Survived',axis=1).as_matrix()

output = forest.predict(test_data).astype(int)

display(output)
#test_Y = model.predict( test_X )

df_test = pd.read_csv('../input/test.csv', header = 0)

passenger_id = df_test.PassengerId

test = pd.DataFrame( { 'PassengerId': passenger_id , 'Survived': output } )

test.shape

test.head(5)

test.to_csv( 'titanic_pred.csv' , index = False )