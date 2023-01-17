# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in



#import numpy as np # linear algebra

#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



import numpy as np

import pandas as pd

from sklearn import preprocessing, tree



train_df = pd.read_csv('train.csv')

train_df = train_df.dropna()

def preprocess_data(df):

    processed_df = df.copy()

    le = preprocessing.LabelEncoder()

    processed_df['Sex'] = le.fit_transform(processed_df['Sex'])

    processed_df['Embarked'] = le.fit_transform(processed_df['Embarked'])

    processed_df = processed_df.drop(["Name", "Ticket", "Cabin"], axis=1)

    return processed_df

#processing sex and emabrked column so that it can be fed into ML Algos

train_df = preprocess_data(train_df)

#Creating training sample Feature matrix

X = train_df.drop('Survived', axis=1).values

#print(X)

#creating training sample output matrix

y = train_df['Survived'].values

#create untrained decision tree classifier

clf_dt = tree.DecisionTreeClassifier(max_depth=10)

#Fit training data into the classifier

clf_dt.fit(X, y)

#Prepare input data

test_df = pd.read_csv('test.csv')

processed_test_df = preprocess_data(test_df)

#print(test_df.count())

processed_test_df['Age'] = processed_test_df['Age'].fillna(train_df['Age'].mean())

processed_test_df['Fare'] = processed_test_df['Fare'].fillna(round(processed_test_df['Fare'].mean()))

#print(test_df[test_df['Age'].notnull()].count())

X = processed_test_df.values

#Predict survival for input data

output = clf_dt.predict(X)

test_df.insert(1, 'Survived', output)

output_df = test_df.iloc[:, :2]

output_df.to_csv('output.csv', index=False)
