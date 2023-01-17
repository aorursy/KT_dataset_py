import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
# Read in the 2 datasets

training_data = '../input/adult-training.csv'

test_data = '../input/adult-test.csv'



columns = ['Age','Workclass','fnlgwt','Education','Education Num','Marital Status',

           'Occupation','Relationship','Race','Sex','Capital Gain','Capital Loss',

           'Hours/Week','Country','Above/Below 50K']



df_train_set = pd.read_csv(training_data, names=columns)

df_test_set = pd.read_csv(test_data, names=columns, skiprows=1)

train_set = pd.DataFrame(df_train_set)

test_set = pd.DataFrame(df_test_set)
train_set.head()
# Just cleaning up some whitespace

for col in train_set.columns:

    if type(train_set[col][0]) == str:

        print("Working on " + col)

        train_set[col] = train_set[col].apply(lambda val: val.replace(" ",""))
train_set['Workclass'].replace(['?'],'Unknown',inplace=True)

train_set['Occupation'].replace(['?'],'Unknown',inplace=True)
train_set.head()
replace = ['Workclass', 'Marital Status', 'Occupation', 'Relationship', 'Race', 'Sex']



for col in replace:

    train_set = pd.concat([train_set, pd.get_dummies(train_set[col],prefix=col,prefix_sep=':')], axis=1)

    train_set.drop(col,axis=1,inplace=True)



# Do this one last so it's at the far right of the DataFrame

train_set = pd.concat([train_set, pd.get_dummies(train_set['Above/Below 50K'],drop_first=True)], axis=1)

train_set.drop('Above/Below 50K', axis=1,inplace=True)



train_set.drop('Country',axis=1,inplace=True)

train_set.drop('Education', axis=1,inplace=True)



train_set.head()
test_set.head()
# Just cleaning up some whitespace

for col in test_set.columns:

    if type(test_set[col][0]) == str:

        print("Working on " + col)

        test_set[col] = test_set[col].apply(lambda val: val.replace(" ",""))
test_set['Workclass'].replace(['?'],'Unknown',inplace=True)

test_set['Occupation'].replace(['?'],'Unknown',inplace=True)
replace = ['Workclass', 'Marital Status', 'Occupation', 'Relationship', 'Race', 'Sex']



for col in replace:

    print (col)

    test_set = pd.concat([test_set, pd.get_dummies(test_set[col],prefix=col,prefix_sep=':')], axis=1)

    test_set.drop(col,axis=1,inplace=True)



test_set.drop('Above/Below 50K', axis=1,inplace=True)

test_set.drop('Country',axis=1,inplace=True)

test_set.drop('Education', axis=1,inplace=True)



test_set.head()
# Replace the salary column with a single binary representation

# This will be used for Y test

df_test_set = pd.concat([df_test_set, pd.get_dummies(df_test_set['Above/Below 50K'],drop_first=True)], axis=1)

df_test_set.drop('Above/Below 50K',axis=1,inplace=True)

df_test_set.head()
df_test_set.rename(columns = {'>50K.':'>50K'},inplace=True)
df_test_set.head()
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, classification_report
X_train = train_set.drop('>50K',axis=1)

y_train = train_set['>50K']

X_test = pd.DataFrame(test_set)

y_test = df_test_set['>50K']
model = LogisticRegression()
model.fit(X_train,y_train)
predictions = model.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
scaled_model = LogisticRegression()
scaled_model.fit(X_train,y_train)
scaled_predictions = scaled_model.predict(X_test)
# With Scaling X_train and X_test

print(confusion_matrix(y_test,scaled_predictions))
# With Scaling X_train and X_test

print(classification_report(y_test,scaled_predictions))