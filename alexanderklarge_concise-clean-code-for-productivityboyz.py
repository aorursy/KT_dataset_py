### IMPORTS ###

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.model_selection import cross_val_score





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

df_test_raw = pd.read_csv('/kaggle/input/titanic/test.csv')

df_train_raw = pd.read_csv('/kaggle/input/titanic/train.csv')
df_train_raw.head(1)
df_train_raw.info()
df_train_raw.describe(include='all') # by default .describe() ignores categorical columns, but using include='all' includes them, just so you have a good view of the whoe dataset
df_train_raw.dtypes
df_test_raw.info()
df_train_clean = df_train_raw.copy() # apparently you should use the copy function as a safety measure

df_test_clean = df_test_raw.copy()



### USEFUL CODE BELOW! ###

data_cleaner = [df_train_clean,df_test_clean] # this is very snazzy, lets you reference both at the same time
for dataset in data_cleaner:

    dataset.drop(['Cabin'],axis=1,inplace=True)

    dataset.drop(['Ticket'],axis=1,inplace=True)

    dataset.drop(['Name'],axis=1,inplace=True)
for dataset in data_cleaner:

    dataset['Age'].fillna(dataset['Age'].median(),inplace=True)

    dataset['Fare'].fillna(dataset['Fare'].median(),inplace=True)
df_train_clean.head()
df_train_clean = pd.get_dummies(df_train_clean) 

df_test_clean = pd.get_dummies(df_test_clean)
df_train_clean.head()
features = ['Pclass','Age','SibSp','Parch','Fare','Sex_female','Sex_male','Embarked_C','Embarked_Q','Embarked_S']

target = ['Survived']



X_train = df_train_clean[features]

y_train = df_train_clean['Survived']



X_test = df_test_clean[features]

# there is no y in test, as that's the point, that's what we're trying to predict
model = RandomForestClassifier(n_estimators=100,random_state=0)
scores = cross_val_score(model, X_train, y_train,

                              cv=5,

                              scoring='neg_mean_absolute_error')



print("MAE scores:\n", scores)
scores = cross_val_score(model, X_train, y_train,

                              cv=5,

                              scoring='f1') # the F1 score was covered by Andrew Ng, and combines recall and specificity



print("MAE scores:\n", scores)
model.fit(X_train,y_train)
model.predict(X_test)
predictions = model.predict(X_test)
output = pd.DataFrame({'PassengerId': df_test_clean.PassengerId, 'Survived': predictions}) # copied this from https://www.kaggle.com/alexisbcook/titanic-tutorial - just gives you the correcto output

output.to_csv('my_submission.csv', index=False)

print('Oh my, you\'re a Machine Learning God!')