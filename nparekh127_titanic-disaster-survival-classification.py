# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.model_selection import KFold, cross_val_score



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/titanic/train.csv')





print("Columns", df.columns.tolist())
df.head()
survived = df[df['Survived'] == 1]

not_survived = df[df['Survived'] == 0]



male_survived = survived[survived['Sex']=='male']

female_survived = survived[survived['Sex']=='female']



male_not_survived = not_survived[not_survived['Sex']=='male']

female_not_survived = not_survived[not_survived['Sex']=='female']



print("Total no of passenger survived: ", survived['PassengerId'].count())

print("Total no of Male passenger survived: ", male_survived['PassengerId'].count())

print("Total no of Female passenger survived: ", female_survived['PassengerId'].count())



print("Total no of passenger not survived: ", not_survived['PassengerId'].count())

print("Total no of Male passenger not survived: ", male_not_survived['PassengerId'].count())

print("Total no of Female passenger not survived: ", female_not_survived['PassengerId'].count())



print("Total no of passenger onboarded: ", df['PassengerId'].count())
# limit to categorical data using df.select_dtypes()

categorical_df = df.select_dtypes(include=[object])

categorical_df.head(3)
# we can make use of pandas's get_dummies() method or scikitlearn's OneHotEncoder for encoding purpose

# Step 1: call get_dummies method of pandas

# Step 2: concatenate dummy columns withh original dataframe

# Step 3: Drop one of the column from each dummy set



# Step 1: get dummies columns from categorical column

encoded_sex_col = pd.get_dummies(df['Sex'])

encoded_embarked_col = pd.get_dummies(df['Embarked'])

#pd.get_dummies(df['Ticket'])

#pd.get_dummies(df['Cabin'])



# Step 2: Concate

encoded_df = pd.concat([df,encoded_sex_col, encoded_embarked_col], axis='columns')

print(encoded_df.head(3))





# Dummy variable Trap and Multicollinearity

# Now this is a problem we have to deal with when we genereate dummy column. 

# This can easily be understood by just googling so I'm not going in that direction and as a solution we 

# just have drop one column each dummy set that means, fr embarked dummy coolumns, we can drop one columns from "C","Q" or "S".



#Step 3: Drop one of the column from each dummy set 

final = encoded_df.drop(['Q', 'male'], axis='columns')

print("################## Dataframe after Step 3 ############")

print(final.head(3))
# Dropping unnecessary features

final = final.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Sex', 'Embarked'], axis='columns')

print(final.head())





final = final.dropna(axis = 0, how ='any')
# Now this seems pretty good go for training our model but before that we have do a 

# bit of work which is "Train-Test Split" which we would keep in 70:30 ratio.

features = [c for c in final.columns.tolist() if c not in ['Survived']]

target = ['Survived']



X_train, X_test, y_train, y_test = train_test_split(final[features], final[target], test_size=0.30, random_state=42)



print(X_train.head())
# Training model

kf = KFold(n_splits=20)

print(kf)



rfc = RandomForestClassifier(max_depth=50, random_state=0, n_jobs=10, n_estimators=150)

scores = []

for train_index, test_index in kf.split(final):

    #print("TRAIN:", train_index, "TEST:", test_index)

    train, test = final.iloc[train_index], final.iloc[test_index]    

    

    rfc.fit(train[features], train[target].values.ravel())

    print("Score:", rfc.score(test[features], test[target]))

    scores.append(rfc.score(test[features], test[target]))

    

# Instead of looping over indexes returned by kf.split() mwthod we can also achieve in simpler way    

# scores = cross_val_score(rfc, final[features], final[target], cv=20)
y_prediction = rfc.predict(X_test)

y_prediction



print("################## Confusion Matrix ##################")

print(confusion_matrix(y_test, y_prediction))
print("################## Classification Report ##################")

print(classification_report(y_test, y_prediction))



print("Score:", rfc.score(X_test, y_test))