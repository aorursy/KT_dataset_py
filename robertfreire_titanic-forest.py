#data analysis libraries 

import numpy as np

import pandas as pd



#visualization libraries

import matplotlib.pyplot as plt

import seaborn as sns



#import train and test CSV files

train = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")



from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn import svm

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.metrics import accuracy_score



SURVIVED = 'Survived';

PASSENGER_ID = 'PassengerId';

SEX = "Sex";

PCLASS = "Pclass";

EMBARKED = "Embarked";

SIBSP = "SibSp";

PARCH = "Parch";

AGE = "Age";

TITLE = "Title";

NAME = 'Name';

AGEBIN = 'AgeBin';

AGEBIN_CODE = 'AgeBin_Code';

FARE = 'Fare';

CABIN = 'Cabin';

TICKET = 'Ticket';

IS_ALONE = 'isAlone';

FAREBIN = 'FareBin';

FAREBIN_CODE = 'FareBin_Code'



# Obtain title

train[TITLE] = train[NAME].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

test[TITLE] = train[NAME].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]



#Calculate Age

AverageAgeByTitle = train[['Title',AGE]].groupby('Title', as_index=False).mean();



AverageAgeByTitle.reset_index(inplace=True);

df = train[[PASSENGER_ID, TITLE, AGE]].merge(AverageAgeByTitle, on=TITLE, how='left');

train[AGE][train[AGE].isnull()]=df['Age_y']

df = test[[PASSENGER_ID, TITLE, AGE]].merge(AverageAgeByTitle, on=TITLE, how='left');

test[AGE][test[AGE].isnull()]=df['Age_y']



#Set Default valuies for Fare

train[FARE].fillna(train[FARE].median(), inplace = True)

test[FARE].fillna(test[FARE].median(), inplace = True)



#Set Titles to use

title_names = (train[TITLE].value_counts() < 10) #this will create a true false series with title name as index



train[TITLE] = train[TITLE].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)

test[TITLE] = test[TITLE].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)



#Drop columns

train.drop([NAME], axis=1, inplace=True);

test.drop([NAME], axis=1, inplace=True);



# Fill NA

train[EMBARKED].fillna(train[EMBARKED].mode()[0], inplace = True)

test[EMBARKED].fillna(test[EMBARKED].mode()[0], inplace = True)



# One Hot Encoder

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

categories = [SEX, EMBARKED, TITLE]



OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(train[categories]))

OH_cols_test = pd.DataFrame(OH_encoder.transform(test[categories]))



# Remove categorical columns (will replace with one-hot encoding)

train.drop(categories, axis=1, inplace=True);

test.drop(categories, axis=1, inplace=True);



train = pd.concat([train, OH_cols_train], axis=1)

test = pd.concat([test, OH_cols_test], axis=1)



#Age

train[AGEBIN] = pd.cut(train[AGE].astype(int), 5)

test[AGEBIN] = pd.cut(test[AGE].astype(int), 5)



train[AGEBIN_CODE] = LabelEncoder().fit_transform(train[AGEBIN])

test[AGEBIN_CODE] = LabelEncoder().fit_transform(test[AGEBIN])



#Fare

#train[FAREBIN] = pd.qcut(train[FARE], 4)

#test[FAREBIN] = pd.qcut(test[FARE], 4)

#train[FAREBIN_CODE] = LabelEncoder().fit_transform(train[FAREBIN])

#test[FAREBIN_CODE] = LabelEncoder().fit_transform(test[FAREBIN])



# Add Family size

FAMILY_SIZE = 'FamilySize'

train[FAMILY_SIZE] = train[SIBSP] + train[PARCH]

test[FAMILY_SIZE] = train[SIBSP] + train[PARCH]



train[IS_ALONE] = (train[FAMILY_SIZE] == 0)

test[IS_ALONE] = (test[FAMILY_SIZE] == 0)



train.drop([TICKET, CABIN, AGEBIN, FARE], axis=1, inplace=True);

test.drop([TICKET, CABIN, AGEBIN, FARE], axis=1, inplace=True);



# Train model

predictors = train.drop([SURVIVED, PASSENGER_ID], axis=1);

real_predictors = test.drop([PASSENGER_ID], axis=1);



target = train[SURVIVED]

x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.2, random_state = 0)



randomforest = RandomForestClassifier(n_estimators=60, max_depth=4, random_state=1)



randomforest.fit(x_train, y_train)

y_pred = randomforest.predict(x_val)

acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_randomforest)



clf = svm.SVC(random_state=2, C=2)



clf.fit(x_train, y_train)

y_pred = clf.predict(x_val)

acc= round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc)
#real_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

real_model = svm.SVC(random_state=2, C=2)

real_model.fit(predictors, target)

predictions = real_model.predict(real_predictors)#



#predictions = model.predict(X_real) #

output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)