# this came with the kernel I forked from

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# IMPORTING THE GOOD STUFF

import pandas as pd

import numpy as np

import random as rnd





import seaborn as sns

import matplotlib.pyplot as plt

import sklearn.model_selection
train_dataframe=pd.read_csv("../input/train.csv")

test_dataframe=pd.read_csv("../input/test.csv")

gender_submission = pd.read_csv("../input/gender_submission.csv")

print("Got them!")
# Visualisation by pivoting - conclude that class, sex are imortant

# hitsograms and box plots - conlude that port, age are important

# Later, I might want to include whether people who shared tickets or cabins were more likely to die/survive together

# and extract titles too
# Name, sex, age, sibsp, parch, fare, cabin (whether there is a cabin), embarked should be modelled

#train_dataframe['Cabin'].head()
train_dataframe['Cabin?'] = np.where(pd.isnull(train_dataframe['Cabin']), 0,1)

#train_dataframe['Cabin?'].head()
train_dataframe[['Cabin?','Survived']].groupby('Cabin?').mean()
predictors = ['Pclass','Sex', 'Age', 'SibSp','Parch', 'Fare', 'Cabin?', 'Embarked']

train_dataframe[predictors].info()
# so what about the missing ages? What predicts age?

sns.countplot(train_dataframe['Age'], hue = train_dataframe['Sex'])

# there were more adult men than women but children were evenly distributed over boys and girls
fig = plt.figure(figsize = (22,5))

ax=fig.add_axes([.1, .1, .8, .8])

sns.countplot(train_dataframe['Age'], hue = train_dataframe['Pclass'], ax=ax)

# children is about the same for classes, but there are more old people in first class
train_dataframe[['Age','Sex','Pclass']].groupby(['Sex','Pclass']).mean()
train_dataframe[['Age','Sex','Pclass']].groupby(['Sex','Pclass']).median()
def impute_age(cols):

    age = cols[0]

    sex = cols[1]

    pclass = cols[2]

    if pd.isnull(age):

        if sex == 'female':

            if pclass == 1:

                return 35

            elif pclass == 2:

                return 28

            elif pclass == 3:

                return 21.5

            else:

                print('error! pclass should be 1, 2, or 3 but it is '+pclass+'!')

                return np.nan

        elif sex == 'male':

            if pclass == 1:

                return 40

            elif pclass == 2:

                return 30

            elif pclass == 3:

                return 25

            else:

                print('error! pclass should be 1, 2, or 3 but it is '+pclass+'!')

                return np.nan

        else: print('error! sex should be female or male but it is '+sex+'!')

    else:

        return age
train_dataframe['Age']=train_dataframe[['Age','Sex','Pclass']].apply(impute_age,axis=1)
# Now for logistic regression need to one-hot some features

# sex and embarked are categorical

male = pd.get_dummies(train_dataframe['Sex'], drop_first=True)

port = pd.get_dummies(train_dataframe['Embarked'], drop_first=True)

train = pd.concat([train_dataframe, male, port],axis=1)

train[pd.isnull(train['Embarked'])==True]

# just change those to 0.33 don't know
train.loc[61,'Q']=0.33

train.loc[61,'S']=0.33

train.loc[829,'Q']=0.33

train.loc[829,'S']=0.33

train[pd.isnull(train['Embarked'])==True]
predictors_num = ['Pclass', 'male', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin?', 'Q','S']

X_tr, X_test, y_tr, y_test = sklearn.model_selection.train_test_split(train[predictors_num],train['Survived'], random_state = 1)
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

logreg.fit(X_tr, y_tr)
predictions = logreg.predict(X_test)
predictions
import sklearn.metrics

sklearn.metrics.confusion_matrix(y_test, predictions)

# so of the 128 people who died, 14 were misclassified

# of the 95 who died, 29 were misclassified
coefficients = pd.DataFrame(logreg.coef_,columns = X_test.columns)

coefficients

# they are a nit misleading because of differences in scale

# PClass should be x about 3, age by about 50, fare by about 100, sibsp and parch by about 2

# then order of importance is probably sex, class, age, sibsp, cabin, port, parch, fare

# that's quite interesting!!
# Now repeat for the whole dataset

logregfinal = LogisticRegression()

logregfinal.fit(train[predictors_num], train['Survived'])

coefficients = pd.DataFrame(logregfinal.coef_,columns = predictors_num)

coefficients

# they are quite different, as it's done with a much bigger dataset this time

coefficients.to_csv("logistic regression coefficients.csv", index=False)
# now I need to apply the same cleansing to the test data as I did to the training data

test_dataframe['Cabin?'] = np.where(pd.isnull(test_dataframe['Cabin']), 0,1)

# should impute age from all available data, not just training dataset medians

age_data = pd.concat([train_dataframe[['Age','Sex','Pclass']], test_dataframe[['Age','Sex','Pclass']]], axis = 0)

age_data.groupby(['Sex','Pclass']).median()

#i've checked the mean too - by some amazing coincidence, this is just the same
test_dataframe['Age']=test_dataframe[['Age','Sex','Pclass']].apply(impute_age,axis=1)
# can't drop first, must drop female and C

sex = pd.get_dummies(test_dataframe['Sex'])

port = pd.get_dummies(test_dataframe['Embarked'])
sex.drop('female', axis = 1, inplace=True)

port.drop('C', axis = 1, inplace = True)
test = pd.concat([test_dataframe, sex, port],axis=1)

test[pd.isnull(test['Embarked'])==True]

# great, there are none missing! So ready to go
# test.info()

# bummer, somebody's fare is missing!

test[['Fare','Pclass']].groupby('Pclass').median()
test[pd.isnull(test['Fare'])==True]

# so let's give him the median fare for 3rd class

test['Fare']=np.where(pd.isnull(test['Fare'])==True, 7.8958, test['Fare'])
final_predictions = logregfinal.predict(test[predictors_num])

final = pd.DataFrame(final_predictions, columns = ['Survived'])
output = pd.concat([test['PassengerId'],final], axis=1)

output
output.to_csv('csv_to_submit.csv', index=False)