# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas import Series,DataFrame



import seaborn as sb

import matplotlib.pyplot as plt

import xgboost as xgb





#Machine Learning

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Working with Train Datas

#Reading files

train_data = pd.read_csv("../input/train.csv",)

test_data = pd.read_csv("../input/test.csv",)

train_data_only_first_class = train_data.ix[~(train_data['Pclass'] != 1)]

train_data_only_first_class_female = train_data_only_first_class.ix[~(train_data_only_first_class['Sex'] != 'female')]

train_data_only_first_class_female_Cherbourg = train_data_only_first_class_female.ix[~(train_data_only_first_class_female['Embarked'] != 'C')]



for age in train_data_only_first_class_female_Cherbourg['Age']:

    if age > 51:

        train_data_only_first_class_female_Cherbourg_age = train_data_only_first_class_female_Cherbourg.ix[~(train_data_only_first_class_female_Cherbourg['Age'] > age)]



for age in train_data_only_first_class_female_Cherbourg_age['Age']:

    for age in range(14,30):

        train_data_only_first_class_female_Cherbourg_age_under_fourteen_not_over_fiftyone = train_data_only_first_class_female_Cherbourg_age.ix[~(train_data_only_first_class_female_Cherbourg_age['Age'] == age)]

        

train_data = train_data_only_first_class_female_Cherbourg_age_under_fourteen_not_over_fiftyone

y_train = train_data.pop("Survived")





        
#Working with Test data

#Reading files

train_data = pd.read_csv("../input/train.csv",)

test_data = pd.read_csv("../input/test.csv",)

test_data_only_first_class = test_data.ix[~(test_data['Pclass'] != 1)]

test_data_only_first_class_female = test_data_only_first_class.ix[~(test_data_only_first_class['Sex'] != 'female')]

test_data_only_first_class_female_Cherbourg = test_data_only_first_class_female.ix[~(test_data_only_first_class_female['Embarked'] != 'C')]



for age in test_data_only_first_class_female_Cherbourg['Age']:

    if age > 51:

        test_data_only_first_class_female_Cherbourg_age = test_data_only_first_class_female_Cherbourg.ix[~(test_data_only_first_class_female_Cherbourg['Age'] > age)]       

        

for age in test_data_only_first_class_female_Cherbourg_age['Age']:

    for age in range(14,30):

        test_data_only_first_class_female_Cherbourg_age_under_fourteen_not_over_fiftyone = test_data_only_first_class_female_Cherbourg_age.ix[~(test_data_only_first_class_female_Cherbourg_age['Age'] == age)]

        

test_data = test_data_only_first_class_female_Cherbourg_age_under_fourteen_not_over_fiftyone
#Plotting Age vs PassengerId

graph = sb.lmplot(x="Age", y="PassengerId",ci=None,data=train_data, col="Survived",

    palette="muted",col_wrap=2,scatter_kws={"s": 100,"alpha":.5},

    line_kws={"lw":4,"alpha":0.5},hue="Survived",x_jitter=1.0,y_jitter=1.0,size=6)



# remove the top and right line in graph

sb.despine()

# Additional line to adjust some appearance issue

plt.subplots_adjust(top=0.9)



# Set the Title of the graph from here

graph.fig.suptitle('Age vs. PassengerId', fontsize=10,color="b",alpha=0.5)



# Set the xlabel of the graph from here

graph.set_xlabels("Age",size = 10,color="b",alpha=0.5)



# Set the ylabel of the graph from here

graph.set_ylabels("PassengerId",size = 10,color="b",alpha=0.5)
# Plotting of Embarkment at different points and Survival

sb.factorplot(x="Embarked", data=train_data, kind="count",

                   palette="BuPu", hue='Survived', size=6, aspect=1.5)
# Plotting of Class at different points and Survival

sb.factorplot(x="Pclass", data=train_data, kind="count",

                   palette="BuPu", hue='Survived', size=6, aspect=1.5)
# Plotting Sex vs Survival

sb.factorplot(x="Sex", data=train_data, kind="count",

                   palette="BuPu", hue='Survived',size=6, aspect=1.5)
# Plotting Age vs Survival

sb.factorplot(x="Age", data=train_data, kind="count",

                   palette="BuPu", hue='Survived',size=15, aspect=5)
#Defining random values for age, fare in train_data and test_data



train_data = train_data_only_first_class_female_Cherbourg_age_under_fourteen_not_over_fiftyone.fillna(0)



def random_age():

    for age in train_data["Age"]:

            sum_age = train_data.Age.sum(axis=0)/train_data.index.size

            age = np.random.randint((sum_age)//1)

    return age

train_data["Age"].fillna(random_age, inplace=True)

test_data.describe()

test_data = test_data.fillna(0)



def random_age_test():

    for age in test_data["Age"]:

            sum_age = test_data.Age.sum(axis=0)/test_data.index.size

            age = np.random.randint((sum_age)//1)

    return age

test_data["Age"].fillna(random_age_test, inplace=True)





def random_fare_test():

    for age in test_data["Fare"]:

            sum_fare = test_data.Fare.sum(axis=0)/test_data.index.size

            fare = np.random.randint((sum_fare)//1)

    return age

test_data["Fare"].fillna(random_fare_test, inplace=True)
#Describe train_data

train_data.describe()
#Describe test_data

test_data.describe()
#Numerical values in train_data

numeric_variables = list(train_data.dtypes[train_data.dtypes != "object"].index)

train_data[numeric_variables].head()
#Numerical values in test_data

numeric_variables_test = list(test_data.dtypes[test_data.dtypes != "object"].index)

test_data[numeric_variables_test].head()
#XgBoost for numerical variables

X_train = train_data[numeric_variables]

#test_data.drop(["PassengerId"],axis=1)

X_test = test_data[numeric_variables]

print(X_test)

y_test = y_train

#Fit the model with X_train and y_train

xgbm = xgb.XGBClassifier(max_depth=5, n_estimators=100, learning_rate=1).fit(X_train,y_train)



#Predict the response values for the observations in X_train

prediction = xgbm.predict(X_test)

#Compute the scoxgbmre for the random forest model

xgbm.score(X_train,y_train)
#Logistic Regression for numerical variables



logreg = LogisticRegression()



#Fit the model with X_train and y_train

logreg.fit(X_train,y_train)



#Predict the response values for the observations in X_train

prediction = logreg.predict(X_test)





#Check how many predictions were generated

len(prediction)

print(len(prediction))

print(prediction)





#Compute the score for the logistic regression model

logreg.score(X_train,y_train)
#Random Forest for numerical variables

random_forest = RandomForestClassifier(n_estimators=100)



#Fit the model with X_train and y_train

random_forest.fit(X_train,y_train)



#Predict the response values for the observations in X_train

prediction = random_forest.predict(X_test)

#Compute the score for the random forest model

random_forest.score(X_train,y_train)
test_data = test_data_only_first_class_female_Cherbourg_age_under_fourteen_not_over_fiftyone

submission = pd.DataFrame({

        "PassengerId": test_data["PassengerId"],

        "Survived": prediction

    })

submission.to_csv('titanic.csv', index=False)