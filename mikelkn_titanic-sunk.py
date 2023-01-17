# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#part 1- define and read  all and then chek to see if there are any missing variables anywhere
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

titanic = pd.read_csv('../input/train.csv') # shape = 891 x 12
titanic_test = pd.read_csv('../input/test.csv') #shape = 418x 11
 #let do some cleaning of data we are going to need th Pclass, Sex,
#and children column to make predictions. eliminate all NaNs
print(titanic.head(15))
                              
# first the Age columns
data_all = [titanic, titanic_test]
for x in data_all:
    x["Age"] = x["Age"].fillna(x["Age"].mean())
# if embraked has some missing data
Embarked= titanic["Embarked"].value_counts(normalize = True)
#print(Embarked)
#we replace them with the majority S
data_all = [titanic, titanic_test]
for x in data_all:
    x["Embarked"] = x["Embarked"].fillna ("S")
print(titanic.head(15))    #embarked all good
#from above we can see tht we have 5 objects and 2 floats we need to convert to numerical features
#First up - Fare - we fill it up and we change it to an int64 using astype()
data_all = [titanic, titanic_test]
for y in data_all:
    y["Fare"] = y["Fare"].fillna(0)
    y["Fare"] = y["Fare"].astype(int)    #check the titanic.info() and you see fare is all good too
#second up- Age
    y["Age"] = y["Age"].astype(int)      #not more floats now
    titanic.info()
    #print(titanic.head(15)) 
#we need numerical features for sex, name, cabin and tickets so we try to format them
#SEX for both the train and the test features
data_all = [titanic, titanic_test]
input_sex = {'male': 0,'female': 1}
input_embarked = {'S': 0, 'Q' : 1, 'C': 2}
for v in data_all:
    #v["Sex"][v["Sex"] == "male"] = 0    no good this way, better way below
    #v["Sex"][v["Sex"] == "female"] = 1
    v["Sex"] = v["Sex"].map(input_sex)
    #embarked
    v["Embarked"] = v["Embarked"].map(input_embarked)
    
#print(titanic.head(15))
#too complicated to change ticket and cabin so i may just have to drop the columns so
print(titanic.head(15)) 


titanic.info()


titanic.drop(titanic.columns[[3,8,10]], axis=1, inplace = True)
titanic_test.drop(titanic_test.columns[[2,7,9]], axis=1, inplace = True)

print(titanic.head(15))

from sklearn.ensemble import RandomForestRegressor
# we train our data
myprediction_model = RandomForestRegressor()      #define my model
y_train = titanic["Survived"].values
predictors_train = ["Age", "Sex", "Pclass", "Fare"]
X_train = titanic[predictors_train].values
myprediction_model.fit(X_train, y_train)           # i fit or train it
# we start testing them witht the predictors_train column above
X_test = titanic_test[predictors_train]
#then i predict
predicted_survival = myprediction_model.predict(X_test)
print(predicted_survival.shape)

my_titanic_submission["Survived"] = my_titanic_submission["Survived"].astype(int)
print(my_titanic_submission)
my_titanic_submission = pd.DataFrame({"PassengerId": titanic_test["PassengerId"],
                             "Survived":predicted_survival})
my_titanic_submission.to_csv('TitanicSubmission.csv', index = False)

