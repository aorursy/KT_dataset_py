# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



titanic=pd.read_csv("../input/train.csv")

titanic_test=pd.read_csv("../input/test.csv")



titanic.head()

# Any results you write to the current directory are saved as output.
#Let's look at the data

titanic.describe()
#Age has some missing values (count=714, all other counts=891)

#We can use the median age to fill in the missing values



titanic["Age"]=titanic["Age"].fillna(titanic["Age"].median())



#Let looks at the values again to see what is looks like now

titanic.describe()
#Convert Character variables to numeric 

#Characters varaibles include: Name, Sex, Ticket, Cabin and Embarked

#We will not use name, ticket and cabin in our prediction so we can ignore those ones for now



#lets convert Sex to numeric by assinging all 'male' values to the number 0 and all  'female' values to the number 1

titanic.loc[titanic['Sex']=='male','Sex']=0

titanic.loc[titanic['Sex']=='female','Sex']=1



#Now lets conver 'Embarked' to numeric

#Embarked has some missing values so we need to deal with those first. S is the most common embarked locations so we will assume all missing values are S

titanic['Embarked']=titanic['Embarked'].fillna('S')



#Now we will convert 'Embarked' to numeric:

#S=0

#C=1

#Q=2

titanic.loc[titanic['Embarked']=='S','Embarked']=0

titanic.loc[titanic['Embarked']=='C','Embarked']=1

titanic.loc[titanic['Embarked']=='Q','Embarked']=2

print(titanic['Embarked'].unique())
#Build a Linear Regression Model

#Import linear regression class from sklearn

from sklearn.linear_model import LinearRegression



#import the sklearn helper for cross validation

from sklearn.cross_validation import KFold



#Determine what the predictors will be:

predictors=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']



#Initalize the algorithm class

alg=LinearRegression()



#Create cross Validation folds for the train dataset



#Set random_state to ensure we get the same splits every time we run this



kf=KFold(titanic.shape[0], n_folds=3, random_state=1)



predictions=[]

for train, test in kf:

    #The predictors from our new training dataset

    train_predictors=(titanic[predictors].iloc[train,:])

    #The target variables we're using for our algorithm

    train_target=titanic["Survived"].iloc[train]



    #Now lets train the algorithm with our predictors and target

    alg.fit(train_predictors, train_target)



    

    #Now we can make predictions on our new test dataset

    test_predictions=alg.predict(titanic[predictors].iloc[test,:])

    predictions.append(test_predictions)
#Define error metric, we'll use percentage of correct predictions



#we need to concatenate are 3 numpy array for predictions

predictions=np.concatenate(predictions, axis=0)



#We need the predictions to be 0 or 1

predictions[predictions>0.5]=1

predictions[predictions<=0.5]=0



#Calculate accuracy

accuracy=sum(predictions[predictions==titanic["Survived"]])/len(predictions)

print(accuracy)
#Our accuracy is not very good in our first model (78.3%)

#Lets try using logistic regression to output values between 0 and 1

from sklearn.linear_model import LogisticRegression

#Import the cross validation package

from sklearn import cross_validation



#initalize our algorithm

alg=LogisticRegression(random_state=1)



#compute the accuracy score for all the cross validation fold



scores=cross_validation.cross_val_score(alg,titanic[predictors],titanic["Survived"], cv=3)



print(scores.mean())
#Now we need to repeat the steps above on the titanic_test data set to submit our predictions



#Fill missing age with median age from Titanic Dataset

titanic_test['Age']=titanic_test['Age'].fillna(titanic['Age'].median())



#Convert Sex to numeric

titanic_test.loc[titanic_test['Sex']=='male','Sex']=0

titanic_test.loc[titanic_test['Sex']=='female','Sex']=1



#fill missing Embarked data with S

titanic_test['Embarked']=titanic_test['Embarked'].fillna('S')



#Convert Embarked to numeric

titanic_test.loc[titanic_test['Embarked']=='S', 'Embarked']=0

titanic_test.loc[titanic_test['Embarked']=='C', 'Embarked']=1

titanic_test.loc[titanic_test['Embarked']=='Q', 'Embarked']=2



#The test dataset has a missing Fare so we will use the median of Titanic_test fares

titanic_test['Fare']=titanic_test['Fare'].fillna(titanic_test['Fare'].median())
#Now we just build an algorithm on the training data and make predictions on the test data



#Initalize the algorithm

alg=LogisticRegression(random_state=1)



#Train the algorithm using all the training data

alg.fit(titanic[predictors],titanic["Survived"])



#And make the predictions on the test data

predictions=alg.predict(titanic_test[predictors])



#Create a datafram with only the passangerID and Survived



submission=pd.DataFrame({

    "PassengerId":titanic_test["PassengerId"],

    "Survived":predictions

})
#Our Logisitc Regression accuracy was not great (~75%), so lets try to improve our model



#import sklearn random forest implementation

from sklearn import cross_validation

from sklearn.ensemble import RandomForestClassifier



predictors=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']



#Initialize are algorthm w/defaul parameters:

    #n_estimators is the number of trees we want to make

    #min_samples_split is the minimum number of rows we need to make a split

    #min_samples_leaf is the minimum number of samples we can have at the place where a tree branch ends (the bottom points of the tree)

    

alg=RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=4, min_samples_leaf=2)

    

#Make cross validations predictions

scores=cross_validation.cross_val_score(alg,titanic[predictors],titanic['Survived'],cv=3)

    

print(scores.mean())
#Lets add some additional Variables to see if we can improve our model:



titanic['FamilySize']=titanic['SibSp']+titanic['Parch']



#Length of name could imply how wealthy someone is

titanic['NameLength']=titanic['Name'].apply(lambda x: len(x))



#Use a regular expression to extract titles from name:

import re



#Create a function to search names for titles

def get_title(name):

    title_search=re.search(' ([A-za-z]+)\.', name)

    #If a title is found extract it and return it

    if title_search:

        return title_search.group(1)

    return ""



titles=titanic["Name"].apply(get_title)





#Map each title to an integer so we can include it in our model, some titles are rare and were grouped with other titels:

title_mapping={'Mr':1,'Miss':2, 'Mrs':3,'Master':4,'Dr':5,'Rev':6,'Major':7,'Col':8,'Mlle':8,'Mme':8,

               'Don':9,'Lady':10,'Countess':10,'Jonkheer':10,'Sir':9,'Capt':7,'Ms':2}



for k,v in title_mapping.items():

    titles[titles==k]=v

    

print(pd.value_counts(titles))



titanic["Title"]=titles
#A persons survival might have been influenced by if their family members survived, we can use last name and familysize to get a unique familyID variable



import operator



#A dictionary mapping family name to id

family_id_mapping={}



#A function to get the id given a row

def get_family_id(row):

    last_name=row["Name"].split(",")[0]

    family_id="{0}{1}".format(last_name, row['FamilySize'])

    

    #We can now lookup a family ID in the mapping

    if family_id not in family_id_mapping:

        if len(family_id_mapping)==0:

            current_id=1

        else:

            current_id=(max(family_id_mapping.items(), key=operator.itemgetter(1))[1]+1)

        family_id_mapping[family_id]=current_id

    return family_id_mapping[family_id]



family_ids=titanic.apply(get_family_id,axis=1)



#compress all families smaller than 3 into 1 code

family_ids[titanic['FamilySize']<3]=-1



titanic['FamilyId']=family_ids



print(pd.value_counts(family_ids))
#feature selection is an important part of model building

#We can use univariate feature selection to help determine which columns correlate most closely with what we're trying to predict(Survived)



from sklearn.feature_selection import SelectKBest, f_classif

#Lets update our predictors first

predictors=['Pclass', 'Sex','Age','SibSp','Parch','Fare','Embarked','FamilySize','Title','FamilyId']



#Perform feature selection

selector=SelectKBest(f_classif,k=5)

selector.fit(titanic[predictors],titanic["Survived"])



#Get p-values for each selector

scores=-np.log10(selector.pvalues_)



#plot the scores

import matplotlib.pyplot as plt

plt.bar(range(len(predictors)),scores)

plt.xticks(range(len(predictors)),predictors,rotation='vertical')

plt.show()



#Lets build the model with the 4 best predictors

predictors=['Pclass','Sex','Fare','Title']



alg=RandomForestClassifier(random_state=1,n_estimators=150,min_samples_split=8, min_samples_leaf=4)



scores=cross_validation.cross_val_score(alg,titanic[predictors],titanic["Survived"],cv=3)



print(scores.mean())
#Gradient boosting classifier builds on decisions trees, this method uses the error from previous trees to build new trees

#This method can lead to overfitting, you can help this by limiting the number of trees and the tree depth



from sklearn.ensemble import GradientBoostingClassifier



algorithms= [

    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3),["Pclass","Sex","Age","Fare","Embarked","FamilySize","Title","FamilyId"]],

[LogisticRegression(random_state=1),["Pclass","Sex","Fare","FamilySize","Title","Age","Embarked"]]

]



#Initialize the cross validation folds

kf=KFold(titanic.shape[0],n_folds=3,random_state=1)



predictions=[]

for train, test in kf:

    train_target=titanic["Survived"].iloc[train]

    full_test_predictions=[]

    for alg, predictors in algorithms:

        alg.fit(titanic[predictors].iloc[train,:],train_target)

        test_predictions=alg.predict_proba(titanic[predictors].iloc[test,:].astype(float))[:,1]

        full_test_predictions.append(test_predictions)

    test_predictions=(full_test_predictions[0]+full_test_predictions[1])/2

    test_predictions[test_predictions<=.5]=0

    test_predictions[test_predictions>.5]=1

    predictions.append(test_predictions)

        

predictions=np.concatenate(predictions,axis=0)



accuracy=sum(predictions[predictions==titanic["Survived"]])/len(predictions)



print(accuracy)

        
#Now we need to add the variables we created to our test set so we can run the predictions on it



titles=titanic_test["Name"].apply(get_title)



title_mapping['Dona']=10

print(title_mapping)



for k,v in title_mapping.items():

    titles[titles==k]=v

titanic_test["Title"]=titles



print(pd.value_counts(titanic_test['Title']))



titanic_test["FamilySize"]=titanic_test['SibSp']+titanic_test['Parch']



print(family_id_mapping)



family_ids=titanic_test.apply(get_family_id,axis=1)

family_ids[titanic_test["FamilySize"]<3]=-1

titanic_test["FamilyId"]=family_ids



titanic_test['NameLength']=titanic_test['Name'].apply(lambda x : len(x))



print(titanic_test["NameLength"])
#Now lets make our predictions on the test dataset and create a submission dataframe

predictors=['Pclass','Sex','Age','Fare','Embarked','FamilySize','Title','FamilyId']



algorithms=[

    [GradientBoostingClassifier(random_state=1,n_estimators=25, max_depth=3),predictors],

    [LogisticRegression(random_state=1),['Pclass','Sex','Fare','FamilySize','Title','Age','Embarked']]

]



full_predictions=[]

for alg, predictors in algorithms:

    alg.fit(titanic[predictors],titanic["Survived"])

    predictions=alg.predict_proba(titanic_test[predictors].astype(float))[:,1]

    full_predictions.append(predictions)

    

predictions=(full_predictions[0]*3+full_predictions[1])/4

predictions[predictions<=.5]=0

predictions[predictions>.5]=1

predictions=predictions.astype(int)



submission=pd.DataFrame({

    "PassengerId": titanic_test["PassengerId"],

    "Survived":predictions

})

submission.to_csv("submission.csv",index=False)
#Looking for ways to improve the model

titanic.head(40)

Families=titanic[titanic['FamilySize']>3]

cabin=titanic[titanic['Cabin'].notnull()]

cabin.head(100)

Fortune=titanic[titanic['']]
#Trying to improve the model with additiona Features:



#Cabin Features



#number of women in family



#National origin of Last name
#Trying to imporve the model itself:



#try randomforest classifier in the ensemble



#Support Vector Machine



#Neural networks



#Boosting with different base classifier

algorithms= [

    [GradientBoostingClassifier(random_state=1, n_estimators=30, max_depth=3),["Pclass","Sex","Age","Fare","Embarked","FamilySize","Title","FamilyId"]],

[LogisticRegression(random_state=1),["Pclass","Sex","Fare","FamilySize","Title","Age","Embarked"]]

]



#Initialize the cross validation folds

kf=KFold(titanic.shape[0],n_folds=3,random_state=1)



predictions=[]

for train, test in kf:

    train_target=titanic["Survived"].iloc[train]

    full_test_predictions=[]

    for alg, predictors in algorithms:

        alg.fit(titanic[predictors].iloc[train,:],train_target)

        test_predictions=alg.predict_proba(titanic[predictors].iloc[test,:].astype(float))[:,1]

        full_test_predictions.append(test_predictions)

    test_predictions=(full_test_predictions[0]+full_test_predictions[1])/2

    test_predictions[test_predictions<=.5]=0

    test_predictions[test_predictions>.5]=1

    predictions.append(test_predictions)

        

predictions=np.concatenate(predictions,axis=0)



accuracy=sum(predictions[predictions==titanic["Survived"]])/len(predictions)



print(accuracy)

        