import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np # linear algebra

import matplotlib.pyplot as plt #plotting library

import seaborn as sns #statistical data visualization

import os

%matplotlib inline
train = pd.read_csv('../input/train.csv') #importing the trainning data set

train.info() #quick look at all the colums and data types. Also good way to check if the data was imported correctly.
train.describe() #Quick statisical overview of the numerical (int64,float64) data. 

#Object types will not show up.ie. Name,Sex Ticket, Cabin, and Embarked

train.head(15) #Gives you the the first 15 rows of data
#Looking at the the head of the dataframe we can assume that PassenderID and Ticket 

#are random unique identififier and will have no impact on the predictive outcome 

#so I will drop them from the dataframe



#Dopping Passenger ID and Ticket becasue it will have no vlaue to our machine learning model

train.drop(['PassengerId','Ticket'],axis=1,inplace=True)
#Now checking for missing values in dataset

plt.subplots(figsize=(9,5))

sns.heatmap(train.isnull(),yticklabels=False, cbar=False, cmap="YlGnBu_r")
#With so many null values in Cabin I will go ahead and drop it

train.drop('Cabin',axis=1,inplace=True)

#Age has a good amount of null valuse but I will find a way to handle the null values later.

#Embarked also has a few missing values but I will handle that later too.
#Plotting Survived against Sex

plt.subplots(figsize=(9,5))

sns.countplot(x='Survived',hue='Sex',palette='Set2',data=train)

#As we can see from the countplot below that Male's have less chance to survive than Female's.

#So there is a strong possibility that sex may play an important role in the prediciton of who

#survived

#Plotting Survived against PClass

sns.catplot(x="Pclass", col="Survived",palette='Set2',data=train,kind="count")



#Pclass could play an important roll in the prediction of who survied, with more surviving 

#than dying in Pclass 1 and almost 4 times the amount of people dying in Pclass 3.
#Plotting Survived against Embarked

sns.catplot(x="Embarked", col="Survived",palette='Set2',data=train,kind="count")
# Correlation matrix between numerical values (SibSp Parch Age, and Fare values) and Survived 

#Checking for multicollinearity (also known as collinearity) which are two or more explanatory 

#variables in a multiple regression model that are highly linearly related. 



plt.subplots(figsize=(9,5))

ax = sns.heatmap(train[["Survived","Pclass","Age","SibSp","Parch","Fare"]].corr(),annot=True, fmt = ".2f",cmap="Blues")

#From the correlation heatmap below no variable seems to highly correlated with another 

#so I won't have to drop any.
def impute_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    if pd.isnull(Age):

        if Pclass == 1:

            return 37

        elif Pclass == 2:

            return 29

        else:

            return 24

    else:

        return Age



train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)



#Checking agian to see if there are any more missing values in dataset

#sns.heatmap(train.isnull(),yticklabels=False,cbar=False)

plt.subplots(figsize=(9,5))

sns.heatmap(train.isnull(),yticklabels=False, cbar=False, cmap="YlGnBu_r")

#We can see from the heatmap above that we have handled all the missing Age values 

#but we can see that we still have a few Embarked rows/values we need to deal with.

#To do so we will will complete the Embarked NA rows/values with the mode values of the column.

train['Embarked'].fillna(train['Embarked'].mode()[0],inplace=True)
#Quick look at how Embarked, Age, Sex, and Pclass are spread out. One interesting things: Nobody with title rare in pclass 3. 

a = sns.catplot(x="Age", y="Embarked",hue="Sex", row="Pclass",

                 data=train,orient="h", height=4, aspect=3, palette="Set2",

                 kind="violin", dodge=True, cut=0, bw=.2)
#Checking the distribution of the fare variable

train['Fare'].hist(bins = 60,color="g")

#Looking the the distribution we can see that fare is skewed to the right.

#We will apply a log transformation to the Fare variable to reduce skewness distribution

train["Fare"] = train["Fare"].map(lambda i: np.log(i) if i > 0 else 0)
#We can see that after we apply the log transformation the distribution looks more like a normal distribution

#train['Fare'].hist(bins = 60,color="g",)

sns.distplot(train["Fare"],bins = 60,color="g")
#Dropping SibSp and Parch but creating a family feature with them.

train['FamilySize'] = train['SibSp'] + train['Parch'] +1

train.drop('SibSp',axis=1,inplace=True)

train.drop('Parch',axis=1,inplace=True)
train.head()
train_title = [i.split(",")[1].split(".")[0].strip() for i in train["Name"]]

train["Title"] = pd.Series(train_title)

train["Title"].head()
train.head()
#Here we will create four separate categories

#Converting Title to categorical values

train["Title"] = train["Title"].replace(['Lady','the Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'],'Rare')

train["Title"] = train["Title"].replace('Mlle','Miss')

train["Title"] = train["Title"].replace('Ms','Miss')

train["Title"] = train["Title"].replace('Mme','Mrs')

#Adding dummy variables to the Title column. More on this later in section 2.4

title = pd.get_dummies(train['Title'],drop_first=True)

train = pd.concat([train,title],axis=1)

train.head()
#Quick look at how Title, Age, Sex, and Pclass are spread out. 

#One interesting things: Nobody with title rare in pclass 3.

b = sns.catplot(x="Age", y="Title",hue="Sex", row="Pclass",

                data=train,orient="h", height=4, 

                aspect=3, palette="Set2",kind="violin")
#Now that I have what I want from the Name variable I will drop Name

#I can also go ahead and drop Title because I already have the information in the indivdual titles

train.drop(['Name','Title'],axis=1,inplace=True)

train.head()
#Adding dummy variables to Sex and Embarked

sex =  pd.get_dummies(train['Sex'],drop_first=True)

embark = pd.get_dummies(train['Embarked'],drop_first=True)

train = pd.concat([train,sex,embark],axis=1)

train.head()
#Now we can go ahead and drop the Sex and Embarked column because we have the needed information at the end with male, Q, S. 

train.drop(['Sex','Embarked'],axis=1,inplace=True)

train.head()
#First we have to split out dataset into X and Y. X being the all the variable and Y being Survived or not i.e. 0 or 1.

x = train.drop('Survived',axis=1)

y = train['Survived']
#Next we split the dataset into the train and test set

#Test will be 30% of the data and the train will be 70%. By setting test_size = .3

#This way we can test our models predictions on the test set to see how we did.

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=1)
#from sklearn.model_selection import GridSearchCV

#from sklearn.ensemble import RandomForestClassifier



#rf = RandomForestClassifier(max_features='auto', oob_score=True, random_state=1, n_jobs=-1)



#param_grid = { "criterion" : ["gini", "entropy"], "min_samples_leaf" : [1,2,3,5], "min_samples_split" : [10,11,12,13], "n_estimators": [350, 400, 450, 500,550], "max_depth":[6,7,8,9]}



#gs = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='accuracy', cv=3, n_jobs=-1)



#gs = gs.fit(train.iloc[:, 1:], train.iloc[:, 0])



#print(gs.best_score_)

#print(gs.best_params_)

#print(gs.scorer_)



#Example of the output

#0.8451178451178452

#{'criterion': 'gini', 'max_depth': 8, 'min_samples_leaf': 1, 'min_samples_split': 11, 'n_estimators': 375}
#Building the Random Forest Classification model

from sklearn.ensemble import RandomForestClassifier

rfmodel = RandomForestClassifier(random_state=0,n_estimators=450,criterion='gini',n_jobs=-1,max_depth = 8,min_samples_leaf=1,min_samples_split= 11)

#Fitting the model to x_train and y_train

rfmodel.fit(x_train,y_train)

#Predicting the model on the x_test

predictions = rfmodel.predict(x_test)
#classification report showing are predictions vs the actually result

from sklearn.metrics import classification_report

print(classification_report(y_test,predictions))
#Printing out the confusion matrix. 

from sklearn.metrics import  confusion_matrix

confusion_matrix(y_test,predictions)
import eli5

from eli5.sklearn import PermutationImportance



perm = PermutationImportance(rfmodel, random_state=1).fit(x_test, y_test)

eli5.show_weights(perm, feature_names = x_test.columns.tolist())
#Applying K-Fold Cross Validation

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator=rfmodel,X= x_train,y=y_train,cv=10)

accuracies #Prints out the 10 different Cross Validation scores.

#As you can see there seems to be a decent amount of variation from as low as .8033 to as hiigh as .9048.
accuracies.mean() #Prints out the  average of the 10 scores.