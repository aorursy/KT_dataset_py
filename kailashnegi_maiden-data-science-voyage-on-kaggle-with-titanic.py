#Libraries for data processing and analysis

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#Library to access directory related functions

import os



#Libraries for data visualization

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



#To ignore warnings

import warnings

warnings.filterwarnings("ignore")
#Checking files in the directory

print(os.listdir("../input"))
data_train = pd.read_csv("../input/train.csv")

data_test = pd.read_csv("../input/test.csv")
#Shape of dataset

print("Shape of train set: ",data_train.shape)

print("Shape of test set: ",data_test.shape)
#Columns or features in the dataset

data_train.columns
data_train.head()
data_test.head()
data_train.describe()
Missing_Val = pd.concat([data_train.isnull().sum(),data_test.isnull().sum()],axis=1,keys=['Train DataSet', 'Test DataSet'])
Missing_Val[Missing_Val.sum(axis=1)>0]
#Let see which is the most frequent port of embarkation in the data.

data_train.groupby(["Embarked"]).count()["PassengerId"]
#Imputing Age and Embarked columns and Removing Cabin column. Will impute Age with median Age and Embarked column with S (most frequent). 

data_train["Age"] = data_train["Age"].fillna(data_train["Age"].median())

data_train["Embarked"]=data_train["Embarked"].fillna("S")

#Dropping Cabin column

data_train.drop(columns=["Cabin"],axis=1,inplace=True)

data_train.isnull().sum()
#Let's plot a bar graph of Survival by class



sns.barplot(x="Pclass",y="Survived",data=data_train).set_title("Pclass vs. Survived")

#Let see if there is some relationship between people who survived and sex



sns.barplot(x="Sex",y="Survived",data=data_train).set_title("Sex vs. Survived")
#Let see if point of embarkation has any relation with Survival

sns.barplot(x="Embarked",y="Survived",data=data_train).set_title("Embarked vs. Survived")
sns.catplot(x="Embarked",y="Fare",hue="Survived",data=data_train,kind="violin",height=5,aspect=1,legend=True,palette={0:"r",1:"g"})
#Let's look at a relation between, Pclass, Embarkation point, Age and Survived.

sns.catplot(x="Age", y="Embarked",hue="Survived", row="Pclass",data=data_train,orient="h", height=5, aspect=3, palette={0:"r",1:"g"},kind="violin", dodge=True, cut=0, bw=.2)
#Let see if point of Sibsp has any relation with Survival

sns.barplot(x="SibSp",y="Survived",data=data_train).set_title("Number of Siblings and Spouses vs. Survived")
#Checking which age group was high in numbers

sns.distplot(a=data_train["Age"],kde=False)
#Let's make bin of age groups and then see it's relationship with Survival

bins = [-1,12, 17, 24, 35, 60, np.inf]

labels = ['Child', 'Teenager', 'Young Adult', 'Adult', 'Middle Aged', 'Senior']

data_train['AgeGroup'] = pd.cut(data_train["Age"], bins, labels = labels)

data_train.head()
sns.barplot(x="AgeGroup",y="Survived",data=data_train).set_title("Age vs. Survived")
#Let's make bin of age groups and then see it's relationship with Survival

bins = [0,50, 100, 200, np.inf]

labels = ['Basic Economy', 'Regular Economy', 'Luxury', 'Super Luxury']

data_train['Fare_Bin'] = pd.cut(data_train["Fare"], bins, labels = labels)

data_train.head()
sns.barplot(x="Fare_Bin",y="Survived",data=data_train).set_title("Fare vs. Survived")
data_train = pd.read_csv("../input/train.csv")

data_test = pd.read_csv("../input/test.csv")



#Storing target variable in a temporary variable for later use

Response_Var = data_train.Survived



data_train.drop(["Survived"],axis=1,inplace=True)

data_combined = data_train.append(data_test)

data_combined.reset_index(inplace=True)

data_combined.drop(["Ticket","Cabin"],axis=1,inplace=True)
data_combined.head()
data_combined.shape
title = set()

for i in data_combined["Name"]:

    title.add(i.split(",")[1].split(".")[0].strip())

print(title)
# Let's make a dictionary to map all the titles with categories of titles



title_dict = {

    "Jonkheer" : "Royalty",

    "Dr" : "Officer",

    "Mme" : "Mrs",

    "Major" : "Officer",

    "Rev" : "Officer",

    "Mr" : "Mr",

    "Dona" : "Royalty",

    "Ms" : "Mrs",

    "Mrs" : "Mrs",

    "Mlle" : "Miss",

    "Master" : "Master",

    "Don" : "Royalty",

    "Sir" : "Royalty",

    "the Countess" : "Royalty",

    "Lady" : "Royalty",

    "Col" : "Officer",

    "Miss" : "Miss",

    "Capt" : "Officer"

}



#Creating a new column Title

data_combined["Title"] = data_combined["Name"].map(lambda name:name.split(',')[1].split('.')[0].strip())



#Mapping Title with title categories

data_combined["Title"] = data_combined["Title"].map(title_dict)

data_combined.head()
#To check if there is any row where Title has not been filled correctly.

data_combined["Title"].isnull().sum()
#Now we can drop Name column as we've extracted Title and Name column won't add any value to our model anymore.

data_combined.drop(["Name"],axis=1,inplace=True)

data_combined.head()
#Calculating median age based on class, sex, title for training data only. Then we can use this dataframe to impute our Age column in both Train and Test data set

train_grp = data_combined.iloc[:891].groupby(["Pclass","Sex","Title"])

train_grp_median = train_grp.median()

train_grp_median = train_grp_median.reset_index()[["Pclass","Sex","Title","Age"]]

train_grp_median
#Let's impute Age column in our combined data set

for i in range(len(data_combined["Age"])):

    if pd.isnull(data_combined["Age"].iloc[i]):

        condition = (train_grp_median["Sex"]==data_combined["Sex"].iloc[i]) & (train_grp_median["Title"]==data_combined["Title"].iloc[i]) & (train_grp_median["Pclass"]==data_combined["Pclass"].iloc[i])

        data_combined["Age"].iloc[i] = train_grp_median[condition]["Age"].iloc[0]
data_combined["Fare"].fillna(data_combined[:891]["Fare"].mean(), inplace=True)
data_combined["Embarked"].fillna(data_combined[:891]["Embarked"].mode()[0],inplace=True)
#Let's check if there are variables with missing values

data_combined.isnull().sum()
#Let's see which all variables we need to encode

data_combined.dtypes
# We will use pd.get_dummies() to encode



Pclass_dum = pd.get_dummies(data=data_combined["Pclass"],prefix="Pclass",prefix_sep="_")

#Adding dummy variables into main dataset

data_combined = pd.concat([data_combined,Pclass_dum],axis=1)

#Dropping the original Pclass variable since it is not required now

data_combined.drop(labels="Pclass",axis=1,inplace=True)

data_combined.head()
#Since Sex has only two categories so we can just convert them to 0 and 1 so no need to create dummies

data_combined["Sex"] = data_combined["Sex"].map({"male":1,"female":0})

data_combined.head()
#Creating dummies for Embarked column

Embarked_dum = pd.get_dummies(data=data_combined["Embarked"],prefix="Embarked",prefix_sep="_")

data_combined = pd.concat([data_combined,Embarked_dum],axis=1)

data_combined.drop(labels="Embarked",axis=1,inplace=True)

data_combined.head()
#Creating dummies for title column

Title_dum = pd.get_dummies(data=data_combined["Title"],prefix="Title",prefix_sep="_")

data_combined = pd.concat([data_combined,Title_dum],axis=1)

data_combined.drop(labels="Title",axis=1,inplace=True)

data_combined.head()
data_combined.reset_index()

data_combined.drop(labels="index",axis=1,inplace=True)
data_combined.head()
# Feature Importance

from sklearn import metrics

from sklearn.ensemble import ExtraTreesClassifier

train = data_combined[:891]

test = data_combined[891:]

# fit an Extra Trees model to the data

Feature_Imp_model = ExtraTreesClassifier()
#We had already created a variable called Response_Var as target 

Feature_Imp_model.fit(train, Response_Var)

# display the relative importance of each attribute

print(Feature_Imp_model.feature_importances_)
Importance_Df = pd.DataFrame()

Importance_Df["Variables"] = train.columns

Importance_Df["Importance"] = Feature_Imp_model.feature_importances_

Importance_Df.sort_values(by=['Importance'],ascending=True,inplace=True)

Importance_Df.set_index("Variables", inplace=True)
Importance_Df.plot(kind="bar")
from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
predictors = train.drop(["PassengerId"],axis=1)

target = Response_Var

x_train,x_validation,y_train,y_validation = train_test_split(predictors,target,test_size=0.25,random_state=123)
Logistic_Regression = LogisticRegression()

K_Nearest_Neighbor = KNeighborsClassifier()

Decision_Tree = DecisionTreeClassifier()

Support_Vector = SVC()

Random_Forest_Classifier = RandomForestClassifier()

Gradient_Boosting_Classifier = GradientBoostingClassifier()

model_lst = [Logistic_Regression,K_Nearest_Neighbor,Decision_Tree,Support_Vector,Random_Forest_Classifier,Gradient_Boosting_Classifier]

    
#Running models and validating

for model in model_lst:

    model.fit(x_train,y_train)

    y_predicted = model.predict(x_validation)

    print("Accuracy of {0} model is {1}".format(str(model.__class__).split(".")[3].split("'")[0],accuracy_score(y_validation,y_predicted)))
PassengerId = test["PassengerId"]

prediction = Gradient_Boosting_Classifier.predict(test.drop("PassengerId",axis=1))
output_df = pd.DataFrame({"PassengerId" : PassengerId, "Survived" : prediction})

output_df.to_csv("Submission.csv",index=False)