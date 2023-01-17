# data processing



import numpy as np

import pandas as pd 



# data visualization



import seaborn as sns

%matplotlib inline

from matplotlib import pyplot as plt



from sklearn import linear_model

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.naive_bayes import GaussianNB

from sklearn import model_selection

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn import feature_selection

from sklearn import metrics

from sklearn.preprocessing import StandardScaler,minmax_scale



import warnings

warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", None)

train= pd.read_csv("/kaggle/input/titanic/train.csv")

test= pd.read_csv("/kaggle/input/titanic/train.csv")

train_d = train.copy()

test_d = test.copy()
train.describe(include="all")
test.describe(include="all")
train.head()
test.head()
train.dtypes
train.shape
test.shape
train.info()
test.info()
train.mean()
print(train.columns)
print(test.columns)
train["Pclass"].value_counts()
test["Pclass"].value_counts()
#draw a bar plot of survival by Pclass

sns.barplot(x="Pclass", y="Survived", data=train)



total_survived_plcass1 = train[train.Pclass == 1]["Survived"].sum()

total_survived_plcass2 = train[train.Pclass == 2]["Survived"].sum()

total_survived_plcass3 = train[train.Pclass == 3]["Survived"].sum()

#print percentage of people by Pclass that survived

print("Total Pclass1 survived is: " + str((total_survived_plcass1)))

print("Total Pclass2 survived is: " + str((total_survived_plcass2)))

print("Total Pclass3 survived is: " + str((total_survived_plcass3)))

print("Total Pclass  survived is: " + str((total_survived_plcass1+total_survived_plcass2+total_survived_plcass3)))

print("Percentage of Pclass = 1 who survived:", train["Survived"][train["Pclass"] == 1].value_counts(normalize = True)[1]*100)

print("Percentage of Pclass = 2 who survived:", train["Survived"][train["Pclass"] == 2].value_counts(normalize = True)[1]*100)

print("Percentage of Pclass = 3 who survived:", train["Survived"][train["Pclass"] == 3].value_counts(normalize = True)[1]*100)
train["Sex"].value_counts()
test["Sex"].value_counts()
sns.barplot(x="Sex", y="Survived",data=train)



#print percentages of females vs. males that survive



total_survived_females = train[train.Sex == "female"]["Survived"].sum()

total_survived_males = train[train.Sex == "male"]["Survived"].sum()





print("Total female survived is: " + str((total_survived_females )))

print("Total   male survived is: " + str(( total_survived_males)))

print("Total people survived is: " + str((total_survived_females + total_survived_males)))

print("Percentage of females who survived:", train["Survived"][train["Sex"] == "female"].value_counts(normalize = True)[1]*100)

print("Percentage  of  males who survived:", train["Survived"][train["Sex"] == "male"].value_counts(normalize = True)[1]*100)
sns.barplot(x="Pclass", y="Survived", hue="Sex", data=train)

plt.ylabel("Survival Rate")

plt.title("Survival Rates Based on Gender and Class")

#help(sns.barplot)

train[["Sex","Survived"]].groupby("Sex").mean()

sns.barplot(x="Sex", y="Survived", hue="Pclass", data=train)

plt.ylabel("Survival Rate")

plt.title("Survival Rates Based on Gender and Class")
train["SibSp"].value_counts()
test["SibSp"].value_counts()
#draw a bar plot for SibSp vs. survival

sns.barplot(x="SibSp", y="Survived", data=train)



total_survived_sibs0 = train[train.SibSp == 0]["Survived"].sum()

total_survived_sibs1 = train[train.SibSp == 1]["Survived"].sum()

total_survived_sibs2 = train[train.SibSp == 2]["Survived"].sum()

total_survived_sibs3 = train[train.SibSp == 3]["Survived"].sum()

total_survived_sibs4 = train[train.SibSp == 4]["Survived"].sum()

print("Total SibSb 0 survived is: " + str((total_survived_sibs0 )))

print("Total SibSb 1 survived is: " + str((total_survived_sibs1 )))

print("Total SibSb 2 survived is: " + str((total_survived_sibs2 )))

print("Total SibSb 3 survived is: " + str((total_survived_sibs3 )))

print("Total SibSb 4 survived is: " + str((total_survived_sibs4 )))

print("Percentage of SibSp = 0 who survived:", train["Survived"][train["SibSp"] == 0].value_counts(normalize = True)[1]*100)

print("Percentage of SibSp = 1 who survived:", train["Survived"][train["SibSp"] == 1].value_counts(normalize = True)[1]*100)

print("Percentage of SibSp = 2 who survived:", train["Survived"][train["SibSp"] == 2].value_counts(normalize = True)[1]*100)

print("Percentage of SibSp = 3 who survived:", train["Survived"][train["SibSp"] == 3].value_counts(normalize = True)[1]*100)

print("Percentage of SibSp = 4 who survived:", train["Survived"][train["SibSp"] == 4].value_counts(normalize = True)[1]*100)
train["Parch"].value_counts()
test["Parch"].value_counts()
#draw a bar plot for Parch vs. survival

sns.barplot(x="Parch", y="Survived", data=train)

plt.show()
train["Embarked"].value_counts()
test["Embarked"].value_counts()
#draw a bar plot for Embarked vs. survival

sns.barplot(x="Embarked", y="Survived", data=train)

plt.show()
train["Age"].describe()
test["Age"].describe()
sns.boxplot(x=train["Age"]);
sns.boxplot(x=test["Age"]);
train["Fare"].describe()
test["Fare"].describe()
sns.boxplot(x=train["Fare"]);
sns.boxplot(x=test["Fare"]);
free_ticket_train=train[train["Fare"] == 0]["PassengerId"].count()
free_ticket_test=test[train["Fare"] == 0]["PassengerId"].count()
free_ticket_total=free_ticket_train+free_ticket_test
print("Free Tickets Total:"+str(free_ticket_total))
print("Fare free who survived:", train["Survived"][train["Fare"] == 0].value_counts(normalize = True))
train.isnull().sum()
import missingno as msno
msno.matrix(train);
msno.matrix(test);
train["Embarked"].value_counts()
#replacing the missing values in the Embarked feature with S

train = train.fillna({"Embarked": "S"})

test = train.fillna({"Embarked": "S"})
def create_Title(train):

    titles = {

        "Mr" :         "Mr",

        "Mme":         "Mrs",

        "Ms":          "Mrs",

        "Mrs" :        "Mrs",

        "Master" :     "Master",

        "Mlle":        "Miss",

        "Miss" :       "Miss",

        "Capt":        "Rare",

        "Col":         "Rare",

        "Major":       "Rare",

        "Dr":          "Rare",

        "Rev":         "Rare",

        "Jonkheer":    "Rare",

        "Don":         "Rare",

        "Sir" :        "Rare",

        "Countess":    "Rare",

        "Dona":        "Rare",

        "Lady" :       "Rare"

    }

    extracted_titles =train["Name"].str.extract(' ([A-Za-z]+)\.',expand=False)

    train["Title"] = extracted_titles.map(titles)
create_Title(train)

def create_Title(test):

    titles = {

        "Mr" :         "Mr",

        "Mme":         "Mrs",

        "Ms":          "Mrs",

        "Mrs" :        "Mrs",

        "Master" :     "Master",

        "Mlle":        "Miss",

        "Miss" :       "Miss",

        "Capt":        "Rare",

        "Col":         "Rare",

        "Major":       "Rare",

        "Dr":          "Rare",

        "Rev":         "Rare",

        "Jonkheer":    "Rare",

        "Don":         "Rare",

        "Sir" :        "Rare",

        "Countess":    "Rare",

        "Dona":        "Rare",

        "Lady" :       "Rare"

    }

    extracted_titles =test["Name"].str.extract(' ([A-Za-z]+)\.',expand=False)

    test["Title"] = extracted_titles.map(titles)
create_Title(test)
a=list(train["Age"][train["Age"].isnull()].index)
train["Title"].iloc[a] 
train.head()

train.groupby("Title")["Age"].mean()
train.groupby("Title")["Age"].median()
train["Age"].fillna(train.groupby("Title")["Age"].transform("mean"),inplace = True )
train["Title"].iloc[a] 
test["Age"].fillna(test.groupby("Title")["Age"].transform("mean"),inplace = True )
train = train.drop(["Cabin"], axis = 1)

test = test.drop(["Cabin"], axis = 1)
train.isnull().sum()
msno.matrix(train);
test.isnull().sum()
train = train.drop(["Ticket"], axis = 1)

test = test.drop(["Ticket"], axis = 1)
test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("mean"),inplace = True )
msno.matrix(test);
train_fare=train["Fare"]
sns.boxplot(x=train_fare);
Q1 = train_fare.quantile(0.05)

Q3 = train_fare.quantile(0.95)

IQR = Q3-Q1
low_limit = Q1-1.5*IQR

up_limit= Q3 + 1.5*IQR
low_limit
up_limit
(train_fare > up_limit)
train.loc[train["Fare"] > up_limit,"Fare"] = up_limit

test.loc[test["Fare"] > up_limit,"Fare"] = up_limit
sns.boxplot(x=train_fare);
lbe = LabelEncoder()

lbe.fit_transform(train["Sex"])

train["Sex"] = lbe.fit_transform(train["Sex"])

lbe.fit_transform(test["Sex"])

test["Sex"] = lbe.fit_transform(test["Sex"])
train.head()
test.head()
embarked_mapping = {"S": 1, "C": 2, "Q": 3}



train["Embarked"] = train["Embarked"].map(embarked_mapping)

test["Embarked"] = test["Embarked"].map(embarked_mapping)
train.head()
test.head()
train[["Title","PassengerId"]].groupby("Title").count()
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}



train["Title"] = train["Title"].map(title_mapping)

train.head(100)
test["Title"] = test["Title"].map(title_mapping)

test.head()
train = train.drop(["Name"], axis = 1)

test = test.drop(["Name"], axis = 1)
 ## Map Fare values into groups of numerical values:

train["FareBand"] = pd.qcut(train["Fare"], 4, labels = [1, 2, 3, 4])

test["FareBand"] = pd.qcut(test["Fare"], 4, labels = [1, 2, 3, 4])
# Drop Fare values:

train = train.drop(["Fare"], axis = 1)

test = test.drop(["Fare"], axis = 1)
train.head()
test.head()
train.head()
train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
test["FamilySize"] = test["SibSp"] + test["Parch"] + 1
# Create new feature of family size:



#train["Single"] = train["FamilySize"].map(lambda s: 1 if s == 1 else 0)

#train["SmallFam"] = train["FamilySize"].map(lambda s: 1 if  s == 2  else 0)

#train["MedFam"] = train["FamilySize"].map(lambda s: 1 if 3 <= s <= 4 else 0)

#train["LargeFam"] = train["FamilySize"].map(lambda s: 1 if s >= 5 else 0)
train.head()
# Create new feature of family size:



#test["Single"] = test["FamilySize"].map(lambda s: 1 if s == 1 else 0)

#test["SmallFam"] = test["FamilySize"].map(lambda s: 1 if  s == 2  else 0)

#test["MedFam"] = test["FamilySize"].map(lambda s: 1 if 3 <= s <= 4 else 0)

#test["LargeFam"] = test["FamilySize"].map(lambda s: 1 if s >= 5 else 0)
test.head()
train.corr()["Survived"].abs().sort_values(ascending=False)

#train = train.drop(["FamilySize"], axis = 1)

#test = test.drop(["FamilySize"], axis = 1)
corr = train.corr()
corr
train = pd.get_dummies(train, columns = ["Embarked"], prefix="Em")
train.head()
test = pd.get_dummies(test, columns = ["Embarked"], prefix="Em")
test.head()
train.corr()["Survived"].abs().sort_values(ascending=False)

train = pd.get_dummies(train, columns = ["Title"])

test = pd.get_dummies(test, columns = ["Title"])

train.head()
# Create categorical values for Pclass:

train["Pclass"] = train["Pclass"].astype("category")

train = pd.get_dummies(train, columns = ["Pclass"],prefix="Pc")

train.head()
test["Pclass"] = test["Pclass"].astype("category")

test = pd.get_dummies(test, columns = ["Pclass"],prefix="Pc")
test.head()
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score





y = train["Survived"]

x= train.drop(['Survived'], axis=1)



x_train, x_test, y_train, y_test = train_test_split(x,y,

                                                    test_size = 0.20, random_state = 42)
from sklearn.ensemble import RandomForestClassifier



rf_model = RandomForestClassifier().fit(x_train, y_train)

y_pred = rf_model.predict(x_test)

acc_randomforest = round(accuracy_score(y_test, y_pred) * 100, 2)

print(acc_randomforest)
from sklearn.ensemble import GradientBoostingClassifier



gbm_model = GradientBoostingClassifier().fit(x_train, y_train)



y_pred = gbm_model.predict(x_test)

acc_gbm = round(accuracy_score(y_test,y_pred) * 100, 2)

print(acc_gbm)
nb = GaussianNB()

nb_model=nb.fit(x_train, y_train)

nb_model

nb_model.predict(x_test)

y_pred = nb_model.predict(x_test)
acc_gnb = round(accuracy_score(y_test, y_pred)*100,2)

print(acc_gnb)
round(cross_val_score(nb_model, x_test, y_test, cv = 10).mean()*100,2)
from sklearn.tree import DecisionTreeClassifier
cart = DecisionTreeClassifier()

cart_model = cart.fit(x_train, y_train)



cart_model
!pip install skompiler

from skompiler import skompile

print(skompile(cart_model.predict).to("python/code"))
y_pred = cart_model.predict(x_test)

acc_dt=round(accuracy_score(y_test, y_pred)*100,2)
print(acc_dt)
knn= KNeighborsClassifier()

knn_model=knn.fit(x_train, y_train)

y_pred = knn_model.predict(x_test)

acc_knn = round(accuracy_score(y_test, y_pred)*100,2)

print(acc_knn)
loj= LogisticRegression(solver="liblinear")

loj_model=loj.fit(x_train, y_train)



pred_logreg = loj_model.predict(x_test)

acc_logreg = round(accuracy_score(y_test, pred_logreg)*100,2)



print(acc_logreg)
model_performance = pd.DataFrame({

    "Model": ["Random Forest", "Gradient Boosting", "Gaussian NB", 

               "K Nearest Neighbors","Logistic Regression",  

              "Decision Tree"],

    "Accuracy": [acc_randomforest, acc_gbm, acc_gnb, 

               acc_knn,acc_logreg,acc_dt]

})



model_performance.sort_values(by="Accuracy", ascending=False)
knn_params = {"n_neighbors": np.arange(1,50)}
knn = KNeighborsClassifier()

knn_cv = GridSearchCV(knn, knn_params, cv=10)

knn_cv.fit(x_train, y_train)
print("Best Score:" + str(knn_cv.best_score_))

print("Best parametrs: " + str(knn_cv.best_params_))
knn = KNeighborsClassifier(22)

knn_tuned=knn.fit(x_train, y_train)
round(knn_tuned.score(x_test, y_test)*100,2)
y_pred=knn_tuned.predict(x_test)
round(accuracy_score(y_test, y_pred)*100,2)
gbm_params = {"learning_rate" : [0.001, 0.01, 0.1, 0.05],

             "n_estimators": [100,500,100],

             "max_depth": [3,5,10],

             "min_samples_split": [2,5,10]}
gbm = GradientBoostingClassifier()



gbm_cv = GridSearchCV(gbm, gbm_params, cv = 10, n_jobs = -1, verbose = 2)
gbm_cv.fit(x_train, y_train)
print("Best Paramters: " + str(gbm_cv.best_params_))
gbm = GradientBoostingClassifier(learning_rate = 0.1, 

                                 max_depth = 3,

                                min_samples_split =10 ,

                                n_estimators = 500)
gbm_tuned =  gbm.fit(x_train,y_train)
y_pred = gbm_tuned.predict(x_test)

round(accuracy_score(y_test, y_pred)*100,2)
cart_grid = {"max_depth": range(1,10),

            "min_samples_split" : list(range(2,50)) }
cart = DecisionTreeClassifier()

cart_cv = GridSearchCV(cart, cart_grid, cv = 10, n_jobs = -1, verbose = 2)

cart_cv_model = cart_cv.fit(x_train, y_train)
print("Best Paramters: " + str(cart_cv_model.best_params_))
cart = DecisionTreeClassifier(max_depth = 3, min_samples_split = 2)

cart_tuned = cart.fit(x_train, y_train)
y_pred = cart_tuned.predict(x_test)

round(accuracy_score(y_test, y_pred)*100,2)
rf_params = {"max_depth": [2,5,8,10],

            "max_features": [2,5,8],

            "n_estimators": [10,500,1000],

            "min_samples_split": [2,5,10]}
rf_model = RandomForestClassifier()



rf_cv_model = GridSearchCV(rf_model, 

                           rf_params, 

                           cv = 10, 

                           n_jobs = -1, 

                           verbose = 2) 
rf_cv_model.fit(x_train, y_train)
print("Best Paramters: " + str(rf_cv_model.best_params_))
rf_tuned = RandomForestClassifier(max_depth = 5, 

                                  max_features = 5, 

                                  min_samples_split = 2,

                                  n_estimators = 10)



rf_tuned.fit(x_train, y_train)
y_pred = rf_tuned.predict(x_test)

round(accuracy_score(y_test, y_pred)*100,2)
Importance = pd.DataFrame( {"Importance": rf_tuned.feature_importances_*100},

                         index = x_train.columns)
Importance.sort_values(by = "Importance", 

                       axis = 0, 

                       ascending = True).plot(kind ="barh", color = "r")



plt.xlabel("Variable Importance Level")
train["Survived"].value_counts().plot.barh();
train["Survived"].value_counts()
models = [

    knn_tuned,

    loj_model,

    nb_model,

    cart_tuned,

    rf_tuned,

    gbm_tuned,

   

    

    

]





for model in models:

    names = model.__class__.__name__

    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("-"*28)

    print(names + ":" )

    print("Accuracy: {:.4%}".format(accuracy))
result = []



results = pd.DataFrame(columns= ["Models","Accuracy"])



for model in models:

    names = model.__class__.__name__

    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)    

    result = pd.DataFrame([[names, accuracy*100]], columns= ["Models","Accuracy"])

    results = results.append(result)

    

    

sns.barplot(x= "Accuracy", y = "Models", data=results, color="green")

plt.xlabel("Accuracy %")

plt.title("Accuracy Rate of Models");    
#set ids as PassengerId and predict survival 

ids = test["PassengerId"]

predictions = gbm_tuned.predict(test.drop("PassengerId", axis=1))



#set the output as a dataframe and convert to csv file named submission.csv

output = pd.DataFrame({ "PassengerId" : ids, "Survived": predictions })

output.to_csv("submission.csv", index=False)