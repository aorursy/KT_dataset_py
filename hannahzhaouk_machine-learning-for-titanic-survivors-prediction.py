import numpy as np
import pandas as pd
import seaborn as sns

%matplotlib inline 
from matplotlib import pyplot as plt
sns.set_style("whitegrid")
training=pd.read_csv("../input/train.csv")
testing=pd.read_csv("../input/test.csv")
training.head()
training.describe()
training.dtypes
print(training.keys())
print(testing.keys())
def NaN_table(training, testing):
    print("Training Data Condition")
    print(pd.isnull(training).sum())
    print(" ")
    print("Testing Data Condition")
    print(pd.isnull(testing).sum())

NaN_table(training, testing)
training.drop(labels=["Cabin"],axis=1,inplace=True)
testing.drop(labels=["Cabin"],axis=1,inplace=True)
NaN_table(training, testing)
agedis=training.copy()
agedis.dropna(inplace=True)
sns.distplot(agedis["Age"])
training["Age"].fillna(training["Age"].median(),inplace=True)
testing["Age"].fillna(testing["Age"].median(),inplace=True)
training["Embarked"].fillna("S",inplace=True)
testing["Fare"].fillna(testing["Fare"].median(),inplace=True)
NaN_table(training, testing)
sns.countplot(x="Survived",data=training, hue="Sex")
plt.title("Distribution of Survival Proportion based on Gender")
plt.show()

female_survival=training[training.Sex=="female"]["Survived"].sum()
male_survival=training[training.Sex=="male"]["Survived"].sum()
total_passengers=training["Survived"].count()
total_survived=training["Survived"].sum()


print("Total number of survivals:", total_survived)
print("Survival rate:"+"{:.2%}".format(total_survived/total_passengers))
print(" ")
print("Female survived:", female_survival)
print("Male survived:", male_survival)
print("Percentage of female survived:"+"{:.2%}".format(female_survival/total_survived))
print("Percentage of male survived:"+"{:.2%}".format(male_survival/total_survived))
sns.barplot(x="Pclass", y="Survived", data=training)
plt.title("Distribution of Survivals based on Class")
plt.ylabel("Survival Rate")
plt.xlabel("Class")
plt.show()
sns.barplot(x="Pclass",y="Survived", data=training, hue="Sex")
plt.title("Survival Rate based on Class and Gender")
plt.ylabel("Survival Rate")
plt.xlabel("Class")
sns.barplot(x="Sex", y="Survived", data=training, hue="Pclass")
plt.title("Survival Rate based on Class and Gender")
plt.ylabel("Gender")
plt.xlabel("Class")
survived_age = training[training.Survived == 1]["Age"]
not_survived_age=training[training.Survived == 0]["Age"]
plt.subplot(1,2,1)
sns.distplot(survived_age,kde=False)
plt.axis([0,80,0,100])
plt.title("Survived")
plt.ylabel("Proportion")
plt.subplot(1,2,2)
sns.distplot(not_survived_age,kde=False)
plt.axis([0,80,0,100])
plt.title("Didn't Survive")
plt.subplots_adjust(right=1.7)
plt.show()
training.sample(5)
training.drop(labels=["PassengerId"], axis=1,inplace=True)
training.drop(labels=["Ticket"], axis=1, inplace=True)
training.sample(5)
testing.drop(["PassengerId"],axis=1,inplace=True)
testing.drop(["Ticket"],axis=1,inplace=True)
testing.sample(5)
training.loc[training["Sex"]=="male", "Sex"] = 0
training.loc[training["Sex"]=="female","Sex"] = 1

training.loc[training["Embarked"]=="S", "Embarked"] = 0
training.loc[training["Embarked"]=="C", "Embarked"] = 1
training.loc[training["Embarked"]=="Q", "Embarked"] = 2

testing.loc[testing["Sex"]=="male", "Sex"] = 0
testing.loc[testing["Sex"]=="female", "Sex"] = 1

testing.loc[testing["Embarked"]=="S", "Embarked"] = 0
testing.loc[testing["Embarked"]=="C", "Embarked"] = 1
testing.loc[testing["Embarked"]=="Q", "Embarked"] = 2
training.sample(5)
training["famsize"]= training["SibSp"] + training["Parch"]+1
testing["famsize"]= testing["SibSp"] + testing["Parch"]+1
training["IsAlone"]=training.famsize.apply(lambda x: 1 if x==1 else 0)
testing["IsAlone"]=testing.famsize.apply(lambda x: 1 if x==1 else 0)
for name in training["Name"]:
    training["Title"]= training["Name"].str.extract("([A-Za-z]+)\.",expand=True)

for name in testing["Name"]:
    testing["Title"]= testing["Name"].str.extract("([A-Za-z]+)\.",expand=True)

titles=set(training["Title"])
print(titles)
title_list=list(training["Title"])
frequency_titles = []

for l in titles:
    frequency_titles.append(title_list.count(l))

print(frequency_titles)
title=list(titles)
title_frame=pd.DataFrame({"Title":title,
                          "Frequency":frequency_titles})

title_frame=title_frame.loc[:,["Title","Frequency"]]
print(title_frame)
replace_title={"Don":"Other","Jonkheer":"Other","Ms":"Other", "Col":"Other", "Lady":"Other", "Major":"Other", "Mlle":"Other", "Sir":"Other", "Mme":"Other", "Countess":"Other", "Capt":"Other"}
training.replace({"Title":replace_title}, inplace=True)
testing.replace({"Title":replace_title},inplace=True)
training.loc[training["Title"]=="Miss","Title"]=0
training.loc[training["Title"]=="Mr","Title"]=1
training.loc[training["Title"]=="Mrs","Title"]=2
training.loc[training["Title"]=="Master","Title"]=3
training.loc[training["Title"]=="Dr","Title"]=4
training.loc[training["Title"]=="Rev","Title"]=5
training.loc[training["Title"]=="Other","Title"]=6

testing.loc[testing["Title"]=="Miss","Title"]=0
testing.loc[testing["Title"]=="Mr","Title"]=1
testing.loc[testing["Title"]=="Mrs","Title"]=2
testing.loc[testing["Title"]=="Master","Title"]=3
testing.loc[testing["Title"]=="Dr","Title"]=4
testing.loc[testing["Title"]=="Rev","Title"]=5
testing.loc[testing["Title"]=="Other","Title"]=6
training.sample(5)
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, accuracy_score
features= ["Pclass", "Sex", "Age", "Fare", "Embarked", "famsize", "IsAlone", "Title"]
x_train=training[features] #define training features
y_train=training["Survived"] #define predicted label in training set

x_testing=testing[features] #define testing features set

#y_testing is the term that we're trying to predict with the models
from sklearn.model_selection import train_test_split

x_training, x_valid, y_training, y_valid= train_test_split(x_train,y_train,test_size=0.2,random_state=0)
    
#Create x_valid, y_valid as validation data set
#SVC Model

svc_clf=SVC().fit(x_training, y_training)
predict_svc=svc_clf.predict(x_valid)
accuracy_svc=accuracy_score(y_valid,predict_svc)

print(accuracy_svc)
#LinearSVC Model

linsvc_clf=LinearSVC().fit(x_training, y_training)
predict_linsvc=linsvc_clf.predict(x_valid)
accuracy_linsvc=accuracy_score(y_valid, predict_linsvc)

print(accuracy_linsvc)
#RandomForest Model

RF_clf=RandomForestClassifier().fit(x_training, y_training)
predict_RF=RF_clf.predict(x_valid)
accuracy_RF=accuracy_score(y_valid, predict_RF)

print(accuracy_RF)
#LogisticRegression Model

logreg_clf=LogisticRegression().fit(x_training, y_training)
predict_logreg=logreg_clf.predict(x_valid)
accuracy_logreg=accuracy_score(y_valid, predict_logreg)

print(accuracy_logreg)
#KNN Model
knn_clf=KNeighborsClassifier().fit(x_training, y_training)
predict_knn=knn_clf.predict(x_valid)
accuracy_knn=accuracy_score(y_valid, predict_knn)

print(accuracy_knn)
#GaussianNB Model

gnb_clf= GaussianNB().fit(x_training, y_training)
predict_gnb=gnb_clf.predict(x_valid)
accuracy_gnb=accuracy_score(y_valid, predict_gnb)

print(accuracy_gnb)
#DecisionTree Model

dt_clf=DecisionTreeClassifier().fit(x_training, y_training)
predict_dt=dt_clf.predict(x_valid)
accuracy_dt=accuracy_score(y_valid, predict_dt)

print(accuracy_dt)
model_perform=pd.DataFrame({
    "Model": ["SVC", "Linear SVC", "Random Forest", "Logistic Regression", "K-Nearest-Neighbors", "Gaussian Naive Bayes", "Decision Tree"],
    "Accuracy":[accuracy_svc, accuracy_linsvc, accuracy_RF, accuracy_logreg, accuracy_knn, accuracy_gnb, accuracy_dt]
})

model_perform=model_perform.loc[:,["Model","Accuracy"]]
model_perform.sort_values(by="Accuracy", ascending=False)

from sklearn.model_selection import GridSearchCV
rf_clf=RandomForestClassifier()
n_range=range(4,15)
param_grid={"n_estimators":n_range,
            "criterion":["gini","entropy"],
            "max_features":["auto", "sqrt","log2"],
            "max_depth":[2,3,5,10],
            "min_samples_split":[2,3,4,5,10]            
           }

grid_cv= GridSearchCV(rf_clf, param_grid, scoring=make_scorer(accuracy_score))
grid_cv=grid_cv.fit(x_train, y_train)

print("Optimized Random Forest Model:",
     grid_cv.best_estimator_)
rf_clf=grid_cv.best_estimator_

rf_clf.fit(x_train, y_train)
#Here is a ValueError that "could not convert string to float: 'Dona'".
#Let's look back at the "Title" feature of testing set
set(testing["Title"])
#So we have to encode "Dona" to number 6 as "Other" category.
replace_dona={"Dona":"Other"}
testing.replace({"Title":replace_dona}, inplace=True)
testing.loc[testing["Title"]=="Other","Title"]=6
set(testing["Title"])
x_testing=testing[features]
submission_predictions = rf_clf.predict(x_testing)
submission=pd.DataFrame({
    "Name":testing["Name"],
    "Survived":submission_predictions
})
submission.to_csv("titanic_prediction.csv", index=False)
print(submission.shape)
