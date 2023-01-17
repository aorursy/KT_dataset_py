# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # data visualization

plt.style.use("bmh")

import seaborn as sns #Visualisation also 

from collections import Counter

import warnings 

warnings.filterwarnings("ignore")  #To ignore mistakes on the code (not necessary)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



#To see different kind of visualisation styles 

#plt.style.available
#To see different kind of visualisation styles 

#plt.style.available
train_df=pd.read_csv("../input/titanic/train.csv")

test_df=pd.read_csv("../input/titanic/test.csv")

test_Id=test_df["PassengerId"]
#Lets see how many columns and attributes are there

train_df.columns
#We can see the first rows of the columns with head .Just to see what information they contain

train_df.head()
#For more specific info we use describe to obtain statistical information such as mean ,std,max etc...

train_df.describe()
# info about columns data category (types )is important for further applications

train_df.info()
def bar_plot(variable):



    """

    input:variable 

    output:bar plot & value count 

    

    """

    #get the feature

    var = train_df[variable]

    #count number of categorical variable (value/sample)

    varValue =var.value_counts()

    #visualize

    plt.figure(figsize=(9,3))

    plt.bar(varValue.index, varValue)

    plt.xticks(varValue.index, varValue.index.values)

    plt.ylabel("Frequency")

    plt.title(variable)

    plt.show()

    print("{}: \n {}".format(variable,varValue))





   

      
category1 = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch',   'Embarked']



for c in category1:

    bar_plot(c)
category2 =["Cabin","Name","Ticket"]



for c in category2:

    print("{} \n".format(train_df[c].value_counts()))
def plot_hist(variable):

    plt.figure(figsize = (9,3))

    plt.hist(train_df[variable],bins =50) #Frekansı değiştirir (bins)

    plt.xlabel(variable)

    plt.ylabel("Frequency")

    plt.title("{} distribution with hist".format(variable))

    plt.show()
numericVar = ["Fare","Age","PassengerId"]

for n in numericVar:

    plot_hist(n)
#Is there any correlation about class and survived ratio 

# Pclass vs Survived

#sort ascending

train_df[["Pclass","Survived"]].groupby(["Pclass"],as_index = False).mean().sort_values(by="Survived",ascending = False)
##Is there any correlation about sex and survived ratio

#Sex vs Survived

train_df[["Sex","Survived"]].groupby(["Sex"],as_index = False).mean().sort_values(by="Survived",ascending = False)
##Is there any correlation about sibling and survived ratio

#Sib vs Survived

train_df[["SibSp","Survived"]].groupby(["SibSp"],as_index = False).mean().sort_values(by="Survived",ascending = False)
##Is there any correlation about parch and survived ratio

#Parch vs Survived

train_df[["Parch","Survived"]].groupby(["Parch"],as_index = False).mean().sort_values(by="Survived",ascending = False)
def detect_outliers(df,features):

    outlier_indices=[]

    for c in features:

        #1st quartile

        Q1 =np.percentile(df[c],25)

        #3rd quartile

        Q3 =np.percentile(df[c],75)

        #IQR

        IQR =Q3-Q1

        #Outlier step

        outlier_step =IQR * 1.5

        #Detecting outliers index

        outlier_list_col =df[(df[c] < Q1 -outlier_step) | (df[c] > Q3  + outlier_step)].index

        #store indeces

        outlier_indices.extend(outlier_list_col)



        

    outlier_indices=Counter(outlier_indices)

    multiple_outliers = list(i for i,v in outlier_indices.items() if v > 2)

    

    return multiple_outliers
train_df.loc[detect_outliers(train_df,["Age","SibSp","Parch","Fare"])]
#drop outliers

train_df =train_df.drop(detect_outliers(train_df,["Age","SibSp","Parch","Fare"]),axis = 0).reset_index(drop = True)
#For control about running it over once

train_df_len =len(train_df)



train_df = pd.concat([train_df,test_df],axis= 0).reset_index(drop=True)

train_df.columns[train_df.isnull().any()]
train_df.isnull().sum()
train_df[train_df["Embarked"].isnull()]
train_df.boxplot(column = "Fare",by ="Embarked")

plt.show()
train_df["Embarked"] =train_df["Embarked"].fillna("C")

train_df[train_df["Embarked"].isnull()]
train_df[train_df["Fare"].isnull()]
train_df["Fare"]=train_df["Fare"].fillna(np.mean(train_df[train_df["Pclass"] == 3]["Fare"]))
list1 =[ "SibSp","Parch","Age","Fare","Survived"]

sns.heatmap(train_df[list1].corr(),annot = True ,fmt= ".2f")

plt.show()

g = sns.factorplot(x = "SibSp", y= "Survived",data =train_df,kind ="bar",size = 9)

g.set_ylabels("Survived Probability")

plt.show()
g = sns.factorplot(x = "Parch", y= "Survived",data =train_df,kind ="bar",size = 9)

g.set_ylabels("Survived Probability")

plt.show()
g = sns.factorplot(x = "Pclass", y= "Survived",data =train_df,kind ="bar",size = 9)

g.set_ylabels("Survived Probability")

plt.show()
g = sns.FacetGrid(train_df, col="Survived")

g.map(sns.distplot, "Age" , bins = 25)

plt.show()
g= sns.FacetGrid(train_df,col = "Survived" ,row ="Pclass")

g.map(plt.hist,"Age",bins = 25)

g.add_legend()

plt.show()
g = sns.FacetGrid(train_df , row = "Embarked", size = 3)

g.map(sns.pointplot,"Pclass","Survived" ,"Sex")

g.add_legend()

plt.show()
g = sns.FacetGrid(train_df, row = "Embarked",col = "Survived")

g.map(sns.barplot,"Sex","Fare")

g.add_legend()

plt.show()
train_df[train_df["Age"].isnull()]
sns.factorplot( x="Sex",y="Age",data=train_df, kind ="box")

plt.show()
sns.factorplot( x="Sex",y="Age",hue ="Pclass",data=train_df, kind ="box")

plt.show()
sns.factorplot( x="Parch",y="Age",data=train_df, kind ="box")

sns.factorplot( x="SibSp",y="Age",data=train_df, kind ="box")

plt.show()
#We need to use heatmap for correlation but in order to see the gender feature in heatmap.

#Need to make sex  the feature numerical.
train_df["Sex"]=[1 if i == "male" else 0 for i in train_df["Sex"]]

sns.heatmap(train_df[["Age","Sex","SibSp","Parch","Pclass"]].corr(),annot =True)

plt.show()
index_nan_age = list(train_df["Age"][train_df["Age"].isnull()].index)

for i in index_nan_age:

    age_pred = train_df["Age"][((train_df["SibSp"] == train_df.iloc[i]["SibSp"])&(train_df["Parch"] == train_df.iloc[i]["Parch"])&(train_df["Pclass"] == train_df.iloc[i]["Pclass"]))].median()

    age_med = train_df["Age"].median()

    if  not np.isnan(age_pred):

        train_df["Age"].iloc[i] = age_pred

    else:

        train_df["Age"].iloc[i]=age_med
train_df[train_df["Age"].isnull()]
train_df["Name"].head(10)
#noktaya göre ayır 

name =train_df["Name"]

train_df["Title"] = [i.split(".")[0].split(",")[-1].strip() for i in name]
sns.countplot(x="Title",data = train_df)

plt.xticks(rotation =60)

plt.show()
# convert to categorical 

train_df["Title"] = train_df["Title"].replace(["Lady","the Countess","Capt","Col","Don","Dr","Major","Rev","Sir","Jonkheer","Dona"],"other")

train_df["Title"] =[0 if i== "Master" else 1 if i =="Miss" or i =="Ms" or i== "Mlle" or i == "Mrs" else 2 if i == "Mr" else 3 for i in train_df["Title"]]
g = sns.factorplot(x = "Title",y= "Survived",data= train_df,kind="bar")

g.set_xticklabels(["Master","Mrs","Mr","other"])

g.set_ylabels("Survival Probability")

plt.show()
train_df.drop(labels = ["Name"],axis = 1,inplace=True)

train_df.head()
train_df = pd.get_dummies(train_df,columns=["Title"])

train_df.head()
train_df["Fsize"] = train_df["SibSp"] + train_df ["Parch"] +1
g = sns.factorplot(x = "Fsize", y = "Survived" ,data =train_df, kind ="bar")

g.set_ylabels("Survival")

plt.show()
train_df["family_size"] = [1 if i<5 else 0 for i in train_df ["Fsize"]]
sns.countplot(x="family_size",data=train_df)

plt.show()
g = sns.factorplot(x = "family_size", y = "Survived" ,data =train_df, kind ="bar")

g.set_ylabels("Survival")

plt.show()
train_df = pd.get_dummies(train_df, columns = ["family_size"])

train_df.head()
train_df["Embarked"].head()
sns.countplot(x="Embarked",data =train_df)
train_df = pd.get_dummies(train_df, columns =["Embarked"])

train_df.head()
tickets = []

for i in list(train_df.Ticket):

    if not i.isdigit():

        tickets.append(i.replace(".","").replace("/","").strip().split(" ")[0])

    else:

        tickets.append("x")

train_df["Ticket"] = tickets
#Prefix T changes the dummies colons from ticket to T

train_df = pd.get_dummies(train_df, columns = ["Ticket"], prefix = "T")

train_df.head(10)
sns.countplot(x="Pclass",data=train_df)

plt.show()
train_df["Pclass"] = train_df["Pclass"].astype("category")

train_df = pd.get_dummies(train_df, columns = ["Pclass"])

train_df.head()
train_df["Sex"] = train_df["Sex"].astype("category")

train_df = pd.get_dummies(train_df, columns = ["Sex"])

train_df.head()
train_df.drop(labels = ["PassengerId","Cabin"], axis = 1,inplace = True)
from sklearn.model_selection import train_test_split,StratifiedKFold,GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier,VotingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score


test = train_df[train_df_len:]

test.drop(labels = ["Survived"],axis=1 ,inplace=True)
train = train_df[:train_df_len]

X_train = train.drop(labels ="Survived",axis =1)

y_train = train["Survived"]

X_train,X_test,y_train,y_test =train_test_split(X_train,y_train,test_size =0.33,random_state =42)

print("x_train",len(X_train))

print("y_train",len(y_train))

print("x_test",len(X_test))

print("y_test",len(y_test))

print("test",len(test))
logreg =LogisticRegression()

logreg.fit(X_train, y_train)

acc_log_train =round(logreg.score(X_train,y_train)*100,2)

acc_log_test =round(logreg.score(X_test,y_test)*100,2)

print("Training Accuracy : % {}".format(acc_log_train))

print("Test Accuracy : % {}".format(acc_log_test))
random_state=42

classifier= [DecisionTreeClassifier(random_state = random_state),

            SVC(random_state = random_state),

            RandomForestClassifier( random_state = random_state),

            LogisticRegression(random_state = random_state),

            KNeighborsClassifier()]

dt_param_grid = {"min_samples_split" : range(10,500,20),

                    "max_depth" : range(1,20,2)}

svm_param_grid = {"kernel":["rbf"],

                 "gamma":[0.001,0.01,0.1,1],

                 "C" :[1,10,50,100,200,300,1000]}

rf_param_grid ={"max_features":[1,3,10],

               "min_samples_split":[2,3,10],

               "min_samples_leaf":[1,3,10],

               "bootstrap":[False],

               "n_estimators":[100,300],

               "criterion":["gini"]}

logreg_param_grid = {"C":np.logspace(-3,3,7),

                    "penalty":["l1","l2"]}

knn_param_grid ={"n_neighbors":np.linspace(1,19,10,dtype=int).tolist(),

                "weights":["uniform","distance"],

                "metric":["eucledian","manhattan"]}

classifier_param =[dt_param_grid,

                   svm_param_grid,

                   rf_param_grid ,

                   logreg_param_grid,

                   knn_param_grid]
cv_result = []

best_estimators = []

for i in range(len(classifier)):

    clf = GridSearchCV(classifier[i], param_grid=classifier_param[i], cv = StratifiedKFold(n_splits = 10), scoring = "accuracy", n_jobs = -1,verbose = 1)

    clf.fit(X_train,y_train)

    cv_result.append(clf.best_score_)

    best_estimators.append(clf.best_estimator_)

    print(cv_result[i])
cv_results = pd.DataFrame({"Cross Validation Means":cv_result, "ML Models":["DecisionTreeClassifier",

                                                                            "SVM",

                                                                            "RandomForestClassifier",

                                                                            "LogisticRegression",

                                                                            "KNeighborsClassifier"]})



g = sns.barplot("Cross Validation Means", "ML Models", data = cv_results)

g.set_xlabel("Mean Accuracy")

g.set_title("Cross Validation Scores")
votingC = VotingClassifier(estimators = [("dt",best_estimators[0]),

                                        ("rfc",best_estimators[2]),

                                        ("lr",best_estimators[3])],

                                        voting = "soft", n_jobs = -1)

votingC = votingC.fit(X_train, y_train)

print(accuracy_score(votingC.predict(X_test),y_test))



test_survived =pd.Series(votingC.predict(test),name="Survived").astype(int)

results = pd.concat([test_Id,test_survived],axis = 1)

results.to_csv("titanic.csv", index = False)