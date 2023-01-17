# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import seaborn as sns

from collections import Counter



import warnings

warnings.filterwarnings("ignore")



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
plt.style.available
a =[1,2,3,4]

plt.plot(a)

plt.show()
train_df = pd.read_csv("/kaggle/input/titanic/train.csv")

test_df = pd.read_csv("/kaggle/input/titanic/test.csv")

test_PassengerId = test_df["PassengerId"]
train_df.columns
train_df.head()
train_df.describe()
train_df.info()
def bar_plot(variable):

    """

        input: variable ex: "Sex"

        output: bar plot & value count

    """

    # get feature 

    var = train_df[variable]

    # count number of categorical variable(value/sample)

    varValue = var.value_counts()

    

    

    #visualize

    plt.figure(figsize=(9,3))

    plt.bar(varValue.index, varValue)

    plt.xticks(varValue.index, varValue.index.values)

    plt.ylabel("Frequency")

    plt.title(variable)

    plt.show()

    print("{}: \n {}".format(variable,  varValue))
category1 = ["Survived", "Sex", "Pclass", "Embarked", "SibSp", "Parch"] #variable olarak goruyor.

for c in category1:

    bar_plot(c)
# we seperated that part because we can confuse to name,cabin and ticket number these are special and we dont understand in vizualize tool. We just run counts.

category2 = ["Cabin", "Name", "Ticket"]

for c in category2:

    print("{} \n".format(train_df[c].value_counts()))
def plot_hist(variable): # bu degiskenleri for i in metodunda fare, age ve passengerId dondurerek degerleri alıp yazmamızı saglıyor (variable)

    plt.figure(figsize = (9,3))

    plt.hist(train_df[variable], bins = 50) #default 10 dur sıklık.

    plt.xlabel(variable)

    plt.ylabel("Frequency")

    plt.title("{} distribution with hist".format(variable))

    plt.show()
numericVar = ["Fare", "Age", "PassengerId"] # python bu kısmı variable olarak goruyor esitlememize gerek kalmıyor. num variable gibi categorical da ayni sekilde.

for n in numericVar: #feature larım plotta yazdırılıyor.

    plot_hist(n)
# Pclass vs Survived

train_df[["Pclass", "Survived"]].groupby(["Pclass"], as_index = False).mean().sort_values(by="Survived",ascending= False) #grupla ve ortalamasını goster. Hayatta kalma oranımız. Sınıflara gore baktığımızda 1. sınıfta olanlar hayatta kalma oranı daha fazla.
# Sex vs Survived

train_df[["Sex", "Survived"]].groupby(["Sex"], as_index = False).mean().sort_values(by="Survived",ascending= False) # prediction 1. sınıf ve kadınsa hayatta kalma olasılığı çok yüksektir diyebiliriz. ML'imiz de buna bakacaktır.
# Sibsp vs Survived

train_df[["SibSp", "Survived"]].groupby(["SibSp"], as_index = False).mean().sort_values(by="Survived",ascending= False) #2 den fazla kisi olursa hayatta kalma oranı dusuyor. 
# Parch vs Survived #groupby buna gore grupla dmeek

train_df[["Parch", "Survived"]].groupby(["Parch"], as_index = False).mean().sort_values(by="Survived",ascending= False) #anlamamız gereken iliskiler ve sınıflandırmalar icin sınıflandırmam gereklidir. Diger feature lar icinde siz yapabilirsiniz bakabilir ve yorumlayabilirsiniz.
#df.c() #hangi feature ın 
def detect_outliers(df,features): #icersine outlier ve feature larımızı alacak.

    outlier_indices = []

    

    for c in features:

        #1st quartile

        Q1 = np.percentile(df[c],25)

        #3rd quartile

        Q3 = np.percentile(df[c],75)

        #IQR

        IQR = Q3-Q1

        #outlier step

        outlier_step = IQR * 1.5

        #detect outlier and their indeces

        outlier_list_col = df[(df[c] < Q1 - outlier_step)  | (df[c] > Q3 + outlier_step )].index

        #store indeces

        outlier_indices.extend(outlier_list_col)#outlier list column ları topluyorum.

        

    outlier_indices = Counter(outlier_indices)

    multiple_outliers = list(i for i, v in outlier_indices.items() if v>2 )  #outlier lar 2 den fazlaysa çıkart. 1 taneyse çıkartmama gerek yok demek

    

    return multiple_outliers
train_df.loc[detect_outliers(train_df,["Age","SibSp","Parch","Fare" ])]
a = ["a","a","a","a","b","b"]

Counter(a) #ornek
#drop outliers

train_df = train_df.drop(detect_outliers(train_df,["Age","SibSp","Parch","Fare" ]), axis = 0).reset_index(drop = True)
test_df.head(10)
train_df_len= len(train_df) #train_df in ilk 998 boyutuda kaydetmek istemiyorum bunun icinde tutucam. 3-4 kere sürekli run etmemek lazım. Tüm variable lar reset section a basıca. Yormamak için.

train_df = pd.concat ([train_df, test_df], axis = 0 ).reset_index(drop = True) #yukarıdan asagıya kaydediyorum.
train_df.head()
train_df.columns[train_df.isnull().any()] #train_df missing value var mı bakıcaz. Hangi feature da olduğunu göreceğiz. Survived normal, 
train_df.isnull().sum() #missing value bul ve topla
train_df[train_df["Embarked"].isnull()]
train_df.boxplot(column ="Fare",by ="Embarked")

plt.show()
train_df["Embarked"] = train_df["Embarked"].fillna("C")

train_df[train_df["Embarked"].isnull()]
train_df[train_df["Fare"].isnull()]
np.mean(train_df[train_df["Pclass"] == 3]["Fare"])
train_df[train_df["Pclass"] == 3]["Fare"]
train_df["Fare"]= train_df["Fare"].fillna(np.mean(train_df[train_df["Pclass"] == 3]["Fare"]))
train_df[train_df["Fare"].isnull()]
list1 = ["SibSp","Parch","Age","Fare", "Survived"]

sns.heatmap(train_df[list1].corr(), annot= True, fmt= ".2f") # if false we dont see number.

plt.show()
g = sns.factorplot( x= "SibSp", y= "Survived", data = train_df, kind = "bar", size=6)

g.set_ylabels("Survived Probability")

plt.show
g= sns.factorplot(x="Parch", y= "Survived", kind= "bar", data=train_df, size=6)

g.set_ylabels("Survived Probability")

plt.show()



#black lines are mid of ratio its changable
g= sns.factorplot(x= "Pclass", y="Survived", data = train_df, kind="bar", size=6)

g.set_ylabels=("Survived Probability")

plt.show()
g= sns.FacetGrid(train_df, col="Survived")

g.map(sns.distplot, "Age", bins=25)

plt.show()
g = sns.FacetGrid(train_df, col="Survived", row="Pclass", size=2)

g.map(plt.hist, "Age", bins=25)

g.add_legend()

plt.show()
g = sns.FacetGrid(train_df, row= "Embarked",size=2)

g.map(sns.pointplot, "Pclass", "Survived", "Sex")

g.add_legend()

plt.show()
g = sns.FacetGrid(train_df, row = "Embarked", col= "Survived", size=2.3)

g.map(sns.barplot, "Sex", "Fare")

g.add_legend()

plt.show()
train_df[train_df["Age"].isnull()] # two train_df dont show me false values. Do not forget that. You can show just true what we need (isnull= mean ) we need totally 256

sns.factorplot(x= "Sex", y= "Age", data=train_df, kind="box")

plt.show()
sns.factorplot(x= "Sex", y= "Age", hue= "Pclass", data=train_df, kind="box")

plt.show()
sns.factorplot(x= "Parch", y= "Age", data=train_df, kind="box")

sns.factorplot(x= "SibSp", y= "Age", data=train_df, kind="box")

plt.show()
train_df["Sex"] = [1 if i == "male" else 0 for i in train_df["Sex"]] # sex is object(str) male and female. We change 1 and 0 like survive rate
sns.heatmap(train_df[["Age", "Sex", "SibSp", "Parch", "Pclass"]].corr(), annot=True)

plt.show()
index_nan_age = list(train_df["Age"][train_df["Age"].isnull()].index)

for i in index_nan_age:

    age_pred = train_df["Age"][((train_df["SibSp"]== train_df.iloc[i]["SibSp"]) & (train_df["Parch"]== train_df.iloc[i]["Parch"]) & (train_df["Pclass"]== train_df.iloc[i]["Pclass"]))].median()

    age_med = train_df["Age"].median()

    if not np.isnan(age_pred):

        train_df["Age"].iloc[i]= age_pred

    else:

        train_df["Age"].iloc[i]= age_med  

        

# Burada anladığım; Age in isnull kısımlarını gez, örneğin Age'in 500. indexi boş. O değere ait; SibSp, ParCh,Pclass larının 500. indexindeki değerlerin hepsini bul. 

# Bu değerlere eşit indexleri bul ve age lerini kaydet. Bunlarında median larını al. Demek
train_df["Name"].head(10)
name = train_df["Name"]

train_df["Title"] = [i.split(".")[0].split(",")[-1].strip() for i in name]

# this method is seperate one name to two parts. Because we can see Mr. and Mrs. Master. Etc. so we can predict title and survive corr.

#another step is Johnson, Mrs. we choose 0. index. But we need just title. Thats why we split , and we take last index (title) I merge two methods in one...

train_df["Title"].head(10)
train_df["Title"].head(10)
# I want to see all title one visualize tool

sns.countplot(x="Title", data=train_df)

plt.xticks(rotation =45)

plt.show()
# convert to categorical

train_df["Title"] = train_df["Title"].replace(["Lady","the Countess","Capt","Col","Don","Dr","Major","Rev","Sir","Jonkheer","Dona"],"other")

train_df["Title"] = [0 if i == "Master" else 1 if i == "Miss" or i == "Ms" or i == "Mlle" or i == "Mrs" else 2 if i == "Mr" else 3 for i in train_df["Title"]]

train_df["Title"].head(20)
sns.countplot(x="Title", data = train_df)

plt.xticks(rotation = 45)

plt.show()
g = sns.factorplot(x = "Title", y= "Survived", data=train_df, kind= "bar")

g.set_xticklabels(["Master","Mrs","Mr","Other"])

g.set_ylabels("Survival Probability")

plt.show()
train_df.drop(labels = ["Name"], axis = 1, inplace = True)
train_df.head()
train_df = pd.get_dummies(train_df,columns =["Title"])

train_df.head()
train_df["Fsize"] = train_df["SibSp"] + train_df["Parch"] + 1
train_df.head()
g = sns.factorplot( x= "Fsize", y= "Survived", data= train_df, kind = "bar")

g.set_ylabels("Survival")

plt.show()
#We decided threshold 5.

train_df["family_size"] = [1 if i<5 else 0 for i in train_df["Fsize"]]
sns.countplot(x= "family_size", data=train_df)

plt.show()
g = sns.factorplot(x= "family_size", y= "Survived", data= train_df, kind = "bar")

g.set_ylabels("Survival")

plt.show()
train_df = pd.get_dummies(train_df, columns = ["family_size"])

train_df.head()
train_df["Embarked"].head()
sns.countplot(x="Embarked", data= train_df)

plt.show()
train_df = pd.get_dummies(train_df, columns = ["Embarked"])

train_df.head()
train_df["Ticket"].head(15)
a = "A/5. 2151"

a.replace(".","").replace("/","").strip().split(" ")[0]
tickets = []

for i in list(train_df.Ticket):

    if not i.isdigit():

        tickets.append(i.replace(".","").replace("/","").strip().split(" ")[0])

    else:

        tickets.append("x")

train_df["Ticket"] = tickets
train_df["Ticket"].head(20)
train_df.head()
train_df = pd.get_dummies(train_df, columns= ["Ticket"], prefix = "T")

train_df.head(10)

sns.countplot(x= "Pclass", data= train_df)

plt.show()
train_df["Pclass"] = train_df["Pclass"].astype("category")

train_df = pd.get_dummies(train_df, columns= ["Pclass"])

train_df.head()
train_df["Sex"] = train_df["Sex"].astype("category")

train_df = pd.get_dummies(train_df, columns=["Sex"])

train_df.head()
train_df.drop(labels = ["PassengerId", "Cabin"], axis = 1, inplace = True)
train_df.columns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
train_df_len
test = train_df[train_df_len:]

test.drop(labels = ["Survived"],axis = 1, inplace = True)
train = train_df[:train_df_len]

X_train = train.drop(labels = "Survived", axis = 1)

y_train = train["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.33, random_state = 42)

print("X_train",len(X_train))

print("X_test",len(X_test))

print("y_train",len(y_train))

print("y_test",len(y_test))

print("test",len(test))
logreg = LogisticRegression()

logreg.fit(X_train, y_train)

acc_log_train = round(logreg.score(X_train, y_train)*100,2) 

acc_log_test = round(logreg.score(X_test,y_test)*100,2)

print("Training Accuracy: % {}".format(acc_log_train))

print("Testing Accuracy: % {}".format(acc_log_test))
random_state = 42

classifier = [DecisionTreeClassifier(random_state = random_state),

             SVC(random_state = random_state),

             RandomForestClassifier(random_state = random_state),

             LogisticRegression(random_state = random_state),

             KNeighborsClassifier()]



dt_param_grid = {"min_samples_split" : range(10,500,20),

                "max_depth": range(1,20,2)}



svc_param_grid = {"kernel" : ["rbf"],

                 "gamma": [0.001, 0.01, 0.1, 1],

                 "C": [1,10,50,100,200,300,1000]}



rf_param_grid = {"max_features": [1,3,10],

                "min_samples_split":[2,3,10],

                "min_samples_leaf":[1,3,10],

                "bootstrap":[False],

                "n_estimators":[100,300],

                "criterion":["gini"]}



logreg_param_grid = {"C":np.logspace(-3,3,7),

                    "penalty": ["l1","l2"]}



knn_param_grid = {"n_neighbors": np.linspace(1,19,10, dtype = int).tolist(),

                 "weights": ["uniform","distance"],

                 "metric":["euclidean","manhattan"]}

classifier_param = [dt_param_grid,

                   svc_param_grid,

                   rf_param_grid,

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
cv_results = pd.DataFrame({"Cross Validation Means":cv_result, "ML Models":["DecisionTreeClassifier", "SVM","RandomForestClassifier",

             "LogisticRegression",

             "KNeighborsClassifier"]})

#We took results in cv_result parameters. We must add ML models with step by step if we mixed some values name, we didnt understand or we didnt predict logical



g = sns.barplot("Cross Validation Means", "ML Models", data = cv_results)

g.set_xlabel("Mean Accuracy")

g.set_title("Cross Validation Scores")
votingC = VotingClassifier(estimators = [("dt",best_estimators[0]),

                                        ("rfc",best_estimators[2]),

                                        ("lr",best_estimators[3])],

                                        voting = "soft", n_jobs = -1)

votingC = votingC.fit(X_train, y_train)

print(accuracy_score(votingC.predict(X_test),y_test))
test_survived = pd.Series(votingC.predict(test), name = "Survived").astype(int)

results = pd.concat([test_PassengerId, test_survived],axis = 1)

results.to_csv("titanic.csv", index = False)
