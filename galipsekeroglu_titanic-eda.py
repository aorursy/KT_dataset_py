# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

plt.style.use("ggplot")

#plt.style.available --> kullanılabilecek tüm tarzların gösterimi



import seaborn as sns

from collections import Counter

import warnings

warnings.filterwarnings("ignore")



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv("/kaggle/input/titanic/train.csv")

test_df = pd.read_csv("/kaggle/input/titanic/test.csv")

test_passengerId = test_df["PassengerId"]
train_df.columns
train_df.head()
train_df.describe().T
train_df.info()
def bar_plot(variable):

    var = train_df[variable]

    var_value = var.value_counts()

    plt.figure(figsize=(9,3))

    plt.bar(var_value.index,var_value)

    plt.xticks(var_value.index,var_value.index.values)

    plt.ylabel("Frequency")

    plt.title(variable)

    plt.show()

    print("{}: \n {}".format(variable,var_value))
category1 =["Survived","Sex","Pclass","Embarked","SibSp","Parch"]

for c in category1:

    bar_plot(c)
category2 =["Cabin","Name","Ticket"]

for c in category2:

    print("{} \n".format(train_df[c].value_counts))

    
def plot_hist(variable):

    plt.figure(figsize=(9,3))

    plt.hist(train_df[variable],bins=30)

    plt.xlabel(variable)

    plt.ylabel("Frequency")

    plt.title("{} distribution with histogram".format(variable))

    plt.show()
numericVar = ["Fare","Age","PassengerId"]

for n in numericVar:

    plot_hist(n)
#Pclass vs Survived

x = train_df[["Pclass","Survived"]].groupby(["Pclass"],as_index = False).mean().sort_values(by="Survived",ascending=False)

x.plot(x="Pclass",y="Survived",kind="barh")

plt.show()
#Sex vs Survived

train_df[["Sex","Survived"]].groupby(["Sex"],as_index = False).mean().sort_values(by="Survived",ascending=False)
#SibSp vs Survived

train_df[["SibSp","Survived"]].groupby(["SibSp"],as_index = False).mean().sort_values(by="Survived",ascending=False)
#Parch vs Survived

train_df[["Parch","Survived"]].groupby(["Parch"],as_index = False).mean().sort_values(by="Survived",ascending=False)
#Parch vs Survived

train_df[["Parch","SibSp","Survived"]].groupby(["Parch","SibSp"],as_index = False).mean().sort_values(by="Survived",ascending=False)
def detect_outliers(df,features):

    outlier_indices =[]

    for c in features:

        #1st quartile

        q1= np.percentile(df[c],25)

        #3rd quartile

        q3= np.percentile(df[c],75)

        #IQR

        iqr = q3-q1

        #Outlier step

        outlier_step = iqr*1.5

        #detect outlier and their indices

        outlier_list_col = df[(df[c] < q1 - outlier_step) | (df[c] > q3 + outlier_step )].index

        #store indices

        outlier_indices.extend(outlier_list_col)

    

    outlier_indices = Counter(outlier_indices) #hangi yolcu kaç tane outlier değer içeriyor

    multiple_outliers = list (i for i,v in outlier_indices.items() if v>2) #bir tane sample'ımda 2'den fazla outlier varsa çıkartmak için index'ini tutuyoruz.

    

    return multiple_outliers
train_df.loc[detect_outliers(train_df,["Age","SibSp","Parch","Fare"])] #Outlier değerler
#drop outliers

train_df = train_df.drop(detect_outliers(train_df,["Age","SibSp","Parch","Fare"]),axis=0).reset_index(drop=True)

train_df
train_df_len = len(train_df)

train_df = pd.concat([train_df,test_df],axis=0).reset_index(drop=True)
train_df.head()
train_df.columns[train_df.isnull().any()]
train_df.isnull().sum()
train_df[train_df["Embarked"].isnull()]
train_df.boxplot(column="Fare",by="Embarked")

plt.show()
train_df["Embarked"] = train_df["Embarked"].fillna("C")

#C'dekilerin medyan değeri 100'e daha yakın,o yüzden ordan binmiş olma ihtimalleri daha yüksek
train_df[train_df["Fare"].isnull()]
np.mean(train_df[train_df["Pclass"] ==3]["Fare"])
train_df["Fare"] = train_df["Fare"].fillna(np.mean(train_df[train_df["Pclass"] ==3]["Fare"]))

#3.sınıfa ait ve ödedikleri ortalama Fare değerini buradaki eksik değere atama işlemi yaptık.
list1 = ["SibSp","Parch","Age","Fare","Survived"]

sns.heatmap(train_df[list1].corr(),annot=True,fmt =".2f")

plt.show()

#annot=True korelasyon matrisi üzerindeki değerleri görebilmemizi sağlıyor
g = sns.factorplot(x = "SibSp" , y="Survived" , data = train_df, kind="bar", size=6)

g.set_ylabels("Survived Probability")

plt.show()

#Having a lot of SibSp have less chance to survive. If SibSp = 0 or 1 or, passenger has more chance to survive.

#We can consider a new feature describing these categories

#Black line represents the standart deviation.
g = sns.factorplot(x="Parch",y="Survived",data=train_df, kind="bar",size=6)

g.set_ylabels("Survived Probability")

plt.show()

#Black line represents the standart deviation. For instance, when Parch is 3, the person would survive in probability between 0.2 to 1

#Sibsp and parch can be used for creating new feature extraction with Th=3

#Small families have more chance to survive.

#There is a standart deviation in survival of passenger when ParCh=3
g = sns.factorplot(x="Pclass",y="Survived",data=train_df, kind="bar",size=6)

g.set_ylabels("Survived Probability")

plt.show()
g = sns.FacetGrid(train_df , col ="Survived")

g.map(sns.distplot,"Age",bins=25) #Compare with "Age" column

plt.show()

#On the second graph, we can clearly see that children were prior to be survived according to Age vs Survived
g = sns.FacetGrid(train_df,col ="Survived",row = "Pclass",size=3)

g.map(plt.hist, "Age",bins=25) #Compare with "Age" column

g.add_legend()

plt.show()
g = sns.FacetGrid(train_df,row = "Embarked",size=2)

g.map(sns.pointplot, "Pclass","Survived","Sex")

g.add_legend()

plt.show()
g = sns.FacetGrid(train_df,row = "Embarked",col="Survived",size=2.3)

g.map(sns.barplot,"Sex","Fare")

g.add_legend()

plt.show()
train_df[train_df["Age"].isnull()]
sns.factorplot(x="Sex",y="Age",data=train_df , kind = "box")

plt.show()

#Male ve Female için yaşlarda medyan değerleri hemen hemen aynı yani medyan ile yaş tahmini yapamam bu durumda

#Sex is not informative for age prediction, age distribution seems to be same.
sns.factorplot(x="Sex",y="Age",hue="Pclass",data=train_df , kind = "box")

plt.show()
sns.factorplot(x="Parch",y="Age",data=train_df , kind = "box")

sns.factorplot(x="SibSp",y="Age",data=train_df , kind = "box")

plt.show()
train_df["Sex"] = [1 if i =="male" else 0 for i in train_df["Sex"] ]
sns.heatmap(train_df[["Age","Sex","SibSp","Parch","Pclass"]].corr(),annot=True)

plt.show()

#Biz yukarıda yaş ve cinsiyet arasında bir ilişki söz konusu değil demiştik ve bu korelasyon tablosu bunu doğruluyor.

#Age is not correlated with sex but it is correlated with parch,sibsp and pclass
index_nan_age = list(train_df["Age"][train_df["Age"].isnull()].index)

for i in index_nan_age:

    age_prediction = train_df["Age"][((train_df["SibSp"] == train_df.iloc[i]["SibSp"]) & (train_df["Parch"] == train_df.iloc[i]["Parch"]) & (train_df["Pclass"] == train_df.iloc[i]["Pclass"]))].median()

    age_median = train_df["Age"].median()

    if not np.isnan(age_prediction): #age_prediction yaparken SibSp,Parch veya Pclass kolonu boş gelirse NaN dönecek

        train_df["Age"].iloc[i] = age_prediction

    else:

        train_df["Age"].iloc[i] = age_median

train_df[train_df["Age"].isnull()]
train_df["Name"].head(5)
name = train_df["Name"]

train_df["Title"] = [i.split(".")[0].split(",")[-1].strip() for i in name] 

#noktaya göre ayırınca elimizde sol ve sağ olarak 2 tane string dizisi kalacak. Strip ile en sonda kalan boşluk da atılıyor.

train_df["Title"]
sns.countplot(x="Title",data=train_df)

plt.xticks(rotation=60)

plt.show()
#Convert to categorical

train_df["Title"] = train_df["Title"].replace(["Lady","the Countess","Capt","Col","Don","Dr","Major","Rev","Sir","Jonkheer","Dona"],"Other")
train_df["Title"] = [0 if i =="Master" else 1 if i =="Miss" or i == "Ms" or i=="Mlle" or i =="Mrs" else 2 if i =="Mr" else 3 for i in train_df["Title"]]
sns.countplot(x="Title",data=train_df)

plt.xticks(rotation=60)

plt.show()
g = sns.factorplot(x="Title",y="Survived",data=train_df,kind="bar")

g.set_xticklabels(["Master","Miss-Mrs","Mr","Other"])

g.set_ylabels("Survival Probability")

plt.show()
train_df.drop(labels=["Name"],axis=1,inplace=True)
train_df = pd.get_dummies(train_df,columns=["Title"])

train_df.head()
train_df["Fsize"] = train_df["SibSp"] + train_df["Parch"] + 1

#Parch ve SibSp'nin 0 olma durumlarına bakarak kişinin kendisini aile olarak sayabilmek adına 1 ekliyoruz.
#Fsize vs Survival

g = sns.factorplot(x="Fsize", y="Survived",data=train_df, kind = "bar")

g.set_ylabels("Survival Probability")

plt.show()
train_df["family_size"] = [1 if i < 5 else 0 for i in train_df["Fsize"]]

train_df.head(5)
sns.countplot(x="family_size",data=train_df)

plt.show()
g = sns.factorplot(x="family_size",y="Survived",data=train_df,kind="bar")

g.set_ylabels("Survival Rate")

plt.show()
train_df = pd.get_dummies(train_df,columns=["family_size"])
train_df["Embarked"].value_counts()
sns.countplot(x="Embarked",data=train_df)

plt.show()
train_df = pd.get_dummies(train_df, columns=["Embarked"])

train_df.head(5)
train_df["Ticket"].head(10)
ticket_list = list()

for i in list(train_df.Ticket):

    if not i.isdigit():

        ticket_list.append(i.replace(".","").replace("/","").strip().split(" ")[0])

    else:

        ticket_list.append("X")

train_df["Ticket"] = ticket_list

train_df["Ticket"].head(15)
train_df = pd.get_dummies(train_df,columns=["Ticket"],prefix="T")

train_df.head()
sns.countplot(x="Pclass",data=train_df)

plt.show()
train_df["Pclass"] = train_df["Pclass"].astype("category")

train_df = pd.get_dummies(train_df,columns=["Pclass"])

train_df.head(5)
train_df["Sex"] =train_df["Sex"].astype("category")

train_df = pd.get_dummies(train_df,columns=["Sex"])

train_df.head(5)
train_df.drop(labels = ["PassengerId","Cabin"],axis=1,inplace=True)
from sklearn.model_selection import train_test_split,StratifiedKFold,GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier,VotingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
train_df_len

#train_df'in boyutu, test/train split yaparken index'leri ayırırken kullanacağız
test = train_df[train_df_len:]

test.drop(labels=["Survived"],axis=1,inplace=True)

test.head(5)
train = train_df[:train_df_len]

X_train = train.drop(labels="Survived",axis=1)

y_train = train["Survived"]

X_train,X_test,y_train,y_test = train_test_split(X_train,y_train,test_size=0.33,random_state=42)

print("X_test",len(X_test))

print("X_train",len(X_train))

print("y_train",len(y_train))

print("y_test",len(y_test))

print("test",len(test)) #validasyon
logreg = LogisticRegression()

logreg.fit(X_train,y_train)

acc_log_train = round(logreg.score(X_train,y_train)*100,2)

acc_log_test = round(logreg.score(X_test,y_test)*100,2)

print("Training Accuracy : % {}".format(acc_log_train))

print("Testing Accuracy : % {}".format(acc_log_test))
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



g = sns.barplot("Cross Validation Means", "ML Models", data = cv_results)

g.set_xlabel("Mean Accuracy")

g.set_title("Cross Validation Scores")
votingC = VotingClassifier(estimators = [("dt",best_estimators[0]),("rfc",best_estimators[2]),("lr",best_estimators[3])],voting="soft",n_jobs=-1)

#estimators içerisine kullanmak istediğimiz modelleri yazıyorum.

votingC = votingC.fit(X_train,y_train)

print(accuracy_score(votingC.predict(X_test),y_test))
test_survived = pd.Series(votingC.predict(test),name="Survived").astype(int)

results = pd.concat([test_passengerId,test_survived],axis=1)

results.to_csv("titanicSurvived.csv",index=False)