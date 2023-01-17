# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid") #style of plot

#plt.style.avaliable -> show the avaliable styles of plots



import seaborn as sns



from collections import Counter



import warnings

warnings.filterwarnings("ignore") #do not show errors from python



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("/kaggle/input/titanic/train.csv")

test_df = pd.read_csv("/kaggle/input/titanic/test.csv")

test_PassengerId = test_df["PassengerId"] #for accuracy in the future
train_df.columns
train_df.head()
train_df.describe().T
train_df.info()
def bar_plot(variable):

    """

    input: variable ex: sex

    output: bar plot & value count

    """

    var = train_df[variable]

    varValue = var.value_counts()

    

    plt.figure(figsize = (9,3))

    plt.bar(varValue.index, varValue)

    plt.xticks(varValue.index, varValue.index.values)

    plt.ylabel("Frequency")

    plt.title(variable)

    plt.show()

    

    print("{}: \n {}".format(variable,varValue))
category = ["Survived","Sex","Pclass","Embarked","Parch","SibSp"]

for i in category:

    bar_plot(i)
def plt_hist(variable):

    plt.figure(figsize=(9,3))

    plt.hist(train_df[variable],bins = 10)

    plt.xlabel(variable)

    plt.ylabel("Frequency")

    plt.title("{} distribution with hist".format(variable))

    plt.show()
numVar = [ "Age","Fare","PassengerId"]

for i in numVar:

    plt_hist(i)
#pclass - survived

train_df[["Pclass","Survived"]].groupby(["Pclass"],as_index = False).mean()
train_df[["Pclass","Survived"]].groupby(["Pclass"],as_index = False).mean().sort_values(by = "Survived",ascending=False) #sorting
#Sex - Survived

train_df[["Sex","Survived"]].groupby(["Sex"],as_index = False).mean().sort_values(by = "Survived",ascending=False) #sorting
#Embarked - survived

train_df[["Embarked","Survived"]].groupby(["Embarked"],as_index = False).mean().sort_values(by = "Survived",ascending=False) #sorting
#Parch - Survived

train_df[["Parch","Survived"]].groupby(["Parch"],as_index = False).mean().sort_values(by = "Survived",ascending=False) #sorting
#SibSp - survived

train_df[["SibSp","Survived"]].groupby(["SibSp"],as_index = False).mean().sort_values(by = "Survived",ascending=False) #sorting
#birbiriyle ayrı bilgi veren featureları birleştirip ayrı feature yapabilirim.
#istatiksel anlamda karar vermemizi zorlaştırır. Ortalamayı olduğundan daha yüksek ve ya daha düşük gösterir.

#IQR = Q3 - Q1

#1.5 x IQR = Z

#(Q1-Z) ile (Q3 + Z) dışındakiler outlier

def outlier_detection(df,features):

    outlier_indices = []

    

    for i in features:

        #Q1

        q1 = np.percentile(df[i],25)

        #Q3

        q3 = np.percentile(df[i],75)

        #IQR

        ıqr = q3 - q1

        #Ouitler Step

        outlier_step = ıqr * 1.5

        #detect outlier and their indeces

        outlier_list = df[(df[i] < q1 - outlier_step) | (df[i] > q3 + outlier_step)].index

        #store indices

        outlier_indices.extend(outlier_list)

    #print(outlier_indices) 

    

    from collections import Counter #sayma işlemi

    outlier_indices = Counter(outlier_indices)

    #print(outlier_indices)

    

    #eğer o outlier sadece o feature da outlier ise çıkarmaya çok gerek yok. 2den fazla yerde ise çıkarmak daha mantıklı

    multiple_outliers = list(i for i,v in outlier_indices.items() if v >2) #bu sözlük yapısı ondan

    

    return multiple_outliers

        
outlier_detection(train_df,["Age","SibSp","Parch","Fare"])
indexes = outlier_detection(train_df,["Age","SibSp","Parch","Fare"])

train_df.loc[indexes] #yani 2den fazla feature için outlier olanlar
#drop outlier

train_df = train_df.drop(indexes,axis = 0).reset_index(drop=True)
train_df_len = len(train_df)

train_df = pd.concat([train_df,test_df],axis = 0).reset_index(drop=True) #satır concat
train_df.columns[train_df.isnull().any()] #existing missing value which feature
#train_df.isnull() #boş mu- if it is null, output is True
train_df.isnull().sum()
train_df[train_df["Embarked"].isnull()]
train_df[train_df["Ticket"]=="113572"] #aynı ticket belki aynı limandan kesilmiştir diye başka aynı numaraya sahip yolcu var mı? yokmuş
train_df.boxplot(column = "Fare",by = "Embarked")

plt.show()
#Bu iki yolcunun embarked'ı C olma olasılığı daha yüksek.

train_df["Embarked"] = train_df["Embarked"].fillna("C")
train_df[train_df["Embarked"].isnull()] #control
#Fare

train_df[train_df["Fare"].isnull()]
train_df[train_df["Ticket"]=="3701"]
train_df[["Embarked","Fare"]].groupby(["Embarked"],as_index = False).mean()
train_df["Fare"][train_df["Embarked"]=="S"].describe().T
train_df["Fare"][train_df["Pclass"]==3].describe().T #fill average ticket value in 3.class 
train_df["Fare"] = train_df["Fare"].fillna(np.mean(train_df[train_df["Pclass"]==3]["Fare"]))
#Fare

train_df[train_df["Fare"].isnull()] #control
list1 = ["SibSp","Parch","Age","Fare","Survived"]

sns.heatmap(train_df[list1].corr(), annot = True, fmt = ".2f") #annot=üzerine sayı yazdırma, fmt = virgülden sonra iki basamak

plt.show()
g = sns.factorplot(x = "SibSp", y="Survived",data=train_df,kind="bar",size = 6) #size boyutu

g.set_ylabels("Survived Probability")

plt.show()

#siyah çizgiler standart sapma
g = sns.factorplot(x = "Parch",y = "Survived",data = train_df,size=6)

g.set_ylabels("Survived Probability")

plt.show()
g = sns.factorplot(x = "Parch",y = "Survived",data = train_df,size=6,kind = "bar")

g.set_ylabels("Survived Probability")

plt.show()
g = sns.factorplot(x = "Pclass",y = "Survived",data = train_df,size=6,kind = "bar")

g.set_ylabels("Survived Probability")

plt.show()
g = sns.FacetGrid(train_df,col="Survived")

g.map(sns.distplot,"Age",bins = 25)

plt.show() #survived ı 1 olanların ve 0 olanların yaş dağılımları
g = sns.FacetGrid(train_df,col = "Survived",row = "Pclass",size = 2)

g.map(plt.hist,"Age",bins = 25)

plt.show()
g = sns.FacetGrid(train_df, row = "Embarked", size = 3)

g.map(sns.pointplot, "Pclass","Survived","Sex")

g.add_legend()

plt.show()
g = sns.FacetGrid(train_df,row="Embarked",col = "Survived",size=3)

g.map(sns.barplot,"Sex","Fare")

plt.show()
train_df[train_df["Age"].isnull()]
sns.factorplot(x="Sex",y = "Age",data = train_df, kind="box")

plt.show()
sns.factorplot(x="Sex",y = "Age", hue = "Pclass",data = train_df, kind="box")

plt.show()
sns.factorplot(x="Parch",y = "Age",data = train_df, kind="box")

sns.factorplot(x="SibSp",y = "Age",data = train_df, kind="box")

plt.show()
train_df["Sex"] = [1 if i =="male" else 0 for i in train_df["Sex"]]
#correlation

sns.heatmap(train_df[["Age","Sex","Parch","SibSp","Pclass"]].corr(),annot=True,fmt=".2f")

plt.show()
index_nan_age = list(train_df["Age"][train_df["Age"].isnull()].index)

for i in index_nan_age:

    age_pred = train_df["Age"][((train_df["SibSp"]==train_df.iloc[i]["SibSp"]) & (train_df["Parch"]==train_df.iloc[i]["Parch"]) & (train_df["Pclass"]==train_df.iloc[i]["Pclass"]))].median()

#yani bu age i nan olan indexteki sibsp, parch ve pclass değerleri aynı olan diğer indexlerdeki kişilerin median değerini yaş olarak kullan demek.

#ama 3ünün de aynı olduğu index olmayabilir. Bundan dolayı o nan larada da median değerleri atanır.

    age_med = train_df["Age"].median()

    if not np.isnan(age_pred): #nan değilse age pred

        train_df["Age"].iloc[i] = age_pred

    else: #nan değer ise

        train_df["Age"].iloc[i] = age_med
train_df[train_df["Age"].isnull()] #control
#Name -- Title #name sacma ama title a göre yaşama olayına bakabilirz.

train_df["Name"].head()

s = "Braund, Mr. Owen Harris" #ilk noktaya göre ikiye ayırıp daha sonra da virgüle göre ayırıp Mr. ve Mrs. alabiliriz.

s.split(".")[0].split(",")[-1].strip()
name = train_df["Name"]

train_df["Title"] = [i.split(".")[0].split(",")[-1].strip() for i in name ]
train_df["Title"].head()
train_df["Title"].unique() #bunlar aslında title yani ünvan
sns.countplot(x = "Title",data=train_df)

plt.xticks(rotation = 60)

plt.show()
train_df[["Title","Survived"]].groupby(["Title"],as_index=False).mean().sort_values(by="Survived",ascending = False)
train_df[train_df["Title"]=="Dona"]
#I form 2 title feature. Title=mine, Title2 = DATAI. I want to use these title seperatingly.

len(train_df["Title"].unique())
#pandas.factorize( ['B', 'C', 'D', 'B'] )[0]

train_df["Title"]= pd.factorize(train_df["Title"])[0]
train_df["Title"].unique()
#Datai Title2

name = train_df["Name"]

train_df["Title2"] = [i.split(".")[0].split(",")[-1].strip() for i in name ]
sns.countplot(x = "Title2",data=train_df)

plt.xticks(rotation = 60)

plt.show()
train_df["Title2"] = train_df["Title2"].replace(["Lady","the Countess","Capt","Col","Don","Dr","Major","Rev","Sir","Jonkheer","Dona"],"Others")

sns.countplot(x = "Title2",data=train_df)

plt.xticks(rotation = 60)

plt.show()
#convert to categorical

train_df["Title2"] = [0 if i == "Master" else 1 if i == "Miss" or i == "Ms" or i == "Mlle" or i == "Mrs" else 2 if i =="Mr" else 3 for i in train_df["Title2"] ]



sns.countplot(x = "Title2",data=train_df)

plt.xticks(rotation = 60)

plt.show()
sns.factorplot(x ="Title2",y ="Survived", kind = "bar",data = train_df)

plt.show()
#Name is not neccessary for training.

train_df.drop(labels= ["Name"],axis = 1,inplace = True)

train_df.head(7)
#Biz bunları title ı 0,1,2,3 yapmaktansa 4title yapmak daha mantıklı

train_df = pd.get_dummies(train_df,columns = ["Title2"])

train_df.head()
#Forming new feature with SibSp and Parch

train_df["Fsize"] = train_df["SibSp"] + train_df["Parch"] + 1 #Aile 0 kisiden oluşmaz.
#Relationship with survived

g =sns.factorplot(x="Fsize",y="Survived",kind="bar",data = train_df)

g.set_ylabels("Survival Probability")

plt.show()
train_df["FamilySize"] = [1 if i < 5 else 0 for i in train_df["Fsize"]]

train_df.head(5)
sns.countplot(x = "FamilySize",data=train_df)

plt.show()
g =sns.factorplot(x="FamilySize",y="Survived",kind="bar",data = train_df)

g.set_ylabels("Survival Probability")

plt.show()
train_df = pd.get_dummies(train_df,columns=["FamilySize"])

train_df.head()
train_df = pd.get_dummies(train_df,columns=["Embarked"])

train_df.head()
train_df["Ticket"] #sol taraftakiler aynı seyler olmayanlara X diyebilirz.
tickets = []

for i in train_df.Ticket:

    if not i.isdigit(): #yani içinde sayıdan farklı karekter varsa bu koşula girer. isdigit() sadece rakam varsa True döndürür

        tickets.append(i.replace(",","").replace(".","").replace("/","").strip().split(" ")[0])

    else:

        tickets.append("X")

train_df["Ticket"] = tickets
train_df.head(10)
train_df["Ticket"].unique()
train_df = pd.get_dummies(train_df,columns=["Ticket"],prefix = "T")

train_df.head()
train_df["Pclass"] = train_df["Pclass"].astype("category")

train_df = pd.get_dummies(train_df,columns=["Pclass"])

train_df.head()
train_df["Sex"] = train_df["Sex"].astype("category")

train_df = pd.get_dummies(train_df,columns=["Sex"])

train_df.head()
df = train_df.copy()

train_df.drop(labels = ["PassengerId","Cabin"],axis = 1,inplace = True)
train_df.columns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
train_df.drop(labels = ["Title"],axis = 1,inplace = True)
test = train_df[train_df_len:]

test.drop(labels = ["Survived"], axis = 1,inplace = True)

test.head()
train = train_df[:train_df_len]

x_train = train.drop(labels = ["Survived"],axis = 1)

y_train = train["Survived"]



x_train,x_test,y_train,y_test = train_test_split(x_train,y_train,test_size = 0.33,random_state = 42)

print("x_train",len(x_train))

print("x_test",len(x_test))

print("y_train",len(y_train))

print("y_test",len(y_test))

print("test",len(test))
log_reg = LogisticRegression()

log_reg.fit(x_train,y_train)

acc_train = round(log_reg.score(x_train,y_train)*100,2)

acc_test = round(log_reg.score(x_test,y_test)*100,2)

print("Training Accuracy: %{}".format(acc_train))

print("Test Accuracy: %{}".format(acc_test))
random = 42

classifier = [DecisionTreeClassifier(random_state=random),

             SVC(random_state=random),

             RandomForestClassifier(random_state=random),

             LogisticRegression(random_state=random),

             KNeighborsClassifier()]

dt_param_grid = {"min_samples_split":range(10,500,20),

                "max_depth":range(1,20,2)}

svc_param_grid = {"kernel":["rbf"],

                 "gamma":[0.001,0.01,0.1,1],

                 "C":[1,10,50,100,200,300,1000]}

rf_param_grid = {"max_features": [1,3,10],

                "min_samples_split":[2,3,10],

                "min_samples_leaf":[1,3,10],

                "bootstrap":[False],

                "n_estimators":[100,300],

                "criterion":["gini"]}

logreg_param_grid = {"C":np.logspace(-3,3,7),

                    "penalty":["l1","l2"]}

knn_param_grid = {"n_neighbors":np.linspace(1,19,10,dtype = int).tolist(),

                 "weights":["uniform","distance"],

                 "metric":["euclidean","manhattan"]}

classifier_param = [dt_param_grid,svc_param_grid,rf_param_grid,logreg_param_grid,knn_param_grid]
cv_result = []

best_estimators = []

for i in range(len(classifier)):

    clf = GridSearchCV(classifier[i],param_grid=classifier_param[i],cv = StratifiedKFold(n_splits=10),scoring="accuracy",n_jobs = -1, verbose = 1)

    clf.fit(x_train,y_train)

    cv_result.append(clf.best_score_)

    best_estimators.append(clf.best_estimator_)

    print(cv_result[i])
cv_results = pd.DataFrame({"Cross Validation Means":cv_result,"ML Models":["DecisonTree","SVC","RandomForest","LogReg","KNN"]})

g = sns.barplot("Cross Validation Means","ML Models",data = cv_results)

g.set_xlabel("Mean Accuracy")

g.set_title("Cross Validation Score")

plt.show()
votingC = VotingClassifier(estimators = [("dt",best_estimators[0]),

                                        ("rf",best_estimators[2]),

                                        ("lr",best_estimators[3])],

                          voting = "soft",n_jobs = -1)

VotingC = votingC.fit(x_train,y_train)

print(accuracy_score(VotingC.predict(x_test),y_test))
df.drop(labels = ["Title"],axis = 1,inplace = True)
titanic_test = df[train_df_len:]

titanic_test.drop(labels = ["Cabin","Survived"],axis = 1,inplace = True)
titanic_test.head()

test_PassengerId = titanic_test["PassengerId"].values

test_PassengerId
test_survived = pd.Series(votingC.predict(test),name = "Survived").astype(int)

test_PassengerId = pd.Series(test_PassengerId,name ="PassengerId")

results = pd.concat([test_PassengerId,test_survived],axis = 1)

results.to_csv("titanic.csv",index = False)