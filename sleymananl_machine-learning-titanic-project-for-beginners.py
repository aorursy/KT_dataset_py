import pandas as pd



import numpy as np
import matplotlib.pyplot as plt



plt.style.use("seaborn-whitegrid")



import seaborn as sns



from collections import Counter



import warnings



warnings.filterwarnings("ignore")





#plt.style.available(görselleştirmede kullanabilecek styllar)
train_df=pd.read_csv("/kaggle/input/titanic/train.csv")

test_df=pd.read_csv("/kaggle/input/titanic/test.csv")

test_PassengerId=test_df["PassengerId"]


train_df.columns
train_df.head()
train_df.describe()
train_df.info()
def bar_plot(variable):

    #get feature

    var=train_df[variable]

    

    #count number of categorical values

    var_value=var.value_counts()

    

    #visualization

    plt.figure(figsize=(4,10))

    plt.bar(var_value.index,var_value)

    plt.xticks(var_value.index,var_value.index.values)

    plt.ylabel("Frequency")

    plt.title("Variable")

    plt.show()

    print(" {}: \n {}".format(variable,var_value))

    
category1=["Survived","Sex","Pclass","Embarked","SibSp","Parch"]

for c in category1:

    bar_plot(c)
def plot_hist(variable):

    plt.xlabel(variable)

    plt.ylabel("Frequency")

    plt.title("{} distrubition with histogram".format(variable))

    plt.hist(train_df[variable])

    plt.show()

    
Numeric_var=["Fare","Age","PassengerId"]



for i in Numeric_var:

    plot_hist(i)
#Pclass-Survived involvement

train_df[["Pclass","Survived"]].groupby(["Pclass"],as_index=False).mean()
#Sex-Survived involvement

train_df[["Sex","Survived"]].groupby(["Sex"],as_index=False).mean().sort_values(by="Survived",ascending=False)
#SibSp-Survived involvement

train_df[["SibSp","Survived"]].groupby(["SibSp"],as_index=False).mean().sort_values(by="Survived",ascending=False)
train_df[["Parch","Survived"]].groupby(["Parch"],as_index=False).mean().sort_values(by="Survived",ascending=False)
def detect_outliers(train_df,features):

    outlier_indices = []

    

    for c in features:

        #1 quartile:

        Q1= np.percentile(train_df[c],25)

        

        #3 quartile:

        Q3= np.percentile(train_df[c],75)

        #IQR

        IQR= Q3 - Q1

        

        #Outlier Step

        outlier_step=IQR * 1.5

        

        #Detect Outlier and their indeces

        outlier_list_col=train_df[(train_df[c] < Q1- outlier_step) | (train_df[c] > Q3+ outlier_step)].index

        

        #store indices

        outlier_indices.extend(outlier_list_col)

        

    outlier_indices= Counter(outlier_indices)

    

    multiple_outliers= list(i for i, v in outlier_indices.items() if v >2)

    

    return multiple_outliers

          

    
train_df.loc[detect_outliers(train_df,["Parch","Fare","Age","SibSp"])]

        
#droping process



train_df=train_df.drop(detect_outliers(train_df,["Parch","Fare","Age","SibSp"]),axis=0).reset_index(drop=True)

train_df.head()
train_df_len=len(train_df)

train_df=pd.concat([train_df,test_df],axis=0).reset_index(drop=True)

train_df.head()
train_df.columns[train_df.isnull().any()]

train_df.isnull().sum()
train_df[train_df["Embarked"].isnull()]

train_df.boxplot(column="Fare",by="Embarked"); # passengers who paid 80$ got on generally C point so we can fill the missing values on embarked through "C".
train_df["Embarked"]=train_df["Embarked"].fillna("C")

train_df[train_df["Embarked"].isnull()]
train_df[train_df["Fare"].isnull()]
train_df["Fare"]=train_df["Fare"].fillna(np.mean(train_df[train_df["Pclass"]==3]["Fare"]))
train_df[train_df["Fare"].isnull()] #we filled the missing values with Fare average




import seaborn as sns

list1=["SibSp","Age","Pclass","Parch","Fare","Survived"]



sns.heatmap(train_df[list1].corr(),annot=True,fmt=".2f") #annot=boxların içindeki değerleri ifade eder. #fmt=değerlerin vigülden sonraki basamak sayısını ifade eder

plt.show()





sns.factorplot(x="SibSp",y="Survived",data=train_df,kind="bar")

plt.ylabel("survived probability")

plt.xlabel("number of siblings")

plt.title("the correlation between survived probability and number of siblings")

plt.show()
sns.factorplot(x="Parch",y="Survived",data=train_df,kind="bar")

plt.ylabel("survived probability")

plt.xlabel("large of parch")

plt.title("the correlation between survived probability and number of siblings")

plt.show()
train_df[train_df["Age"].isnull()]
sns.boxplot(x="Sex",y="Age",hue="Pclass",data=train_df)

plt.show()
sns.boxplot(x="Parch",y="Age",data=train_df)

plt.show()



sns.boxplot(x="SibSp",y="Age",data=train_df)

plt.show()
train_df["Sex"]=[1 if i =="male" else 0 for i in train_df["Sex"] ]
sns.heatmap(train_df[["Sex","SibSp","Pclass","Parch","Age"]].corr(),annot=True)
index_nan_age=list(train_df["Age"][train_df["Age"].isnull()].index)

index_nan_age
for i in index_nan_age:

    age_pred=train_df["Age"][train_df["SibSp"]==train_df.iloc[i]["SibSp"]].median()

    age_med=train_df["Age"].median()

    

    if not np.isnan(age_pred):

        train_df["Age"].iloc[i]=age_pred

    else:

        train_df["Age"].iloc[i]=age_med
train_df[train_df["Age"].isnull()]
name=train_df["Name"]

train_df["Title"]=[i.split(".")[0].split(",")[-1].strip() for i in name]  #öncelikle . ya göre satırı ikiye ayırdık ve birinci kısmı seçtik(Title ın olduğu kısım).Daha sonra ise , ile ayırarak son kısmı seçtik.
train_df["Title"]
sns.countplot(x="Title",data=train_df)

plt.xticks(rotation=50)

plt.show
#convert to categorical

train_df["Title"]=train_df["Title"].replace(["Lady","Capt","Col","the Countess","Jonkheer","Dona","Major","Rec","Dr","Don","Sir"],"other")

train_df["Title"]=[0 if i =="Master" else 1 if i == "Miss" or i=="Ms" or i=="Mlle" or i=="Mrs" else 2 if i =="Mr" else 3 for i in train_df["Title"]]
sns.countplot(x="Title",data=train_df)

plt.xticks(rotation=50)

plt.show
g=sns.barplot(x="Title",y="Survived",data=train_df)

g.set_xticklabels(["Master","Mrs","Mr","Other"])  # x değerleri köşeli parantez içinde olmalı  

plt.show()
train_df.drop(labels=["Name"],inplace=True,axis=1)  #axis=1 unutulmamalı 
train_df=pd.get_dummies(train_df,columns=["Title"])#get_dummies in işlevi=nümeric değerlere dönüşen Title columnunu 1 ve 0 olarak sınıflandırmak

train_df
train_df["Fsize"]=train_df["Parch"]+train_df["SibSp"]+1 # +1 in sebebi ,tek kişiyi de bir kişilik aile olarak düşünebiliriz

train_df
g=sns.factorplot(x="Fsize",y="Survived",data=train_df,kind="bar")

g.set_ylabels("Survival pprobability")

g.set_xlabels("Family Size")

plt.show()
train_df["Family_size"]=[1 if i< 5 else 0 for i in train_df["Fsize"]]
train_df.head(10)
sns.countplot(x="Family_size",data=train_df); #countplotta y değeri kullanmamalısın
train_df=pd.get_dummies(train_df,columns=["Family_size"])
train_df.head(10)
g=sns.countplot(x="Embarked",data=train_df)



plt.show()
train_df["Embarked"]=[0 if i=="S" else 1 if i =="C" else 2  for i in train_df["Embarked"]]

train_df.head(10)
train_df=pd.get_dummies(train_df,columns=["Embarked"])

train_df
train_df["Ticket"]
tickets=[]

for i in list(train_df.Ticket):

    if not i.isdigit():

        tickets.append(i.replace(".","").replace("/","").strip().split()[0])

    else:

             tickets.append("x")   
#example for display

a="SOTON/O.Q. 3101262"



a.replace(".","").replace("/","").strip().split()[0]


train_df["Ticket"]=tickets

train_df["Ticket"].head(10)
train_df=pd.get_dummies(train_df,columns=["Ticket"],prefix="T")
train_df.head(20)
sns.countplot(x="Pclass",data=train_df);
train_df["Pclass"]=train_df["Pclass"].astype("category")

train_df=pd.get_dummies(train_df,columns=["Pclass"])
train_df
train_df["Sex"]=train_df["Sex"].astype("category")

train_df=pd.get_dummies(train_df,columns=["Sex"])
train_df.head(10)
train_df.drop(labels=["PassengerId","Cabin"],axis=1,inplace=True)

train_df
from sklearn.model_selection import train_test_split ,GridSearchCV,StratifiedKFold

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier,VotingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
train_df_len
test=train_df[train_df_len:]

test.drop(labels=["Survived"],axis=1,inplace=True)
test.head()
train=train_df[:train_df_len]

x_train=train.drop(labels="Survived",axis=1)

y_train=train["Survived"]

x_train,x_test,y_train,y_test=train_test_split(x_train,y_train,test_size=0.3,random_state=42)
print("x_test=",len(x_test))

print("x_train=",len(x_train))

print("y_test=",len(y_test))

print("y_train=",len(y_train))
logreg=LogisticRegression()

logreg_model=logreg.fit(x_train,y_train)

logreg_pred=logreg_model.predict(x_test)

#accuracy_score

accuracy_score_train=logreg.score(x_train,y_train)

accuracy_score_train
accuracy_score_test=logreg.score(x_test,y_test)

accuracy_score_test
classifier=[DecisionTreeClassifier(random_state=42),

            SVC(random_state=42),

            RandomForestClassifier(random_state=42),

            LogisticRegression(random_state=42),

            KNeighborsClassifier()]

dt_params={"min_samples_split":range(10,500,20),"max_depth":range(10,20,2)}



svc_params= {"kernel":["rbf"],

             "gamma":[0.001,0.01,0.1],

             "C":[10,30,50,100,200]}



rf_params={"min_samples_split":[1,3,10],

          "max_features":[1,3,10],

          "bootstrap":[False],

          "n_estimators":[100,300],

          "criterion":["gini"]}



logreg_params={"C":np.logspace(-3,3,7),

              "penalty":["l1","l2"]}



knn_params={"n_neighbors":np.linspace(1,19,10,dtype=int).tolist(),

           "weights":["uniform","distance"],

           "metric":["euclidean","manhattan"]}



classifier_params=[dt_params,svc_params,rf_params,logreg_params,knn_params]
cv_result=[]

best_estimators=[]

for i in range(len(classifier)):

    clf=GridSearchCV(classifier[i],param_grid=classifier_params[i],cv=StratifiedKFold(n_splits=10))

    

    clf.fit(x_train,y_train)

    

    cv_result.append(clf.best_score_)

    best_estimators.append(clf.best_estimator_)

    print("best score for {}".format(classifier[i]),cv_result[i])

print("best estimators for {}".format(classifier[i]),best_estimators)
cv_result=pd.DataFrame({"ML Models":["DecisionTreeClassifier","SVC","RandomForestClassifier","LogisticRegression","KNeighborsClassifier"],"Cross Validation Means":cv_result})

cv_result









sns.barplot(x="ML Models",y="Cross Validation Means",data=cv_result)

plt.xticks(rotation=25)
votingC=VotingClassifier(estimators=[("dt",best_estimators[0]),("rfc",best_estimators[2]),("lr",best_estimators[3])], voting = "soft",n_jobs=-1)

votingC=votingC.fit(x_train,y_train)

votingC_pred=votingC.predict(x_test)
print("accuracy score for test:",accuracy_score(votingC_pred,y_test))
test_survived=pd.Series(votingC_pred,name="Survived").astype(int)

results=pd.concat([test_PassengerId,test_survived],axis=1)

results.to_csv("titanic.csv",sep=",",index=False)