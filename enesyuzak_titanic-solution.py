#Import libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics

from sklearn.metrics import accuracy_score
#Reading datas

train_f=pd.read_csv("/kaggle/input/titanic/train.csv")

test_f=pd.read_csv("/kaggle/input/titanic/test.csv")

test=pd.DataFrame(test_f)

train=pd.DataFrame(train_f)

allData=train.append(test)

allData=allData.reset_index(drop=True)

ids=train["PassengerId"]

y=train["Survived"]

allData=allData.drop(["Survived"],axis=1)
train.shape
test.shape
allData.shape
train.info()
train.describe()
fig,ax=plt.subplots(figsize=(11,8))

ax=sns.heatmap(train.corr(),annot=True)

bottom,top=ax.get_ylim()

ax.set_ylim(bottom+0.5,top-0.5)

plt.title("Correlation of train")
sns.catplot(x="Pclass",

            y="Age",

            kind="box",

           data=train)
f=sns.FacetGrid(train,col="Survived",row="Pclass")

f.map(plt.hist,"Age",bins=5)
#Feature processing
allData["Ticket"].unique().shape

#Ticket feature has string values. There are 929 unique value and there is no pattern. 

#Can't be categorized. Should drop columns

allData.isnull().mean()
#Age has %20 null values.

allData["Age"]=allData["Age"].fillna(allData["Age"].mean())



#Cabin has %77 null values.

#So we should drop Ticket, Cabin features.

allData=allData.drop(["Cabin","Ticket"],axis=1)
allData.tail()
allData.isnull().sum()

#There is one row nullvalue for Fare. 

allData["Fare"]=allData["Fare"].fillna(allData["Fare"].mean())
#Embarked has 2 null values. Fill "S"

allData["Embarked"]=allData["Embarked"].fillna(allData["Embarked"].mode()[0])
#Feature Extraction

allData["Title"]=allData["Name"].str.split(", ",expand=True)[1].str.split(".",expand=True)[0]

allData["Title"].value_counts()
# Title

other_titles=allData["Title"].value_counts()<61

allData["Title"]=allData["Title"].apply(lambda s:"Other" if other_titles[s]==True else s)



# FamilySize

allData["FamilySize"]=allData["SibSp"]+train["Parch"]



#IsAlone

allData["IsAlone"]=0

allData["IsAlone"]=[1 if x<=1 else 0 for x in allData["FamilySize"]]



#FareCtg

allData['FareCtg'] = pd.qcut(allData['Fare'],5)



#AgeCtg

allData["AgeCtg"]=pd.cut(allData["Age"],4)



#Label Encoding

le=LabelEncoder()

allData["Sex"]=le.fit_transform(allData["Sex"])

allData['FareCtg']=le.fit_transform(allData['FareCtg'])

allData['AgeCtg']=le.fit_transform(allData['AgeCtg'])

allData['Embarked']=le.fit_transform(allData['Embarked'])

allData['Title']=le.fit_transform(allData['Title'])

allData.head()



#Drop all unnecessary features

allData=allData.drop(["PassengerId","SibSp","Parch","FamilySize","Name","Age","Fare"],axis=1)

train_shape=train.shape

train=allData.iloc[0:train_shape[0],:]

print(train.tail())

test=allData.iloc[train_shape[0]:,:]

print(test.head())

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(train,y,test_size=0.33,random_state=0)
#Logistic Regression

from sklearn.linear_model import LogisticRegression

logr=LogisticRegression(random_state=0,solver="newton-cg",max_iter=100)

logr.fit(x_train,y_train)

y_pred=logr.predict(x_test)



accu=np.array([[accuracy_score(y_test,y_pred),"Logistic Regression"]])
#SVC(Support Vector Classifier)

from sklearn.svm import SVC

svc=SVC(kernel="rbf")

svc.fit(x_train,y_train)

y2_pred=svc.predict(x_test)



accu=np.append(accu,[[accuracy_score(y_test,y2_pred),"SVC"]],axis=0)
#Random Forest

from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier(n_estimators=90,criterion="entropy",max_leaf_nodes=67)

rfc.fit(x_train,y_train)

y3_pred=rfc.predict(x_test)



accu=np.append(accu,[[accuracy_score(y_test,y3_pred),"Random Forest"]],axis=0)



#Decision Tree

from sklearn.tree import DecisionTreeClassifier

dtc=DecisionTreeClassifier(criterion="gini")

dtc.fit(x_train,y_train)

y4_pred=dtc.predict(x_test)



accu=np.append(accu,[[accuracy_score(y_test,y4_pred),"Desicion Trees"]],axis=0)

#Naive Bayes

from sklearn.naive_bayes import GaussianNB

from sklearn.naive_bayes import MultinomialNB

from sklearn.naive_bayes import BernoulliNB

gnb=GaussianNB()

gnb.fit(x_train,y_train)

y5_pred=gnb.predict(x_test)



accu=np.append(accu,[[accuracy_score(y_test,y5_pred),"GaussianNB"]],axis=0)



bnb=BernoulliNB()

bnb.fit(x_train,y_train)

y6_pred=bnb.predict(x_test)



accu=np.append(accu,[[accuracy_score(y_test,y6_pred),"BernoulliNB"]],axis=0)



mbn=MultinomialNB()

mbn.fit(x_train,y_train)

y7_pred=bnb.predict(x_test)



accu=np.append(accu,[[accuracy_score(y_test,y7_pred),"MultinomialNB"]],axis=0)

#KNN

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=5,metric="euclidean")

knn.fit(x_train,y_train)

y8_pred=knn.predict(x_test)



accu=np.append(accu,[[accuracy_score(y_test,y8_pred),"KNN"]],axis=0)
from sklearn.ensemble import VotingClassifier

voting=VotingClassifier(estimators=[('LR',LogisticRegression() ),

                                        ('SVM',SVC(C=1.0,gamma=0.1, kernel='rbf',probability=True,)),

                                        ('KNN',KNeighborsClassifier(n_neighbors=5)),

                                        ("RF",RandomForestClassifier(n_estimators=90,random_state=0,criterion="entropy",max_leaf_nodes=67) ),

                                        ('DT',DecisionTreeClassifier(criterion='entropy', max_depth=6))])

voting.fit(x_train,y_train)

y9_pred=voting.predict(x_test)



accu=np.append(accu,[[accuracy_score(y_test,y9_pred),"Voting Class"]],axis=0)



from sklearn.ensemble import AdaBoostClassifier

ada=AdaBoostClassifier(n_estimators=10,random_state=0)

ada.fit(x_train,y_train)

y10_pred=ada.predict(x_test)



accu=np.append(accu,[[accuracy_score(y_test,y10_pred),"AdaBoost"]],axis=0)
accu
from sklearn.ensemble import GradientBoostingClassifier

Grad_boost=GradientBoostingClassifier(n_estimators=500,random_state=0,learning_rate=0.01)

model=Grad_boost.fit(x_train,y_train)

y11_pred=model.predict(x_test)



accu=np.append(accu,[[accuracy_score(y_test,y11_pred),"GradientBoost"]],axis=0)
from sklearn.ensemble import BaggingClassifier

Bagged_Decision=BaggingClassifier(base_estimator=DecisionTreeClassifier(),random_state=0,

                                  n_estimators=90,bootstrap=True,max_samples=500

                                 )



Bagged_Decision.fit(x_train,y_train)

Bagged_Decision.score(x_train, y_train)

y12_pred=Bagged_Decision.predict(x_test)

accu=np.append(accu,[[accuracy_score(y_test,y12_pred),"BaggingBoost"]],axis=0)
from sklearn.linear_model import Perceptron

prc=Perceptron(tol=1e-3)

model=prc.fit(x_train,y_train)

y13_pred=model.predict(x_test)

accu=np.append(accu,[[accuracy_score(y_test,y13_pred),"Perceptron"]],axis=0)
#Stochastic Gradient

from sklearn.linear_model import SGDClassifier

sg=SGDClassifier(loss="log",penalty="l2",max_iter=5)

sg.fit(x_train,y_train)

y14_pred=sg.predict(x_test)

accu=np.append(accu,[[accuracy_score(y_test,y14_pred),"Stochastic Gradient"]],axis=0)

#ExtraTreesClassifier

from sklearn.ensemble import ExtraTreesClassifier

model=ExtraTreesClassifier(n_estimators=90,criterion="entropy",max_leaf_nodes=67)

model.fit(x_train,y_train)

y15_pred=model.predict(x_test)

accu=np.append(accu,[[accuracy_score(y_test,y15_pred),"ExtraTreesClassifier"]],axis=0)
#XGBoost

import xgboost as xgb

xc=xgb.XGBClassifier(objective="binary:logistic", random_state=42)

xc=xc.fit(x_train,y_train)

y16_pred=xc.predict(x_test)

accu=np.append(accu,[[accuracy_score(y_test,y16_pred),"XGBoost"]],axis=0)
#ML algorithms accuracy:

df_accu=pd.DataFrame(accu,columns=["Accuracy","Algorithms"])

df_accu.to_csv("Metrics.csv")
df_accu.sort_values(by="Accuracy",ascending=False)
#The best model is BaggingClassifier
test.head()
y_sub=xc.predict(test)
submission=pd.DataFrame({"PassengerId":test_f["PassengerId"],

                        "Survived":y_sub})
submission.to_csv("gender_submission.csv")