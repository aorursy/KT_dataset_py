import pandas as pd
import seaborn as sns
from matplotlib import pyplot as py
import numpy as np
%matplotlib inline
train=pd.read_csv("../input/insurance.csv") #reading the data
train.head() #used to display the dataframe..
train.describe()

#average insurance premium is 13270..
train["smoker"].value_counts()
#smoker not balancedd
train.groupby(by="sex")["charges"].mean()
sns.violinplot(x="sex",y="charges",data=train)
train.shape
sns.distplot(train["charges"])  #plotting histogram using seaborn displot
                            #funtion to check the skewness in the data
train["charges"].skew()    #skewness in gretaer than it means we have to
                         #normalize it..
from scipy import stats
train["charges"]=stats.boxcox(train["charges"])[0]

#normalizing the charges variable using boxcox tranformation..
train["charges"].skew() #now we see skewness reduces to normal value..
sns.distplot(train["charges"])   #as we can see plot is approx to bell-curve which is good indiaction that our data is normalize
train.isnull().sum()
sns.regplot(x="bmi",y="charges",data=train)
pd.cut(train["bmi"],bins=5).unique()        # diving bmi varibale into 5 diffrent bins.
train=train.loc[train["bmi"]<49]   #removing outlinerss.
pd.cut(train["bmi"],bins=5).unique()
train.loc[(train["bmi"]>22.382) & (train["bmi"]<=28.0804),"ibmi"]=2
train.loc[(train["bmi"]>28.804) & (train["bmi"]<=35.226),"ibmi"]=3

train.loc[(train["bmi"]>35.226) & (train["bmi"]<=41.648),"ibmi"]=4

train.loc[(train["bmi"]>41.648) & (train["bmi"]<=48.07),"ibmi"]=5

train.loc[(train["bmi"]>15.928) & (train["bmi"]<=22.382),"ibmi"]=1


sns.barplot(x="ibmi",y="charges",data=train)
sns.swarmplot(x="ibmi",y="charges",data=train)
sns.regplot(x="age",y="charges",data=train)#linear relation ..
train.loc[train["smoker"]=="yes","smoker"]=1
train.loc[train["smoker"]=="no","smoker"]=0
sns.barplot(x="smoker",y="charges",data=train)
sns.violinplot(x="smoker",y="charges",data=train)
sns.swarmplot(x="ibmi",y="charges",data=train)
train["children"].value_counts()
sns.swarmplot(x="children",y="charges",data=train)
sns.barplot(x="children",y="charges",data=train)
train.groupby(by="children")["charges"].mean()
train["region"].value_counts()
sns.swarmplot(x="region",y="charges",data=train)
sns.swarmplot(x="sex",y="charges",data=train)
sns.barplot(x="sex",y="charges",data=train)
train.drop("sex",axis=1,inplace=True) #dropping sex variable because it don't appears usefull.
train.drop("ibmi",axis=1,inplace=True)
train=pd.get_dummies(data=train).astype("int64")

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier

train,test=train_test_split(train,test_size=0.28)
f=test["charges"]
test.drop("charges",axis=1,inplace=True)
y=train["charges"]
train.drop("charges",axis=1,inplace=True)


from sklearn import metrics
lr=LogisticRegression()
train["bmi"]=train["bmi"].astype("int64")
lr.fit(train,y)
#y_pred=lr.predict(test)
#print("accurac is ",metricsp.accuracy_score(y_pred,f))
y_prd=lr.predict(test)
print("accuracy is",metrics.accuracy_score(y_prd,f))
lr=SVC()
lr.fit(train,y)
y_prd=lr.predict(test)
print("accuracy is",metrics.accuracy_score(y_prd,f))



lr=DecisionTreeClassifier()
lr.fit(train,y)
y_prd=lr.predict(test)
print("accuracy is",metrics.accuracy_score(y_prd,f))

from sklearn.linear_model import LinearRegression

lr=LinearRegression()
lr.fit(train,y)
y_p=lr.predict(test)
x=metrics.mean_squared_error(f,y_p)
x=x**1/2
print(x)
x=metrics.mean_absolute_error(f,y_p)