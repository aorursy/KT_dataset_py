# Importing Packages

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import os

#os.chdir(r"C:\DATA SET")
#pd.set_option('display.max_columns',None)

data=pd.read_csv("../input/snsdata1.csv")

data.head()
data.columns
data.info()
data.nunique()#check no of unique values in per variable

data.gender.count()
data.gender=data.gender.astype("object")
data.info()#check size of the data gender make the categorical  variable the reduce the size of data i.e 1.4 mb
summary=data.describe(percentiles=[.1,.25,.5,.75,.90,.95,.99]).T
summary
summary.to_csv(r"C:\machine learning\summarystats.csv")#convert into csvfile
#data.describe(include=["category"])#object data summary
data.isna().sum()#check the null values and percentage
data.columns#col in data
data.gender.value_counts(dropna=False)#check values in the gender col... 435 are null values
#data.gender.fillna("not-disclosed",inplace=True)
data.gender.value_counts()#value counts in gender col
data.age.isna().sum()#726 values present in age col
q1=data.quantile(.25)
q3=data.quantile(.75)
iqr=q3-q1
lbv=q1-(1.5*iqr)

lbv
ubv=q3+(1.5*iqr)

ubv
sns.boxplot(data["age"])
#data[(data["age"]>ubv)|(data["age"]<lbv)]["age"]
num_var=['age',

 'friends',

 'basketball',

 'football',

 'soccer',

 'softball',

 'volleyball',

 'swimming',

 'cheerleading',

 'baseball',

 'tennis',

 'sports',

 'cute',

 'sex',

 'sexy',

 'hot',

 'kissed',

 'dance',

 'band',

 'marching',

 'music',

 'rock',

 'god',

 'church',

 'jesus',

 'bible',

 'hair',

 'dress',

 'blonde',

 'mall',

 'shopping',

 'clothes',

 'hollister',

 'abercrombie',

 'die',

 'death',

         

 'drunk',

 'drugs']

#detect out lier

for i in data[num_var]:#outliers treatment

    per=data[i].quantile([.01,.99]).values

    #print(per)

    print(data[i][(data[i]<=per[0])|(data[i]>=per[1])])    
for i in data[num_var]:#outliers treatment

    per=data[i].quantile([.01,.99]).values#quantile of 1 per and 99 per

    #print(per)

    a=data[i][data[i]<=per[0]]=per[0]#data is less than 1 per than outlier

    b=data[i][data[i]>=per[1]]=per[1]#data is more than 99 per  cent than out lier

    print(a,b)
data.head()

data.describe()
sns.boxplot(data["age"])
from sklearn.preprocessing import StandardScaler

names=data.columns[5:40]

names
data.columns
features=data[names]

features.head()
sc=StandardScaler()
features=sc.fit_transform(features)
features
data[names]=features
data
#create dummy var

#check null values

data.isna().sum()

#gender 435 na values and age 726 null values

data.gender.fillna("not disclosed",inplace=True)
data.isna().sum()
data.gender.value_counts()
#we decide that age col null values for mean value
data.age.fillna(np.mean(data["age"]),inplace=True)

data.age.isna().sum()
data.head()
#convert categorical var to numeric i.e gender
data.gender.value_counts()
def g_to_num(x):

    if x=="F":

        return 0

    if x=="M":

        return 1

    else:

        return 3
data["gender"]=data["gender"].apply(g_to_num)
data.gender.value_counts()
#kmeans clustr

from sklearn.cluster import KMeans
model=KMeans(n_clusters=3,random_state=0)#assume three cluster
model.fit(data)#train the model
len(model.predict(data))
len(model.labels_)
data["group"]=model.predict(data)#create new col group then prediction of cluster
data[["group"]].head(50)#no of cluster in which group
data.head()
print("clusters of each group ",model.cluster_centers_)#clusters
model.labels_#cluster groups data point belong to which group
model.inertia_
data.groupby(["group"]).age.count()
data.gender.value_counts()
a=data.groupby(["group","gender"]).age.count().reset_index()

a
var=data.columns[2:39]
data.groupby(["group"])[var].mean()#clusters centers
group3=data.groupby(["group"])[var].mean().T

group3#clusters center of each row
data.groupby(["group"])[var].agg(["count","mean","sum","max","min"]).T
#group3.to_csv("C:\machine learning\cluster3.csv")

data.groupby(["group"]).age.count()
sns.barplot(x=np.arange(0,3,1),y=data.groupby(["group"]).age.count())
data.groupby(["group"]).friends.mean()
sns.barplot(x=np.arange(0,3,1),y=data.groupby(["group"]).friends.mean())

plt.xlabel("no of groups")

plt.ylabel("playing customers(friends)")

plt.title("friends sewgmentation") 
data.columns
data.groupby(["group"]).basketball.mean()
sns.barplot(x=np.arange(0,3,1),y=data.groupby(["group"]).basketball.mean())

plt.xlabel("no of groups")

plt.ylabel("playing customers(basketball)")

plt.title("basketball sewgmentation")
data.groupby(["group"]).mean()["football"]
sns.barplot(x=np.arange(0,3,1),y=data.groupby(["group"]).football.mean())

plt.xlabel("no of groups")

plt.ylabel("playing customers(football)")

plt.title("football sewgmentation")
data.groupby(["group"]).soccer.mean()



sns.barplot(x=np.arange(0,3,1),y=data.groupby(["group"]).soccer.mean())

plt.xlabel("no of groups")

plt.ylabel("playing customers(soccer)")

plt.title("soccer sewgmentation")
sns.barplot(x=np.arange(0,3,1),y=data.groupby(["group"])['softball'].mean())

plt.xlabel("no of groups")

plt.ylabel("playing customers(softball)")

plt.title("softball sewgmentation")
names
sns.barplot(x=np.arange(0,3,1),y=data.groupby(["group"])['dance'].mean())


    
#data['gender'] = data['gender_F'].apply(gender_to_numeric)
data.gender.value_counts()
count_dance=data.groupby(["group","gender"]).dance.count().reset_index()#count gender
len(count_dance.gender)*100
g1_male=data[(data.group==0)&(data.gender==1)]#provide data where group belong 0 and gender male
len(g1_male)
len(data[data.group==0])#1403 peoples belog to 0 group
count=data[data.group==0]
cdance=count.groupby("group").dance.count()
gen_co_dan=count["gender"].value_counts()#992 females and 316 are males and not disclosed 95
per_dan=count["gender"].value_counts()/len(count.gender)*100
dance_per=pd.concat([gen_co_dan,per_dan],axis=1,keys=["count","percentage"])

dance_per
print("total dance students in group 0 ",cdance)

print(" dance students in group 0 counts male and female\n",gen_co_dan)#0 for female and 1 for male and 3 for not disclosed

print("0 for female and 1 for male and 3 for not disclosed")

print("percentage of dance students ")

dance_per
data.columns
a
data.groupby(["group"]).friends.count()
data.columns
counteach=data.groupby(["group","gender"])[names].count().T

counteach
counteach.to_csv("C:\machine learning\countmalewach.csv")
aggregate=data.groupby(["group"])[names].agg(["mean","sum","max","min"]).T

aggregate
aggregate.to_csv("C:\machine learning\ggregate.csv")
ma=data.groupby(["group"])[["football"]].mean()

a=ma.reset_index()

a.nlargest(1,columns="football")[["group"]]#mean is maximum belongs to in which group
play=data.groupby(["group"])[names].mean().reset_index()

play1=play.iloc[:,1:].columns
percentage=(play[play1]/len(play))*100

percentage#percentage of every group
percentage.sum()
names
football=play.nlargest(1,columns="football")[["football","group"]]#provide data those group has play max football

football#average of foot ball is max
#play=data.groupby(["group"])["baseball"].mean().reset_index()
basket=play.nlargest(1,columns="baseball")[["baseball","group"]]

basket
football.to_csv("C:\machine learning\Football1.csv")
play.nlargest(1,columns=["volleyball"])[["volleyball","group"]]
play.nlargest(1,columns=["soccer"])[["soccer","group"]]
play.nlargest(1,columns=["softball"])[["softball","group"]]
play.columns
feature=data.iloc[:,0:40].columns#independent var

feature
feature=list(data.drop("group",axis=1))

#target="Exited"

feature

target="group"

target


from sklearn.model_selection import train_test_split
train,test=train_test_split(data,test_size=0.3,random_state=0)

train

test.shape

train[feature]

test[target]
from sklearn.linear_model import LogisticRegression#apply package
model=LogisticRegression(penalty="l2",random_state=0)
model.fit(train[feature],train[target])#train the model or fitting the model
pred=model.predict(test[feature])#prediction of model
print("coffecient of model: ",model.coef_)#coffecient of both groups

print("intercept of model: ",model.intercept_)
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, accuracy_score,precision_score,recall_score, classification_report, confusion_matrix,plot_confusion_matrix



tr=model.predict(train[feature])

train1=confusion_matrix(train[target],tr)#prediction of confusion metrices
train1
test1=confusion_matrix(test[target],pred)

test1
percentage_test=test[target].value_counts()/len(test[target])*100

percentage_test
print(len(test[test[target]==0]))#actual gp1: 418

print(len(test[test[target]==1]))#actual gp2 : 831

print(len(test[test[target]==2]))#actual gp3: 121

#test[target][test[target]==0]
print(len(pred[pred==0]))#predicted gp1 417

print(len(pred[pred==1]))#predicted 831

print(len(pred[pred==2]))# predicted 122
plt.figure(figsize=(9,9))



sns.heatmap(test, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');



plt.ylabel('Actual values');

plt.xlabel('Predicted values');

plt.title("confusion metrices under testing")

plt.show()
accuracy=accuracy_score(test[target],pred)#accuracy of model

#precision=precision_score(test[target],pred)

#recall=recall_score(test[target],pred)

#print("accuracy of model: ",accuracy*100)

#print("precision of model: ",precision*100)

#print("recall of model: ",recall*100)
print("accuracy of model : ",accuracy_score(test[target],pred)*100)