import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
%matplotlib inline
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier 
from sklearn import metrics 
from sklearn.metrics import accuracy_score
df = pd.read_csv("breastcancer.csv")
df.head()
df.drop("id",axis=1,inplace=True)
features_mean= list(df.columns[1:11])
features_se= list(df.columns[11:21])
features_worst=list(df.columns[21:31])
print(features_mean)
print("-----------------------------------")
print(features_se)
print("------------------------------------")
print(features_worst)
df['diagnosis']=df['diagnosis'].map({'M':1,'B':0})
df.head()
sns.countplot(df['diagnosis'])
plt.scatter(df['area_mean'],df['perimeter_mean'])
corr = df[features_mean].corr() # .corr is used for find corelation
plt.figure(figsize=(8,8))
sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 15},
           xticklabels= features_mean, yticklabels= features_mean,
           cmap= 'Blues') 
df1=df.drop(features_worst,axis=1)
cols = ['perimeter_mean','perimeter_se','area_mean','area_se']
df1 = df1.drop(cols, axis=1)
cols = ['concavity_mean',
        'concavity_se', 
        'concave points_mean', 
        'concave points_se']
df1 = df1.drop(cols, axis=1)
df1.head()
useful_features=list(df1.columns[1:])
print(useful_features)
corr1=df.corr()
corr1.nlargest(20,['diagnosis'])['diagnosis']
x=df1
y=df1['diagnosis']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=40)
print(len(x_train),'\t',len(x_test))
model=LogisticRegression()
model.fit(x_train[useful_features],y_train)
pred=model.predict(x_test[useful_features])
print(model.score(x_train[useful_features],y_train)*100)
accuracy=metrics.accuracy_score(pred,y_test)
print("Accuracy : %s" % "{0:.3%}".format(accuracy))
xx =df[[ 'concave points_worst', 'perimeter_worst', 
       'concave points_mean', 'radius_worst',
       'perimeter_mean', 'area_worst',
       'radius_mean', 'area_mean',
       'concavity_mean', 'concavity_worst',
       'compactness_mean', 'compactness_worst',
       'radius_se', 'area_se', 'perimeter_se']]
yy = df['diagnosis'] 
xx_train, xx_test, yy_train, yy_test = train_test_split(xx, yy, test_size=0.2, random_state=1)
model2=LogisticRegression()
model2.fit(xx_train,yy_train)
print(model2.score(xx_train,yy_train)*100)
pred=model2.predict(xx_test)
print(pred)
print(metrics.accuracy_score(pred,yy_test)*100)
model2=LogisticRegression()
model2.fit(xx_test,yy_test)
print(model2.score(xx_test,yy_test)*100)
model3=RandomForestClassifier()
model3.fit(xx_train,yy_train)
pred=model3.predict(xx_test)
pred
print(model3.score(xx_train,yy_train)*100) #trainset accuracy
model3=RandomForestClassifier()
model3.fit(xx_test,yy_test)
print(model3.score(xx_test,yy_test)*100) #testset accuracy
print(metrics.accuracy_score(pred,yy_test)*100)
