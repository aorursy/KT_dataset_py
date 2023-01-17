# importing packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import roc_curve
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# importing the Dataset

a=pd.read_csv("../input/Loan payments data.csv")
# Information of the Dataset

a.info()
# First 5 rows of the Dataset

a.head()
# Cleaning and transforming the data for futher data analysis

list_edu=[]
for i in a['education']:
    list_edu.append(i)
l=LabelEncoder()
l.fit(list_edu)
l1=l.transform(list_edu)

list_gen=[]
for i in a['Gender']:
    list_gen.append(i)
l.fit(list_gen)
l2=l.transform(list_gen)

list_loan_stat=[]
for i in a['loan_status']:
    list_loan_stat.append(i)
l.fit(list_loan_stat)
l3=l.transform(list_loan_stat)

new_a=pd.DataFrame({'loan_status':l3,'Principal':a['Principal'],'terms':a['terms'],'age':a['age'],'education':l1,'Gender':l2,'loan_status':a['loan_status'],'past_due_days':a['past_due_days']})
new_a['past_due_days']=new_a['past_due_days'].fillna(0)
new_a
# Data Visualization

plt.figure(figsize=(12,8))
sns.countplot( x = "loan_status",hue = "education", data=a)
plt.show()
plt.figure(figsize=(12,8))
sns.countplot( x = "loan_status",hue = "Gender", data=a)
plt.show()
plt.figure(figsize=(12,8))
sns.countplot( x = "age",hue = "loan_status", data=a)
plt.show()
x=new_a.drop('loan_status',axis=1).values
y=new_a['loan_status']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=21)
knn=KNeighborsClassifier(n_neighbors=6)
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)
knn_test=knn.score(x_test,y_test)
knn_train=knn.score(x_train,y_train)
cv_results1=cross_val_score(knn,x,y,cv=6)
print(cv_results1)
gnb=GaussianNB()
gnb.fit(x_train,y_train)
y_pred=gnb.predict(x_test)
gnb_test=gnb.score(x_test,y_test)
gnb_train=gnb.score(x_train,y_train)
cv_results2=cross_val_score(gnb,x,y,cv=5)
print(cv_results2)
mnb=MultinomialNB()
mnb.fit(x_train,y_train)
y_pred=mnb.predict(x_test)
mnb_test=mnb.score(x_test,y_test)
mnb_train=mnb.score(x_train,y_train)
cv_results3=cross_val_score(gnb,x,y,cv=5)
print(cv_results3)
bnb=BernoulliNB()
bnb.fit(x_train,y_train)
y_pred=bnb.predict(x_test)
bnb_test=bnb.score(x_test,y_test)
bnb_train=bnb.score(x_train,y_train)
cv_results4=cross_val_score(gnb,x,y,cv=5)
print(cv_results4)
clf=SVC(C=100)
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
clf_test=clf.score(x_test,y_test)
clf_train=clf.score(x_train,y_train)
cv_results5=cross_val_score(clf,x,y,cv=5)
print(cv_results5)
rnd=RandomForestClassifier(n_estimators=100)
rnd.fit(x_train,y_train)
y_pred=rnd.predict(x_test)
rnd_test=rnd.score(x_test,y_test)
rnd_train=rnd.score(x_train,y_train)
cv_results6=cross_val_score(clf,x,y,cv=5)
print(cv_results6)
x=[1.5,2.5,3.5,4.5,5.5,6.5]
y=[knn.score(x_test,y_test),gnb.score(x_test,y_test),mnb.score(x_test,y_test),bnb.score(x_test,y_test),clf.score(x_test,y_test),rnd.score(x_test,y_test)]
plt.style.use('ggplot')
plt.figure(figsize=(12,8))
plt.bar(x,y,color='blue')
plt.xticks(range(2,8),['KNN','Gaussian NB','Multinomial NB','Binomial NB','SVC','Random Forest'])
plt.xlabel('Classifiers')
plt.ylabel('Score')
plt.show()
# Model Comparision

df = pd.DataFrame({'Model':['K-Nearest Neighbors','Gaussian NB','Multinomial NB','Binomial NB','SVM','Random Forest'],
                  'Test_Score':[knn_test,gnb_test,mnb_test,bnb_test,clf_test,rnd_test],
                  'Train_Score':[knn_train,gnb_train,mnb_train,bnb_train,clf_train,rnd_train]})
df.sort_values(by='Test_Score',ascending=False)