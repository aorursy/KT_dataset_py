



import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib.pyplot as plt





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



df = pd.read_csv('/kaggle/input/xAPI-Edu-Data/xAPI-Edu-Data.csv')

df.head()


df.isnull().values.any()


#sütun bilgilerine ulaşıldı

df.info()
df['StageID'].value_counts()
fig= plt.subplots(figsize=(10,7))

sns.countplot(x='gender',data=df,palette="pastel" )

plt.show()
fig= plt.subplots(figsize=(15,7))

sns.countplot(x='gender', hue='Class', data=df,hue_order = ['L', 'M', 'H'],palette="pastel" )

plt.show()
f=plt.figure(figsize=(10,7))

sns.countplot(x="Relation", data=df, palette="pastel");

plt.show()


fig= plt.subplots(figsize=(15,7))

sns.countplot(x='Relation', hue='Class', data=df,hue_order = ['L', 'M', 'H'],palette="pastel" )

plt.show()
from matplotlib.pyplot import pie

plt.figure(figsize = (7,7))

colors=["wheat","aquamarine"]

count = df.ParentAnsweringSurvey.groupby(df.ParentAnsweringSurvey).count()

Answer = count.keys()

pie(count,labels=Answer, colors=colors, autopct='%1.1f%%')
from matplotlib.pyplot import pie

plt.figure(figsize = (7,7))

colors=["wheat","aquamarine"]

count = df.ParentschoolSatisfaction.groupby(df.ParentschoolSatisfaction).count()

Satisfaction = count.keys()

pie(count,labels=Satisfaction, colors=colors, autopct='%1.1f%%')
from matplotlib.pyplot import pie

group_by_sum_of_nationalities = df.NationalITy.groupby(df.NationalITy).count()

group_by_sum_of_nationalities_header = group_by_sum_of_nationalities.keys()

plt.figure(figsize = (15,12))

colors=['#c2c2f0','#ffb3e6',"wheat","aquamarine",'#ff9999','#66b3ff','#99ff99','#ffcc99','#ff9999']

#print(group_by_sum_of_nationalities_header)

pie(group_by_sum_of_nationalities,labels=group_by_sum_of_nationalities_header, colors=colors, autopct='%1.1f%%')
pd.DataFrame(df['raisedhands'].loc[0:481]).astype(int).sum()
fig, (axis1, axis2)  = plt.subplots(2, 1,figsize=(20,15))

sns.countplot(x='Topic', hue='gender', data=df, ax=axis1, palette="pastel")

sns.countplot(x='NationalITy', hue='gender', data=df, ax=axis2, palette="pastel")
f=plt.figure(figsize=(20,7))

sns.countplot(x='GradeID', hue='Class', data=df, order=['G-02', 'G-04', 'G-05', 'G-06', 'G-07', 'G-08', 'G-09', 'G-10', 'G-11', 'G-12'], hue_order = ['L', 'M', 'H'], palette="pastel")

plt.show()
fig, axarr  = plt.subplots(2,2,figsize=(20,10))

sns.barplot(x='Class', y='VisITedResources', data=df, order=['L','M','H'], ax=axarr[0,0],palette="pastel")

sns.barplot(x='Class', y='AnnouncementsView', data=df, order=['L','M','H'], ax=axarr[0,1],palette="pastel")

sns.barplot(x='Class', y='raisedhands', data=df, order=['L','M','H'], ax=axarr[1,0],palette="pastel")

sns.barplot(x='Class', y='Discussion', data=df, order=['L','M','H'], ax=axarr[1,1],palette="pastel")
fig, (axis1, axis2)  = plt.subplots(1, 2,figsize=(20,7))

sns.regplot(x='raisedhands', y='VisITedResources', data=df, ax=axis1)

sns.regplot(x='AnnouncementsView', y='Discussion', data=df, ax=axis2)
df_selected = df.drop(['NationalITy','StageID','SectionID','PlaceofBirth', 'GradeID', "Topic", 'Semester', "Relation", 'ParentAnsweringSurvey', "ParentschoolSatisfaction", 'StudentAbsenceDays', "Class"], 

                        axis = 1)
df_selected
df_selected_copy=df_selected
df_selected_copy.gender=[1 if i=="M" else 0 for i in df_selected_copy.gender]
df_selected_copy
y=df_selected_copy.gender.values

x_df=df_selected_copy.drop("gender",axis=1)
y
x=(x_df-np.min(x_df))/(np.max(x_df)-np.min(x_df))
x
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=52)
from sklearn.svm import SVC



svm=SVC(random_state=1)

svm.fit(x_train,y_train)

print("test accuracy is: ",svm.score(x_test,y_test))
from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()

lr.fit(x_train,y_train)

print("test accuracy is {}".format(lr.score(x_test,y_test)))
from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier(n_estimators=100,random_state=1)

rf.fit(x_train,y_train)

print("test accuracy is: ",rf.score(x_test,y_test))
from sklearn.naive_bayes import GaussianNB

nb=GaussianNB()

nb.fit(x_train,y_train)

print("test accuracy is: ",nb.score(x_test,y_test))
from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier()

dt.fit(x_train,y_train)



print("Atest accuracy is: " ,dt.score(x_test,y_test))