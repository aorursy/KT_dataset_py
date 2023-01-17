import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv("../input/xAPI-Edu-Data/xAPI-Edu-Data.csv")

df
df.head()
df.shape
df.info()
df.describe()
df.isnull().sum()
df.rename(index=str, columns={'gender':'Gender', 

                              'NationalITy':'Nationality',

                              'raisedhands':'RaisedHands',

                              'VisITedResources':'VisitedResources'},

                               inplace=True)

df.columns
for i in range(1,17):

    print(df.iloc[:,i].value_counts())

    print("*"*20)
print("Class Unique Values : ", df["Class"].unique())

print("Topic Unique Values : ", df["Topic"].unique())

print("StudentAbsenceDays Unique Values : ", df["StudentAbsenceDays"].unique())

print("ParentschoolSatisfaction Unique Values : ", df["ParentschoolSatisfaction"].unique())

print("Relation Unique Values : ", df["Relation"].unique())

print("SectionID Unique Values : ", df["SectionID"].unique())

print("Gender Unique Values : ", df["Gender"].unique())
nationality = sns.countplot(x='Nationality', data=df ,palette='Set1')

nationality.set(xlabel='Ülkeler', ylabel='Kişi Sayısı', title='Kişilerin Ülkelere Dağılımı')

plt.setp(nationality.get_xticklabels(), rotation=60)

plt.show()
gender = sns.countplot(x='Class', hue='Gender', data=df, palette='rainbow')

gender.set(xlabel='Sınıf', ylabel='Kişi Sayısı', title='Sınıflardaki Cinsiyet Dağılımı')

plt.show()
sns.scatterplot(x="RaisedHands", y="VisitedResources", hue = 'Gender', data=df)
y=df.Class.values

x=df.drop("Class",axis=1)

np.unique(y)
#x = pd.get_dummies(data = x, columns = ['Class'] , prefix = ['Class'] , drop_first = False)

x = pd.get_dummies(data = x, columns = ['StudentAbsenceDays'] , prefix = ['StudentAbsenceDays'] , drop_first = False)

x = pd.get_dummies(data = x, columns = ['ParentschoolSatisfaction'] , prefix = ['ParentschoolSatisfaction'] , drop_first = False)

x = pd.get_dummies(data = x, columns = ['Relation'] , prefix = ['Relation'] , drop_first = False)

x = pd.get_dummies(data = x, columns = ['SectionID'] , prefix = ['SectionID'] , drop_first = False)

x = pd.get_dummies(data = x, columns = ['Gender'] , prefix = ['Gender'] , drop_first = False)

x = pd.get_dummies(data = x, columns = ['PlaceofBirth'] , prefix = ['PlaceofBirth'] , drop_first = False)

x = pd.get_dummies(data = x, columns = ['Semester'] , prefix = ['Semester'] , drop_first = False)

x = pd.get_dummies(data = x, columns = ['ParentAnsweringSurvey'] , prefix = ['ParentAnsweringSurvey'] , drop_first = False)

x = pd.get_dummies(data = x, columns = ['GradeID'] , prefix = ['GradeID'] , drop_first = False)

x = pd.get_dummies(data = x, columns = ['StageID'] , prefix = ['StageID'] , drop_first = False)

x = pd.get_dummies(data = x, columns = ['Topic'] , prefix = ['Topic'] , drop_first = False)

x = pd.get_dummies(data = x, columns = ['Nationality'] , prefix = ['Nationality'] , drop_first = False)

x.head()
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=52)
from sklearn.linear_model import LogisticRegression

lr=LogisticRegression(random_state=0)

lr.fit(x_train, y_train)
y_pred=lr.predict(x_test)

print(y_pred)
print("test accuracy is {}".format(lr.score(x_test,y_test)))
#confusion matrix

#18 ve 50 doğru bilinen değerler, diğer köşegenler yanlış bilinen değerlerin sayısı 

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred)

print(cm)
from sklearn.neighbors import KNeighborsClassifier
error_rate = []

for i in range(1,51):    

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(x_train,y_train)

    pred_i = knn.predict(x_test)

    error_rate.append(np.mean(pred_i != y_test))



plt.figure(figsize=(8,4))

plt.plot(range(1,51),error_rate,color='darkred', marker='o',markersize=10)

plt.title('Error Rate vs. K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')
knn=KNeighborsClassifier(n_neighbors=8, metric='minkowski')

knn.fit(x_train,y_train)

y_pred=knn.predict(x_test)

print(y_pred)
cm=confusion_matrix(y_test,y_pred)

print(cm)
print("test accuracy is {}".format(knn.score(x_test,y_test)))
from sklearn.metrics import classification_report

from sklearn.linear_model import SGDClassifier

sgd =  SGDClassifier(loss='modified_huber', shuffle=True,random_state=101)

sgd.fit(x_train, y_train)

y_pred=sgd.predict(x_test)

print(sgd.score(x_test,y_test))

print('Classification report: \n',classification_report(y_test,y_pred))
from sklearn.svm import SVC

svm=SVC(kernel='linear', random_state=1)

svm.fit(x_train,y_train)

 
y_pred=svm.predict(x_test)

print(y_pred)
cm=confusion_matrix(y_test,y_pred)

print(cm)
print("accuracy of svm algorithm: ",svm.score(x_test,y_test))
from sklearn.naive_bayes import GaussianNB

gnb=GaussianNB()

gnb.fit(x_train,y_train)
y_pred=gnb.predict(x_test)
cm=confusion_matrix(y_test,y_pred)

print(cm)
print("Accuracy of naive bayees algorithm: ",gnb.score(x_test,y_test))
from sklearn.tree import DecisionTreeClassifier

dtc=DecisionTreeClassifier()

dtc.fit(x_train,y_train)
y_pred=dtc.predict(x_test)

cm=confusion_matrix(y_test,y_pred)

print(cm)
print("Accuracy of decision tree algorithm: ",dtc.score(x_test,y_test))
from sklearn.ensemble import RandomForestClassifier

results = []

n_estimator_options = [20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]

for trees in n_estimator_options:

    model = RandomForestClassifier(trees, oob_score=True, n_jobs=-1, random_state=101)

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    accuracy = np.mean(y_test==y_pred)

    results.append(accuracy)



plt.figure(figsize=(8,4))

pd.Series(results, n_estimator_options).plot(color="darkred",marker="o")
results = []

max_features_options = ['auto',None,'sqrt',0.95,0.75,0.5,0.25,0.10]

for trees in max_features_options:

    model = RandomForestClassifier(n_estimators=75, oob_score=True, n_jobs=-1, random_state=101, max_features = trees)

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    accuracy = np.mean(y_test==y_pred)

    results.append(accuracy)



plt.figure(figsize=(8,4))

pd.Series(results, max_features_options).plot(kind="bar",color="darkred",ylim=(0.7,0.9))

results = []

min_samples_leaf_options = [5,10,15,20,25,30,35,40,45,50]

for trees in min_samples_leaf_options:

    model = RandomForestClassifier(n_estimators=75, oob_score=True, n_jobs=-1, random_state=101, max_features = 0.5, min_samples_leaf = trees)

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    accuracy = np.mean(y_test==y_pred)

    results.append(accuracy)



plt.figure(figsize=(8,4))

pd.Series(results, min_samples_leaf_options).plot(color="darkred",marker="o")
rf = RandomForestClassifier(n_estimators=75, oob_score=True, n_jobs=-1, random_state=101, max_features = 0.5 , min_samples_leaf = 50)

rf.fit(x_train,y_train)

y_pred = rf.predict(x_test)

print(rf.score(x_test,y_test))



rf_cm = confusion_matrix(y_test,y_pred)

print('Confusion matrix: \n',rf_cm)



print('Classification report: \n',classification_report(y_test,y_pred))
from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier(n_estimators=10, criterion='entropy')

rfc.fit(x_train,y_train)
y_pred=rfc.predict(x_test)

y_proba=rfc.predict_proba(x_test)
cm=confusion_matrix(y_test,y_pred)

print(cm)

print(y_proba[:,0]) 
print("Accuracy of Random Forest algorithm: ",rfc.score(x_test,y_test))