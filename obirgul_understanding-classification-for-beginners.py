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
df.info()
df.describe()
df.columns
df.isnull().sum()
df.rename(index=str, columns={'gender':'Gender', 

                              'NationalITy':'Nationality',

                              'raisedhands':'RaisedHands',

                              'VisITedResources':'VisitedResources'},

                               inplace=True)

df.columns
# Exploring nationalities

nationality = sns.countplot(x='Nationality', data=df ,palette='coolwarm')

nationality.set(xlabel='Nationality', ylabel='Count', title='Nationality')

plt.setp(nationality.get_xticklabels(), rotation=60)

plt.show()
gender = sns.countplot(x='Class', hue='Gender', data=df, palette='coolwarm')

gender.set(xlabel='Class', ylabel='Count', title='Gender comparison')

plt.show()
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

x.head(3)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=52)
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix



#Naive Bayes

nb =  GaussianNB()

nb.fit(x_train, y_train)

y_pred=nb.predict(x_test)



print("Accuracy of naive bayees algorithm: ",nb.score(x_test,y_test))
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
knn = KNeighborsClassifier(n_neighbors=9)

knn.fit(x_train,y_train)

y_pred=knn.predict(x_test)



print('KNN (K=9) accuracy is: ',knn.score(x_test,y_test))
from sklearn.linear_model import SGDClassifier

sgd =  SGDClassifier(loss='modified_huber', shuffle=True,random_state=101)

sgd.fit(x_train, y_train)

y_pred=sgd.predict(x_test)

print(sgd.score(x_test,y_test))

print('Classification report: \n',classification_report(y_test,y_pred))
from sklearn.svm import SVC



svm=SVC(random_state=1)

svm.fit(x_train,y_train)



#accuracy

print("accuracy of svm algorithm: ",svm.score(x_test,y_test))
from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier()

dt.fit(x_train,y_train)



print("Accuracy score for Decision Tree Classification: " ,dt.score(x_test,y_test))
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
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=75, oob_score=True, n_jobs=-1, random_state=101, max_features = 0.5 , min_samples_leaf = 50)

rf.fit(x_train,y_train)

y_pred = rf.predict(x_test)

print(rf.score(x_test,y_test))



rf_cm = confusion_matrix(y_test,y_pred)

print('Confusion matrix: \n',rf_cm)



print('Classification report: \n',classification_report(y_test,y_pred))
y=df.Relation.values

x=df.drop("Relation",axis=1)

np.unique(y)
x = pd.get_dummies(data = x, columns = ['Class'] , prefix = ['Class'] , drop_first = False)

x = pd.get_dummies(data = x, columns = ['StudentAbsenceDays'] , prefix = ['StudentAbsenceDays'] , drop_first = False)

x = pd.get_dummies(data = x, columns = ['ParentschoolSatisfaction'] , prefix = ['ParentschoolSatisfaction'] , drop_first = False)

#x = pd.get_dummies(data = x, columns = ['Relation'] , prefix = ['Relation'] , drop_first = False)

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
#Naive Bayes

nb =  GaussianNB()

nb.fit(x_train, y_train)

y_pred=nb.predict(x_test)

gaussian_accuracy = nb.score(x_test,y_test)

print("Accuracy of naive bayees algorithm: ",gaussian_accuracy)
error_rate = []

for i in range(1,100):    

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(x_train,y_train)

    pred_i = knn.predict(x_test)

    error_rate.append(np.mean(pred_i != y_test))



plt.figure(figsize=(8,4))

plt.plot(range(1,100),error_rate,color='darkred', marker='o',markersize=10)

plt.title('Error Rate vs. K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')
knn = KNeighborsClassifier(n_neighbors=98)

knn.fit(x_train,y_train)

y_pred=knn.predict(x_test)

knn_acc = knn.score(x_test,y_test)

print('KNN (K=98) accuracy is: ',knn_acc)
svm=SVC(random_state=1)

svm.fit(x_train,y_train)

svm_pred = svm.score(x_test,y_test)

#accuracy

print("accuracy of svm algorithm: ",svm_pred)
dt=DecisionTreeClassifier()

dt.fit(x_train,y_train)

dt_acc = dt.score(x_test,y_test)

#print('Classification report: \n',classification_report(y_test,y_pred))

print("Accuracy score for Decision Tree Classification: " , dt_acc)
from sklearn.linear_model import SGDClassifier

sgd =  SGDClassifier(loss='modified_huber', shuffle=True,random_state=101)

sgd.fit(x_train, y_train)

y_pred=sgd.predict(x_test)

print(sgd.score(x_test,y_test))

print('Classification report: \n',classification_report(y_test,y_pred))
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

    model = RandomForestClassifier(n_estimators=30, oob_score=True, n_jobs=-1, random_state=101, max_features = trees)

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    accuracy = np.mean(y_test==y_pred)

    results.append(accuracy)



plt.figure(figsize=(8,4))

pd.Series(results, max_features_options).plot(kind="bar",color="darkred",ylim=(0.7,0.9))
results = []

min_samples_leaf_options = [5,10,15,20,25,30,35,40,45,50]

for trees in min_samples_leaf_options:

    model = RandomForestClassifier(n_estimators=30, oob_score=True, n_jobs=-1, random_state=101, max_features = None, min_samples_leaf = trees)

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    accuracy = np.mean(y_test==y_pred)

    results.append(accuracy)



plt.figure(figsize=(8,4))

pd.Series(results, min_samples_leaf_options).plot(color="darkred",marker="o")
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=30, oob_score=True, n_jobs=-1, random_state=101, max_features = None , min_samples_leaf = 25)

rf.fit(x_train,y_train)

y_pred = rf.predict(x_test)

print(rf.score(x_test,y_test))



rf_cm = confusion_matrix(y_test,y_pred)

print('Confusion matrix: \n',rf_cm)



print('Classification report: \n',classification_report(y_test,y_pred))
from xgboost import XGBClassifier

results = []

max_depth = [x for x in range(1,10)]

for depth in max_depth:

    xgb = XGBClassifier(max_depth= depth ,n_estimators= 300 , objective="binary:logistic")

    xgb.fit(x_train, y_train)

    y_pred_xgb = xgb.predict(x_test)

    accuracy = np.mean(y_test==y_pred_xgb)

    results.append(accuracy)



plt.figure(figsize=(8,4))

pd.Series(results, max_depth).plot(color="darkred",marker="o")
results = []

n_estimators = [x for x in range(100,500,20)]

for estimators in n_estimators:

    xgb = XGBClassifier(max_depth=7,n_estimators= estimators , objective="binary:logistic")

    xgb.fit(x_train, y_train)

    y_pred_xgb = xgb.predict(x_test)

    accuracy = np.mean(y_test==y_pred_xgb)

    results.append(accuracy)



plt.figure(figsize=(8,4))

pd.Series(results, n_estimators).plot(color="darkred",marker="o")
xgb = XGBClassifier(max_depth=7,n_estimators=220, objective="binary:logistic")

xgb.fit(x_train, y_train)

y_pred_xgb = xgb.predict(x_test)



print('Classification report: \n',classification_report(y_test,y_pred_xgb))

#predictions = [round(value) for value in y_pred_xgb]

#accuracy_xgb = accuracy_score(y_test, predictions)



xgb_x_test = x_test

xgb_cm = confusion_matrix(y_test,y_pred_xgb)

xgb_cm
#Logistic Regression

from sklearn.linear_model import LogisticRegression

#fit

lr=LogisticRegression()

lr.fit(x_train,y_train)

log_acc = lr.score(x_test,y_test)



#confision matrix

lr_cm = confusion_matrix(y_test,y_pred)

log_reg_x_test = x_test

#accuracy

print("test accuracy is {}", log_acc)

lr_cm
# visualize with seaborn library

sns.heatmap(xgb_cm,annot=True,fmt="d") 

plt.show()
# print the first 10 predicted probabilities of class membership

lr.predict_proba(xgb_x_test)[0:10]
arr = lr.predict_proba(xgb_x_test)



# Creating pandas dataframe from numpy array

pred_val = pd.DataFrame({'Predicted_Father': arr[:, 0], 'Predicted_Mum': arr[:, 1]})

pred_val
#pred_val.to_csv(r'\predicted_values.csv')