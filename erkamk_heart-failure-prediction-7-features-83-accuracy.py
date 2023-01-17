

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv("/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv")

data.dropna(inplace = True)
data.info()
data.describe()
columns = data.columns

data.columns
print(data.anaemia.value_counts())

labels = '0', '1',

sizes = [170, 129]

colors = ['darkkhaki','darkcyan']

explode = (0, 0)

fig1, ax1 = plt.subplots(figsize =(10,10))

ax1.pie(sizes,colors = colors ,explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')

plt.title("Anaemia")

plt.show()
print(data.diabetes.value_counts())

labels = '0', '1',

sizes = [174, 125]

colors = ['forestgreen','darkmagenta']

explode = (0, 0)

fig1, ax1 = plt.subplots(figsize =(10,10))

ax1.pie(sizes,colors = colors ,explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')

plt.title("Diabetes")

plt.show()
print(data.high_blood_pressure.value_counts())

labels = '0', '1',

sizes = [194, 105]

colors = ['silver','firebrick']

explode = (0, 0)

fig1, ax1 = plt.subplots(figsize =(10,10))

ax1.pie(sizes,colors = colors ,explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')

plt.title("High Blood Pressure")

plt.show()
print(data.sex.value_counts())

labels = 'Female', 'Male',

sizes = [105, 194]

colors = ['khaki','mediumslateblue']

explode = (0, 0)

fig1, ax1 = plt.subplots(figsize =(10,10))

ax1.pie(sizes,colors = colors ,explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')

plt.title("Gender")

plt.show()
print(data.smoking.value_counts())

labels = '0', '1',

sizes = [203, 96]

colors = ['turquoise','peru']

explode = (0, 0)

fig1, ax1 = plt.subplots(figsize =(10,10))

ax1.pie(sizes,colors = colors ,explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')

plt.title("Smoking")

plt.show()
print(data.DEATH_EVENT.value_counts())

labels = '0', '1',

sizes = [203, 96]

colors = ['whitesmoke','crimson']

explode = (0, 0)

fig1, ax1 = plt.subplots(figsize =(10,10))

ax1.pie(sizes,colors = colors ,explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')

plt.title("Death Event")

plt.show()
plt_columns = ['age', 'creatinine_phosphokinase',

       'ejection_fraction', 'platelets',

       'serum_creatinine', 'serum_sodium', 'time'

       ]

for each in plt_columns:

    fig1, ax1 = plt.subplots(figsize =(10,10))

    plt.hist(data[each], bins=50,color = "darkmagenta")

    plt.xlabel(each)

    plt.ylabel("Frequency")

    plt.grid()

    plt.show()
f, ax = plt.subplots(figsize =(15,10))

corrMatrix = data.corr()

sns.heatmap(corrMatrix, annot=True)

plt.show()
y = data.DEATH_EVENT

data.drop(["DEATH_EVENT" ,"anaemia" , "diabetes" , "high_blood_pressure" ,"sex" , "smoking" ], axis = 1, inplace = True)

x = (data - np.min(data)) / (np.max(data) - np.min(data)) 
from sklearn.model_selection import train_test_split

x_train , x_test , y_train, y_test = train_test_split(x,y,test_size = 0.2 , random_state = 42)

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV

grid = {"n_neighbors" : np.arange(1,50)}

knn = KNeighborsClassifier()

knn_cv = GridSearchCV(knn,grid , cv = 10)

knn_cv.fit(x_train,y_train)

print(knn_cv.best_params_)

print(knn_cv.best_score_)
knn1 = KNeighborsClassifier(n_neighbors = 3)

knn1.fit(x_train,y_train)

knn_max = knn1.score(x_test,y_test)

print("K-NN max score : ",knn1.score(x_test,y_test))
y_pred = knn1.predict(x_test)

y_true = y_test

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true,y_pred)

f, ax = plt.subplots(figsize =(10,10))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="blue",fmt = ".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()
from sklearn.linear_model import LogisticRegression

grid = {"C" : np.logspace(-3,3,7) , "penalty" : ["l1","l2"]}

lr = LogisticRegression()

lr_cv = GridSearchCV(lr,grid , cv = 10)

lr_cv.fit(x_train,y_train)

print("best parameters : ",lr_cv.best_params_)

print("max score : ",lr_cv.best_score_)
lr2 = LogisticRegression(C = 10.0 , penalty = "l2")

lr2.fit(x_train,y_train)

print("Logistic Regression Max Score : ",lr2.score(x_test,y_test))

lr_max = lr2.score(x_test,y_test)
y_pred = lr2.predict(x_test)

y_true = y_test

cm = confusion_matrix(y_true,y_pred)

f, ax = plt.subplots(figsize =(10,10))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="blue",fmt = ".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()
score_list = []

from sklearn.ensemble import RandomForestClassifier

for each in range (1,100):

    rf = RandomForestClassifier(n_estimators = each,random_state = 41)

    rf.fit(x_train,y_train)

    score_list.append(rf.score(x_test,y_test))

print(np.max(score_list))

rf_max = np.max(score_list)
fig1, ax1 = plt.subplots(figsize =(10,10))

plt.plot(score_list,c = "r")

plt.xlabel("n_estimators")

plt.ylabel("Accuracy")

plt.grid()

plt.show()
from sklearn.ensemble import RandomForestClassifier

rf1 = RandomForestClassifier(n_estimators = 7,random_state = 41)

rf1.fit(x_train,y_train)

y_pred = rf1.predict(x_test)

y_true = y_test

cm = confusion_matrix(y_true,y_pred)

f, ax = plt.subplots(figsize =(10,10))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="blue",fmt = ".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()
from sklearn.svm import SVC

svm = SVC()

svm.fit(x_train,y_train)

print(svm.score(x_test,y_test))

svm_score = svm.score(x_test,y_test)
y_pred = svm.predict(x_test)

y_true = y_test

cm = confusion_matrix(y_true,y_pred)

f, ax = plt.subplots(figsize =(10,10))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="blue",fmt = ".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train,y_train)

print(nb.score(x_test,y_test))

nb_score = nb.score(x_test,y_test)
y_pred = nb.predict(x_test)

y_true = y_test

cm = confusion_matrix(y_true,y_pred)

f, ax = plt.subplots(figsize =(10,10))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="blue",fmt = ".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()
result = ({"Random Forest Classifier" : rf_max , "Logistic Regression" : lr_max , "K-Nearest Neighbor" : knn_max , "Support Vector Machine" : svm_score,"Naive Bayes" : nb_score})

result