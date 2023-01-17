import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import pandas as pd 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv(r"/kaggle/input/students-performance-in-exams/StudentsPerformance.csv")

data.info()
data.describe()
data.columns
data.isnull().sum()
print(data["race/ethnicity"].value_counts())

labels = 'Group A', 'Group B','Group C','Group D', "Group E"

sizes = [89,190,319,262,140]

colors = ['skyblue','brown','green','purple','gold']

explode = (0, 0,0,0,0)

fig1, ax1 = plt.subplots(figsize =(10,10))

ax1.pie(sizes,colors = colors ,explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')

plt.title("Race / Ethnicity")

plt.show()
print(data["parental level of education"].value_counts())

labels = 'Some College', 'Associates Degree','High School ','Some High School', "Bachelor's Degree","Master's Degree "

sizes = [226,222,196,179,118,59]

colors = ["red","green","gold","blue","pink","cyan"]

explode = (0, 0,0,0,0,0)

fig1, ax1 = plt.subplots(figsize =(10,10))

ax1.pie(sizes,colors = colors ,explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')

plt.title("Parentel Level of Education")

plt.show()
print(data["lunch"].value_counts())

labels = "Standard","Free / Reduced"

sizes = [645,355]

colors = ["silver","firebrick"]

explode = (0, 0)

fig1, ax1 = plt.subplots(figsize =(10,10))

ax1.pie(sizes,colors = colors ,explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')

plt.title("Lunch")

plt.show()
print(data["test preparation course"].value_counts())

labels = "None","Completed"

sizes = [642,358]

colors = ["orchid","powderblue"]

explode = (0, 0)

fig1, ax1 = plt.subplots(figsize =(10,10))

ax1.pie(sizes,colors = colors ,explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')

plt.title("Test Preparation Course")

plt.show()
fig1, ax1 = plt.subplots(figsize =(10,10))

plt.hist(data["math score"], bins=80,color = "darkorange")

plt.xlabel("Math Score")

plt.ylabel("Frequency")

plt.grid()

plt.show()
fig1, ax1 = plt.subplots(figsize =(10,10))

plt.hist(data["reading score"], bins=80,color = "rosybrown")

plt.xlabel("Reading Score")

plt.ylabel("Frequency")

plt.grid()

plt.show()
fig1, ax1 = plt.subplots(figsize =(10,10))

plt.hist(data["writing score"], bins=80,color = "coral")

plt.xlabel("Writing Score")

plt.ylabel("Frequency")

plt.grid()

plt.show()
df = pd.DataFrame(data,columns=["math score","reading score","writing score"])

f, ax = plt.subplots(figsize =(10,10))

corrMatrix = df.corr()

sns.heatmap(corrMatrix, annot=True)

plt.show()
sns.violinplot(data=data, x="race/ethnicity", y="math score",hue = "gender",

               split=True, inner="quart", linewidth=1,)

sns.despine(left=True)

plt.show()
sns.violinplot(data=data, x="race/ethnicity", y="reading score",hue = "gender",

               split=True, inner="quart", linewidth=1,)

sns.despine(left=True)

plt.show()
sns.violinplot(data=data, x="race/ethnicity", y="writing score",hue = "gender",

               split=True, inner="quart", linewidth=1,)

sns.despine(left=True)

plt.show()
g = sns.catplot(

    data=data, kind="bar",

    x="math score", y="lunch", hue="gender",

    ci="sd", palette="dark", alpha=.6, height=6,

)

g.despine(left=True)

g.set_axis_labels("Math Score", "Lunch")

plt.grid()

plt.show()
g = sns.catplot(

    data=data, kind="bar",

    x="reading score", y="lunch", hue="gender",

    ci="sd", palette="dark", alpha=.6, height=6,

)

g.despine(left=True)

g.set_axis_labels("Reading Score", "Lunch")

plt.grid()

plt.show()
g = sns.catplot(

    data=data, kind="bar",

    x="writing score", y="lunch", hue="gender",

    ci="sd", palette="dark", alpha=.6, height=6,

)

g.despine(left=True)

g.set_axis_labels("Writing Score", "Lunch")

plt.grid()

plt.show()
g = sns.jointplot(

    data=data,

    x="reading score", y="writing score", 

    kind="kde",

)

plt.show()
g = sns.jointplot(

    data=data,

    x="reading score", y="math score", 

    kind="kde",

)

plt.show()
g = sns.jointplot(

    data=data,

    x="math score", y="writing score", 

    kind="kde",

)

plt.show()
data.gender = [1 if each == "female" else 0 for each in data.gender]

data.gender.value_counts()
y = data.gender

data.drop(["gender"],axis = 1 , inplace = True)

data.columns
data = pd.get_dummies(data,columns = ["race/ethnicity","lunch","parental level of education",

                                      "test preparation course"])

data.info()
x = data.astype(int)

x

x= (data-np.min(data)) / (np.max(data)-np.min(data))
x
from sklearn.model_selection import train_test_split

x_train , x_test , y_train, y_test = train_test_split(x,y,test_size = 0.2 , random_state = 42)
from sklearn.svm import SVC

svm1 = SVC(gamma = 0.01 , C = 500 , kernel = "rbf")

svm1.fit(x_train,y_train)

svm1_score = svm1.score(x_test,y_test)

print("SVM Max Score = : ", svm1_score)
y_pred = svm1.predict(x_test)

y_true = y_test

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true,y_pred)

import seaborn as sns

import matplotlib.pyplot as plt



f, ax = plt.subplots(figsize =(8,8))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="blue",fmt = ".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()
knn_list = []

from sklearn.neighbors import KNeighborsClassifier

for each in range(1,100):

    knn1 = KNeighborsClassifier(n_neighbors = each,weights = "uniform",metric = "euclidean" )

    knn1.fit(x_train,y_train)

    knn1_score = knn1.score(x_test,y_test)

    knn_list.append(knn1_score)

knn_max = np.max(knn_list)

print("KNN Max Score = ",knn_max)
y_pred2 = knn1.predict(x_test)

y_true2 = y_test

from sklearn.metrics import confusion_matrix

cm2 = confusion_matrix(y_true2,y_pred2)

import seaborn as sns

import matplotlib.pyplot as plt



f, ax = plt.subplots(figsize =(8,8))

sns.heatmap(cm2,annot = True,linewidths=0.5,linecolor="blue",fmt = ".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C = 10.0, penalty = "l2")

lr.fit(x_train,y_train)

print("Logistic Regression Max Score : ",lr.score(x_test,y_test))

lr_max = lr.score(x_test,y_test)
y_pred3 = lr.predict(x_test)

y_true3 = y_test

from sklearn.metrics import confusion_matrix

cm3 = confusion_matrix(y_true3,y_pred3)

import seaborn as sns

import matplotlib.pyplot as plt



f, ax = plt.subplots(figsize =(8,8))

sns.heatmap(cm3,annot = True,linewidths=0.5,linecolor="blue",fmt = ".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()
score_list = []

from sklearn.ensemble import RandomForestClassifier

for each in range (1,100):

    rf = RandomForestClassifier(n_estimators = each,random_state = 7,bootstrap = "False",criterion="gini",

                                    min_samples_split = 10 , min_samples_leaf = 1)

    rf.fit(x_train,y_train)

    score_list.append(rf.score(x_test,y_test))

    

rf_max = np.max(score_list)

print("RF Max Score : ",rf_max)
y_pred4 = rf.predict(x_test)

y_true4 = y_test

from sklearn.metrics import confusion_matrix

cm4 = confusion_matrix(y_true4,y_pred4)

import seaborn as sns

import matplotlib.pyplot as plt



f, ax = plt.subplots(figsize =(8,8))

sns.heatmap(cm4,annot = True,linewidths=0.5,linecolor="blue",fmt = ".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()

results = {"Support Vector Machine " : svm1_score,

          "Logistic Regression" : lr_max,

          "Random Forest Classifier" : rf_max,

          "K-Nearest Neighbor" : knn_max

          }
results