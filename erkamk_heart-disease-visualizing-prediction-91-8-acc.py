import numpy as np 

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv(r"/kaggle/input/heart-disease-uci/heart.csv")

data.info()
data.describe()
data.columns
data.isnull().sum()
print(data.sex.value_counts())

labels = 'Male', 'Female',

sizes = [207, 96]

colors = ["silver","firebrick"]

explode = (0, 0)

fig1, ax1 = plt.subplots(figsize =(10,10))

ax1.pie(sizes, explode=explode,colors = colors, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')

plt.title("")

plt.show()
fig1, ax1 = plt.subplots(figsize =(20,10))

plt.hist(data.age, bins=80,color = "rosybrown")

plt.xlabel("Age")

plt.ylabel("Frequency")

plt.grid()

plt.show()
print(data.cp.value_counts())

labels = 'Type 1', 'Type 2',  "Type 3" , "Type 4",

sizes = [143, 87,50,23]

explode = (0, 0, 0, 0)

colors = ["silver","firebrick","rosybrown","steelblue"]

fig1, ax1 = plt.subplots(figsize =(10,10))

ax1.pie(sizes, explode=explode, labels=labels,colors=colors , autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')

plt.title("Chest Pain Type")

plt.show()
fig1, ax1 = plt.subplots(figsize =(20,10))

plt.hist(data.trestbps, bins=80,color = "rosybrown")

plt.xlabel("Resting blood preasure")

plt.ylabel("Frequency")

plt.grid()

plt.show()
fig1, ax1 = plt.subplots(figsize =(20,10))

plt.hist(data.chol, bins=80,color = "rosybrown")

plt.xlabel("serum cholestoral in mg/dl")

plt.ylabel("Frequency")

plt.grid()

plt.show()
print(data.fbs.value_counts())

labels = 'FBS > 120', 'FBS < 120',

sizes = [258, 45]

explode = (0, 0)

colors = ["silver","firebrick"]

fig1, ax1 = plt.subplots(figsize =(10,10))

ax1.pie(sizes, explode=explode, labels=labels,colors = colors, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')

plt.title("Fasting Blood Sugar")

plt.show()
print(data.restecg.value_counts())

labels = '0', '1' , '2',

sizes = [147, 152,4]

explode = (0, 0,0)

colors = ["silver","firebrick","rosybrown"]

fig1, ax1 = plt.subplots(figsize =(10,10))

ax1.pie(sizes, explode=explode,colors = colors, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')

plt.title("resting electrocardiographic results")

plt.show()
fig1, ax1 = plt.subplots(figsize =(20,10))

plt.hist(data.thalach, bins=80,color ="rosybrown")

plt.xlabel("maximum heart rate achieved")

plt.ylabel("Frequency")

plt.grid()

plt.show()
print(data.exang.value_counts())

labels = '0', '1' ,

sizes = [204, 99]

explode = (0 , 0)

colors = ["silver","firebrick"]

fig1, ax1 = plt.subplots(figsize =(10,10))

ax1.pie(sizes, explode=explode,colors = colors, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')

plt.title("exercise induced angina")

plt.show()
fig1, ax1 = plt.subplots(figsize =(10,10))

plt.scatter(data.index,data.oldpeak,color = "red",alpha = 0.6)

plt.grid()

plt.xlabel("index")

plt.ylabel("value")

plt.title("ST depression induced by exercise relative to rest")

plt.show()
print(data.slope.value_counts())

labels = '2', '1' , '0' ,

sizes = [142, 140 , 21]

explode = (0 , 0 , 0)

colors = ["silver","firebrick","rosybrown"]

fig1, ax1 = plt.subplots(figsize =(10,10))

ax1.pie(sizes, explode=explode,colors = colors, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')

plt.title("the slope of the peak exercise ST segment")

plt.show()
print(data.ca.value_counts())

labels = '0', '1' , '2' , ' 3' , '4',

sizes = [175, 65 , 38,20,5]

explode = (0 , 0 , 0,0,0)

colors = ["coral","khaki","tan","mediumaquamarine","palevioletred"]

fig1, ax1 = plt.subplots(figsize =(10,10))

ax1.pie(sizes, explode=explode, labels=labels,colors = colors, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')

plt.title("number of major vessels (0-3) colored by flourosopy")

plt.show()
print(data.thal.value_counts())

labels = '0', '1' , '2' , ' 3' ,

sizes = [2, 18 , 166,117]

explode = (0 , 0 ,0,0)

colors = ["khaki","tan","mediumaquamarine","palevioletred"]

fig1, ax1 = plt.subplots(figsize =(10,10))

ax1.pie(sizes, explode=explode, labels=labels,colors = colors ,autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')

plt.title("thal: 0 = normal; 1 = fixed defect; 2 = reversable defect")

plt.show()
print(data.target.value_counts())

labels = '0', '1' ,

sizes = [165,138]

explode = (0 , 0 )

colors = ["silver","firebrick"]

fig1, ax1 = plt.subplots(figsize =(10,10))

ax1.pie(sizes, explode=explode, labels=labels,colors = colors, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')

plt.title("target: 0= less chance of heart attack 1= more chance of heart attack")

plt.show()
df = pd.DataFrame(data,columns=['age','sex','target'])

f, ax = plt.subplots(figsize =(10,10))

corrMatrix = df.corr()

sns.heatmap(corrMatrix, annot=True)

plt.show()
g = sns.catplot(

    data=data, kind="bar",

    x="target", y="age", hue="sex",

    ci="sd", palette="dark", alpha=.6, height=6

)

g.despine(left=True)

g.set_axis_labels("Target", "Age")

plt.show()
g = sns.catplot(

    data=data, kind="bar",

    x="sex", y="chol", hue="target",

    ci="sd", palette="dark", alpha=.6, height=6

)

g.despine(left=True)

g.set_axis_labels("sex", "chol")

plt.show()
y = data.target

data.drop(["target"],axis = 1 , inplace = True)

data_copy = data.copy()

data.columns
data = pd.get_dummies(data,columns = ["sex","cp","fbs","restecg","exang","slope","ca","thal"])

data
x = (data-np.min(data)) / (np.max(data)-np.min(data))

x
from sklearn.model_selection import train_test_split

x_train , x_test , y_train, y_test = train_test_split(x,y,test_size = 0.2 , random_state = 42)
from sklearn.svm import SVC

svm1 = SVC(gamma = 0.1 , C = 1 , kernel = "rbf")

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

    knn1 = KNeighborsClassifier(n_neighbors = each,weights = "distance",metric = "euclidean" )

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

lr = LogisticRegression(C = 10.0 , penalty = "l2")

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

    rf = RandomForestClassifier(n_estimators = each,random_state = 42,bootstrap = "False",criterion="gini",

                                min_samples_split = 2 , min_samples_leaf = 1)

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
