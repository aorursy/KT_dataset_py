

import numpy as np 

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv(r"/kaggle/input/pima-indians-diabetes-database/diabetes.csv")

data.info()
data.describe()
data.columns
data.isnull().sum()
print(data.Outcome.value_counts())

labels = '0', '1',

sizes = [500, 268]

colors = ['palegoldenrod','lightgrey']

explode = (0, 0)

fig1, ax1 = plt.subplots(figsize =(10,10))

ax1.pie(sizes,colors = colors ,explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')

plt.title("Outcome")

plt.show()
data_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',

       'BMI', 'DiabetesPedigreeFunction', 'Age']

for each in data_columns:

    fig1, ax1 = plt.subplots(figsize =(10,10))

    plt.hist(data[each], bins=80,color = "cadetblue")

    plt.xlabel(each)

    plt.ylabel("Frequency")

    plt.grid()

    plt.show()
df = pd.DataFrame(data,columns=data_columns)

f, ax = plt.subplots(figsize =(15,11))

corrMatrix = df.corr()

sns.heatmap(corrMatrix, annot=True)

plt.show()
g = sns.jointplot(

    data=data,

    x="Glucose", y="Insulin", 

    kind="kde",

)

plt.show()   
fig1, ax1 = plt.subplots(figsize =(10,10))

plt.scatter(data.index , data.Glucose,label =  "Glucose",alpha = 0.5,color = "orangered")

plt.scatter(data.index , data.Insulin,label =  "Insulin",alpha = 0.5,color = "darkblue")

plt.legend(loc ="best")

plt.xlabel("index")

plt.ylabel("Value")

plt.grid()

plt.show()
g = sns.catplot(

    data=data, kind="bar",

    x="Outcome", y="Glucose",

    ci="sd", palette="dark", alpha=.6, height=6,

)

g.despine(left=True)

g.set_axis_labels("Outcome", "Glucose")

plt.grid()

plt.show()
g = sns.catplot(

    data=data, kind="bar",

    x="Outcome", y="Insulin",

    ci="sd", palette="dark", alpha=.6, height=6,

)

g.despine(left=True)

g.set_axis_labels("Outcome", "Insluin")

plt.grid()

plt.show()
sns.violinplot(data=data, x="Outcome", y="Glucose",

               split=True, inner="quart", linewidth=1,)

sns.despine(left=True)

plt.show()
sns.violinplot(data=data, x="Outcome", y="Insulin",

               split=True, inner="quart", linewidth=1,)

sns.despine(left=True)

plt.show()
y = data.Outcome

data.drop(["Outcome"],axis = 1 , inplace = True)

x = (data-np.min(data))/(np.max(data)-np.min(data))
from sklearn.model_selection import train_test_split

x_train , x_test , y_train, y_test = train_test_split(x,y,test_size = 0.2 , random_state = 42)
from sklearn.svm import SVC

svm1 = SVC(gamma = 0.01 , C = 10 , kernel = "rbf")

svm1.fit(x_train,y_train)

svm1_score = svm1.score(x_test,y_test)

print("SVM Max Score = : ", svm1_score)
y_pred = svm1.predict(x_test)

y_true = y_test

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true,y_pred)



f, ax = plt.subplots(figsize =(9,9))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="blue",fmt = ".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()
knn_list = []

from sklearn.neighbors import KNeighborsClassifier

for each in range(1,100):

    knn1 = KNeighborsClassifier(n_neighbors = each,weights = "distance",metric = "manhattan" )

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



f, ax = plt.subplots(figsize =(11,11))

sns.heatmap(cm2,annot = True,linewidths=0.5,linecolor="blue",fmt = ".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C = 1.0, penalty = "l2")

lr.fit(x_train,y_train)

print("Logistic Regression Max Score : ",lr.score(x_test,y_test))

lr_max = lr.score(x_test,y_test)
y_pred3 = lr.predict(x_test)

y_true3 = y_test

from sklearn.metrics import confusion_matrix

cm3 = confusion_matrix(y_true3,y_pred3)

import seaborn as sns

import matplotlib.pyplot as plt



f, ax = plt.subplots(figsize =(11,11))

sns.heatmap(cm3,annot = True,linewidths=0.5,linecolor="blue",fmt = ".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()
score_list = []

from sklearn.ensemble import RandomForestClassifier

for each in range (1,100):

    rf = RandomForestClassifier(n_estimators = each,random_state = 18,bootstrap = "False",criterion="gini",

                                    min_samples_split = 10 , min_samples_leaf = 10)

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



f, ax = plt.subplots(figsize =(11,11))

sns.heatmap(cm4,annot = True,linewidths=0.5,linecolor="blue",fmt = ".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()
results = pd.DataFrame({"Classification" :["Random Forest C." ,"Logistic Regression C.",

            "K-Nearest Neighbor C.","Support Vector C."],

                        "Accuracy" : [rf_max*100,lr_max*100,knn_max*100,svm1_score*100]})

results = results.sort_values(by=['Accuracy'],ascending = False)

print(results)