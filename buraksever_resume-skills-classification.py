# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data1=pd.read_csv("../input/resumes_development.csv")

data2=pd.read_csv("../input/resumes_pilot.csv")
data=pd.concat([data1,data2])

data.head(10)
data.info()
data[data.isnull().any(axis=1)].count()
data.drop(["Female", "URM", "Disability", "Unnamed: 0"], axis=1, inplace=True)
plt.figure(figsize=[5,5])

sns.set(style='darkgrid')

sns.countplot(x="Veteran", data=data, palette='RdYlBu')

data.loc[:,'Veteran'].value_counts()
#train test split

y=data.Veteran.values

x=data.drop(["Veteran"], axis=1)



from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=1)
#Logistic Regression



from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(solver='lbfgs')

lr.fit(x_train, y_train)

lr_prediction = lr.predict(x_test)

lr_score = lr.score(x_test,y_test)

print("Logistic Regression Test Accuracy: {}%".format(round(lr.score(x_test,y_test)*100,2)))



#Confusion Matrix



from sklearn.metrics import confusion_matrix

lr_cm = confusion_matrix(y_test, lr_prediction)



#Mean Squared Error



from sklearn.metrics import mean_squared_error

lr_mse = mean_squared_error(y_test, lr_prediction)
#K_Nearest Neighbour (KNN) Classification



from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=14)

knn.fit(x_train, y_train)

knn_prediction = knn.predict(x_test)

knn_score = knn.score(x_test, y_test)

print("KNN Classification Test Accuracy: {}%".format(round(knn.score(x_test,y_test)*100,2)))



#Confusion Matrix



knn_cm = confusion_matrix(y_test, knn_prediction)



#Mean Squared Error



knn_mse = mean_squared_error(y_test, knn_prediction)
#Find Best K Value



score_list = []

for each in range(1,30):

    knn2 = KNeighborsClassifier(n_neighbors = each)

    knn2.fit(x_train, y_train)

    score_list.append(knn2.score(x_test, y_test))

plt.plot(range(1,30), score_list)

plt.xlabel("K Values")

plt.ylabel("Accuracy")

plt.show()
# Support Vector Machine (SVM)



from sklearn.svm import SVC

svm = SVC(random_state=1, gamma='auto')

svm.fit(x_train, y_train)

svm_prediction = svm.predict(x_test)

svm_score = svm.score(x_test, y_test)

print("SVM Classification Test Accuracy: {}%".format(round(svm.score(x_test,y_test)*100,2)))



#Confusion Matrix



svm_cm = confusion_matrix(y_test, svm_prediction)



#Mean Squared Error



svm_mse = mean_squared_error(y_test, svm_prediction)
#Naive Bayes Classification



from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train, y_train)

nb_prediction = nb.predict(x_test)

nb_score = nb.score(x_test, y_test)

print("Naive Bayes Classification Test Accuracy: {}%".format(round(nb.score(x_test,y_test)*100,2)))



#Confusion Matrix



nb_cm = confusion_matrix(y_test, nb_prediction)



#Mean Squared Error



nb_mse = mean_squared_error(y_test, nb_prediction)
#Decision Tree Classification



from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()

dt.fit(x_train, y_train)

dt_prediction = dt.predict(x_test)

dt_score = dt.score(x_test, y_test)

print("Decision Tree Classification Test Accuracy: {}%".format(round(dt.score(x_test,y_test)*100,2)))



#Confusion Matrix



dt_cm = confusion_matrix(y_test, dt_prediction)



#Mean Squared Error



dt_mse = mean_squared_error(y_test, dt_prediction)
#Random Forest Classification



from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=1)

rf.fit(x_train, y_train)

rf_prediction = rf.predict(x_test)

rf_score = rf.score(x_test, y_test)

print("Random Forest Classification Test Accuracy: {}%".format(round(rf.score(x_test,y_test)*100,2)))



#Confusion Matrix



rf_cm = confusion_matrix(y_test, rf_prediction)



#Mean Squared Error



rf_mse = mean_squared_error(y_test, rf_prediction)
#Visualization of Confusion Matrix



plt.figure(figsize=(20,15))



plt.suptitle("Confusion Matrixes", fontsize=18)



plt.subplot(2,3,1)

plt.title("Logistic Regression Confusion Matrix")

sns.heatmap(lr_cm, cbar=False, annot=True, cmap="PuBuGn", fmt="d")



plt.subplot(2,3,2)

plt.title("KNN Classification Confusion Matrix")

sns.heatmap(knn_cm, cbar=False, annot=True, cmap="PuBuGn", fmt="d")



plt.subplot(2,3,3)

plt.title("SVM Classification Confusion Matrix")

sns.heatmap(svm_cm, cbar=False, annot=True, cmap="PuBuGn", fmt="d")



plt.subplot(2,3,4)

plt.title("Naive Bayes Classification Confusion Matrix")

sns.heatmap(nb_cm, cbar=False, annot=True, cmap="PuBuGn", fmt="d")



plt.subplot(2,3,5)

plt.title("Decision Tree Classification Confusion Matrix")

sns.heatmap(dt_cm, cbar=False, annot=True, cmap="PuBuGn", fmt="d")



plt.subplot(2,3,6)

plt.title("Random Forest Classification Confusion Matrix")

sns.heatmap(rf_cm, cbar=False, annot=True, cmap="PuBuGn", fmt="d")



plt.show()
TN = [lr_cm[0,0], knn_cm[0,0], svm_cm[0,0], nb_cm[0,0], dt_cm[0,0], rf_cm[0,0]]

FP = [lr_cm[0,1], knn_cm[0,1], svm_cm[0,1], nb_cm[0,1], dt_cm[0,1], rf_cm[0,1]]

FN = [lr_cm[1,0], knn_cm[1,0], svm_cm[1,0], nb_cm[1,0], dt_cm[1,0], rf_cm[1,0]]

TP = [lr_cm[1,1], knn_cm[1,1], svm_cm[1,1], nb_cm[1,1], dt_cm[1,1], rf_cm[1,1]]

Accuracy = [lr_score, knn_score, svm_score, nb_score, dt_score, rf_score]

MSE = [lr_mse, knn_mse, svm_mse, nb_mse, dt_mse, rf_mse]

Classification = ["Logistic Regression", "KNN Classification", "SVM Classification", "Naive Bayes Classification", 

                  "Decision Tree Classification", "Random Forest Classification"]

list_matrix = [Classification, TN, FP, FN, TP, Accuracy, MSE]

list_headers = ["Model", "TN", "FP", "FN", "TP", "Accuracy", "MSE"]

zipped = list(zip(list_headers, list_matrix))

data_dict = dict(zipped)

df=pd.DataFrame(data_dict)
df
trace1 = {

    'x':df.Model,

    'y':df.TN,

    'name':'True Negative',

    'type':'bar'}



trace2 = {

    'x':df.Model,

    'y':df.FP,

    'name':'False Positive',

    'type':'bar'}



trace3 = {

    'x':df.Model,

    'y':df.FN,

    'name':'False Negative',

    'type':'bar'}



trace4 = {

    'x':df.Model,

    'y':df.TP,

    'name':'True Positive',

    'type':'bar'}



graph = [trace1, trace2, trace3, trace4];

layout = {

  'xaxis': {'title': 'Classification Models'},

  'barmode': 'relative',

  'title': 'Confusion Matrix Values of Classification Models'

};

fig = go.Figure(data = graph, layout = layout)

iplot(fig)
#Accuracy

plt.figure(figsize=(15,10))

ax= sns.barplot(x=df.Model, y=df.Accuracy, palette = sns.cubehelix_palette(len(df.Model)))

ax.set_xticklabels(ax.get_xticklabels(),rotation=30)

plt.xlabel('Classification Models')

plt.ylabel('Accuracy')

plt.title('Accuracy Scores of Classification Models')

for i in ax.patches:

    ax.text(i.get_x()+.19, i.get_height()-0.3, \

            str(round((i.get_height()), 4)), fontsize=15, color='white')

plt.show()
#MSE

plt.figure(figsize=(15,10))

ax= sns.barplot(x=df.Model, y=df.MSE, palette = sns.cubehelix_palette(len(df.Model)))

ax.set_xticklabels(ax.get_xticklabels(),rotation=30)

plt.xlabel('Classification Models')

plt.ylabel('Mean Squared Error')

plt.title('MSE Scores of Classification Models')

for i in ax.patches:

    ax.text(i.get_x()+.19, i.get_height()-0.1, \

            str(round((i.get_height()), 5)), fontsize=15, color='white')

plt.show()
d = {'y_test': y_test, 'Logistic_Regression_prediction': lr_prediction, 'KNN_prediction': knn_prediction, 

     'SVM_prediction': svm_prediction, 'Naive_Bayes_prediction': nb_prediction, 'Decision_Tree_prediction': dt_prediction, 

     'Random_Forest_prediction': rf_prediction}

data1=pd.DataFrame(data=d)

data1.T