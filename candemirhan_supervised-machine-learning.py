# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import parallel_coordinates
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Importing and Making a First Look

iris = pd.read_csv("/kaggle/input/iris/Iris.csv")
iris.head()
iris.info()
iris.columns
# Data Visualization

plt.figure(figsize=(15,8))
parallel_coordinates(iris.drop(["Id"],axis=1), 'Species', colormap=plt.get_cmap("Set1"))
plt.title("Iris Class Visualization")
plt.xlabel("Features")
plt.ylabel("Length")
plt.show()
iris_versicolor = iris[iris.Species == 'Iris-versicolor']
iris_setosa = iris[iris.Species == "Iris-setosa"]
iris_virginica = iris[iris.Species == "Iris-virginica"]

# trace1 =  iris setosa
trace1 = go.Scatter3d(
    x=iris_setosa.SepalLengthCm,
    y=iris_setosa.SepalWidthCm,
    z=iris_setosa.PetalLengthCm,
    mode='markers',
    name = "iris_setosa",
    marker=dict(
        color='rgb(217, 100, 100)',
        size=12,
        line=dict(
            color='rgb(255, 255, 255)',
            width=0.1
        )
    )
)
# trace2 =  iris virginica
trace2 = go.Scatter3d(
    x=iris_virginica.SepalLengthCm,
    y=iris_virginica.SepalWidthCm,
    z=iris_virginica.PetalLengthCm,
    mode='markers',
    name = "iris_virginica",
    marker=dict(
        color='rgb(54, 170, 127)',
        size=12,
        line=dict(
            color='rgb(204, 204, 204)',
            width=0.1
        )
    )
)
trace3 = go.Scatter3d(
    x=iris_versicolor.SepalLengthCm,
    y=iris_versicolor.SepalWidthCm,
    z=iris_versicolor.PetalLengthCm,
    mode='markers',
    name = "iris_setosa",
    marker=dict(
        color='rgb(150, 240, 100)',
        size=12,
        line=dict(
            color='rgb(150, 150, 150)',
            width=0.1
        )
    )
)
data = [trace1, trace2, trace3]
layout = go.Layout(
    title = ' 3D Visualization of Iris Class',
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
# Data Manipulation and Cleaning

iris.Species = [0 if each == "Iris-versicolor" else (1 if each == "Iris-setosa" else 2) for each in iris.Species]
iris.drop(["Id"],axis=1,inplace=True)
iris.Species.value_counts()
y = iris.Species.values
x_data = iris.drop(["Species"],axis=1)
# Normalization

x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))
# Train Test Split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.15, random_state=42)

# Logistic Regression Score by using Math
def initialize_weights_and_bias(dimension):
    w = np.full((dimension,1),0.01)
    b = 0.0
    return w, b
def sigmoid(z):
    y_head = 1 /(1 + np.exp(-z))
    return y_head
def forward_and_backward_propogation(w,b,x_train,y_train):
    # Forward Propogation
    z = np.dot(w.T,x_train) + b
    y_head = sigmoid(z)
    loss = -y_train*np.log(y_head) - (1 - y_train)*np.log(y_head) # Hatalı satır
    cost = np.sum(loss) /x_train.shape[1]
    # Backward Propogation
    derivative_weight = (np.dot(x_train,(y_head-y_train).T)) / x_train.shape[1]
    derivative_bias = np.sum(y_head - y_train) / x_train.shape[1]
    gradients = {"Derivative Weight":derivative_weight, "Derivative Bias":derivative_bias}
    return cost, gradients
def update(w, b, x_train, y_train, learning_rate, number_of_iteration):
    cost_list = list()
    cost_list2 = list()
    index = list()
    for i in range(number_of_iteration):
        cost, gradients = forward_and_backward_propogation(w,b,x_train,y_train)
        cost_list.append(cost)
        index.append(i)
        if(i % 1000 == 0):
            cost_list2.append(cost)
            print("Cost after Iterations %i : %f"%(i,cost))
    #Visualization
    parameters = {"Weight":w, "Bias":b}
    plt.plot(index,cost_list)
    plt.xticks(index,rotation="vertical")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Cost Values")
    plt.show()
    return parameters, gradients, cost_list
def predict(w,b,x_test):
    z = np.dot(w.T,x_train) + b
    y_head = sigmoid(z)
    y_pred = np.zeros((1,x_test.shape[1]))
    for i in range(z.shape[1]):
        if(z[0,i] <= 2/3):
            y_pred[0,i] = 0
        elif(z[0,i] <= 4/3):
            y_pred[0,i] = 1
        else:
            y_pred[0,i] = 2
    return y_pred
def logistic_regression(x_train, y_train, x_test, y_test, learning_rate, num_iterations):
    dimension = x_train.shape[0]
    w, b = initialize_weights_and_bias(dimension)
    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate, num_iterations)
    
    y_prediction_test = predict(parameters["Weight"],parameters["Bias"],x_test)
    y_prediction_train = predict(parameters["Weight"],parameters["Bias"],x_test)
    
    print("Train Accuracy : {}".format(100 - np.mean(np.abs(y_prediction_train - y_train))*100))
    print("Test Accuracy : {}".format(100 - np.mean(np.abs(y_prediction_test - y_test))*100))
    
# ValueError: operands could not be broadcast together with shapes (127,) (1,4) 
    # Hatayı bulabilirmisiniz.?
x_train.shape, x_test.shape, y_train.shape, y_test.shape
logistic_regression(x_train, y_train, x_test, y_test, learning_rate=2,num_iterations=300)
# Logistic Regression Score by Using sklearn Module
lr = LogisticRegression()
lr.fit(x_train, y_train)
print("Test Accuracy of Logistic Regression with sklearn : {}".format(lr.score(x_test,y_test)))
print("Train Accuracy of Logistic Regression with sklearn : {}".format(lr.score(x_train,y_train)))

# Confusion Matrix and Classification Report
lr_pred = lr.predict(x_test)
cm_lr = confusion_matrix(y_test,predict)
print("Confusion matrix:\n\n",confusion_matrix(y_test,predict))
print("\nTP: {}\tFP: {}\tFN: {}\tTN: {}\t".format(cm[0,0],cm[0,1],cm[1,0],cm[1,1]))
print("")
print("Classification Report:\n\n",classification_report(y_test,predict))
# KNN Score
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
print("Test Accuracy of KNN with sklearn : {}".format(knn.score(x_test,y_test)))
print("Train Accuracy of KNN with sklearn : {}".format(knn.score(x_train,y_train)))

# Confusion Matrix and Classification Report
knn_pred = knn.predict(x_test)
cm_knn = confusion_matrix(y_test,predict)
print("Confusion matrix:\n\n",confusion_matrix(y_test,predict))
print("\nTP: {}\tFP: {}\tFN: {}\tTN: {}\t".format(cm[0,0],cm[0,1],cm[1,0],cm[1,1]))
print("")
print("Classification Report:\n\n",classification_report(y_test,predict))
# Finding K Factor for Maximum accuracy
n_neig = range(1,25)
train_accuracy = list()
test_accuracy = list()
for i,j in enumerate(n_neig):
    knn2 = KNeighborsClassifier(n_neighbors = j)
    knn2.fit(x_train,y_train)
    train_accuracy.append(knn2.score(x_train,y_train))
    test_accuracy.append(knn2.score(x_test,y_test))
plt.figure(figsize=[13,8])
plt.plot(n_neig, test_accuracy, label = "Test Accuracy", color = "red")
plt.plot(n_neig, train_accuracy, label = "Train Accuracy", color = "blue")
plt.legend()
plt.title("K Factor versus Accuracy")
plt.xlabel("K Factor")
plt.ylabel("Accuracy")
plt.xticks(n_neig)
plt.show()
print("Best Accuracy is {} with K = {}".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))
# SVM Score
svm = SVC(random_state = 42)
svm.fit(x_train,y_train)
print("Test Accuracy of SVM : {}".format(svm.score(x_test,y_test)))
print("Train Accuracy of SVM : {}".format(svm.score(x_train,y_train)))

# Confusion Matrix and Classification Report
svm_pred = svm.predict(x_test)
cm_svm = confusion_matrix(y_test,predict)
print("Confusion matrix:\n\n",confusion_matrix(y_test,predict))
print("\nTP: {}\tFP: {}\tFN: {}\tTN: {}\t".format(cm[0,0],cm[0,1],cm[1,0],cm[1,1]))
print("")
print("Classification Report:\n\n",classification_report(y_test,predict))
# Naive Bayes Score
nb = GaussianNB()
nb.fit(x_train,y_train)
print("Test Accuracy of Naive Bayes : {}".format(nb.score(x_test,y_test)))
print("Train Accuracy of Naive Bayes : {}".format(nb.score(x_train,y_train)))

# Confusion Matrix and Classification Report
nb_pred = nb.predict(x_test)
cm_nb = confusion_matrix(y_test,predict)
print("Confusion matrix:\n\n",confusion_matrix(y_test,predict))
print("\nTP: {}\tFP: {}\tFN: {}\tTN: {}\t".format(cm[0,0],cm[0,1],cm[1,0],cm[1,1]))
print("")
print("Classification Report:\n\n",classification_report(y_test,predict))
# Decision Tree Score
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
print("Test Accuracy of Decision Tree Classifier : {}".format(dt.score(x_test,y_test)))
print("Train Accuracy of Decision Tree Classifier : {}".format(dt.score(x_train,y_train)))

# Confusion Matrix and Classification Report
dt_pred = dt.predict(x_test)
cm_dt = confusion_matrix(y_test,predict)
print("Confusion matrix:\n\n",confusion_matrix(y_test,predict))
print("\nTP: {}\tFP: {}\tFN: {}\tTN: {}\t".format(cm[0,0],cm[0,1],cm[1,0],cm[1,1]))
print("")
print("Classification Report:\n\n",classification_report(y_test,predict))
# Random Forest Score
rf = RandomForestClassifier(n_estimators=100,random_state=42)
rf.fit(x_train,y_train)
print("Test Accuracy of Random Forest Classifier : {}".format(rf.score(x_test,y_test)))
print("Train Accuracy of Random Forest Classifier : {}".format(rf.score(x_train,y_train)))

# Confusion Matrix and Classification Report
rf_pred = rf.predict(x_test)
cm_rf = confusion_matrix(y_test,predict)
print("Confusion matrix:\n\n",confusion_matrix(y_test,predict))
print("\nTP: {}\tFP: {}\tFN: {}\tTN: {}\t".format(cm[0,0],cm[0,1],cm[1,0],cm[1,1]))
print("")
print("Classification Report:\n\n",classification_report(y_test,predict))