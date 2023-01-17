import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objects as go

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv("/kaggle/input/indian-liver-patient-records/indian_liver_patient.csv")
data.head()
data.info()
#Renaming the columns.



data.rename(columns={'Dataset': 'target', 'Alamine_Aminotransferase': 'Alanine_Aminotransferase', 'Total_Protiens': 'Total_Proteins'}, inplace = True)
data.target.unique()
data.target = [0 if each == 2 else 1 for each in data.target]
#Data contains object variables, I want integers or float variables.



data.Gender = [1 if each == 'Male' else 0 for each in data.Gender]
data.dtypes
data.isna().sum()
#Filling null values.

data['Albumin_and_Globulin_Ratio'].mean()
data.fillna(0.94, inplace = True)
data.info()
correlation = data.corr()

fig, ax = plt.subplots(figsize=(12,12))

sns.heatmap(correlation, annot = True, linewidths = 0.5, ax = ax)

plt.show()
list_ = ["Age", "Total_Bilirubin", "Direct_Bilirubin", "target"]



sns.heatmap(data[list_].corr(), annot = True, fmt = ".2f")

plt.show()
list2 = ["Alkaline_Phosphotase", "Alanine_Aminotransferase", "Aspartate_Aminotransferase", "Total_Proteins", "target"]

sns.heatmap(data[list2].corr(), annot = True, fmt = ".2f")

plt.show()
list3 = ["Albumin_and_Globulin_Ratio", "Albumin", "target"]

sns.heatmap(data[list3].corr(), annot = True, fmt = ".2f")

plt.show()
f, axes = plt.subplots(1, 2, figsize = (12, 8))



sns.countplot(x = "target", data = data, ax=axes[0])

sns.countplot(x = "target", hue = 'Gender', data = data, ax=axes[1])

plt.show()
print("Number of people who suffers from liver disease: {}" .format(data.target.value_counts()[1]))

print("Number of people who does not suffer from liver disease: {}" .format(data.target.value_counts()[0]))
g = sns.FacetGrid(data, col = "target", height = 7)

g.map(sns.distplot, "Age", bins = 25)

plt.show()
#I want to add a column showing globulin values.



ratio = data.Albumin_and_Globulin_Ratio.values

albumin = data.Albumin.values

globulin = []

for i in range(0, 583):

    globulin.append(float("{:.2f}".format(albumin[i] / ratio [i])))
data.insert(9, 'Globulin', globulin, True)
data.head()
g = sns.FacetGrid(data, col = "target", row = "Gender", height = 5)

g.map(plt.hist, "Albumin", bins = 25)

plt.show()
g = sns.FacetGrid(data, col = "target", row = "Gender", height = 5)

g.map(plt.hist, "Globulin", bins = 25)

plt.show()
g = sns.FacetGrid(data, col = "target", row = "Gender", height = 5)

g.map(plt.hist, "Albumin_and_Globulin_Ratio", bins = 25)

plt.show()
patient = data[data.target == 1]

healthy = data[data.target != 1]



trace0 = go.Scatter(

    x = patient['Total_Bilirubin'],

    y = patient['Direct_Bilirubin'],

    name = 'Patient',

    mode = 'markers', 

    marker = dict(color = '#616ADE',

        line = dict(

            width = 1)))



trace1 = go.Scatter(

    x = healthy['Total_Bilirubin'],

    y = healthy['Direct_Bilirubin'],

    name = 'healthy',

    mode = 'markers',

    marker = dict(color = '#F3EC1F',

        line = dict(

            width = 1)))



layout = dict(title = 'Total Bilirubin vs Conjugated Bilirubin',

              yaxis = dict(title = 'Conjugated Bilirubin', zeroline = False),

              xaxis = dict(title = 'Total Bilirubin', zeroline = False)

             )



data2 = [trace0, trace1]



fig = go.Figure(data=data2,

                layout=layout)



fig.show()
g = sns.FacetGrid(data, col = "target", height = 7)

g.map(sns.distplot, "Alkaline_Phosphotase", bins = 25)

plt.show()
g = sns.FacetGrid(data, col = "target", height = 7)

g.map(sns.distplot, "Alanine_Aminotransferase", bins = 25)

plt.show()
g = sns.FacetGrid(data, col = "target", height = 7)

g.map(sns.distplot, "Aspartate_Aminotransferase", bins = 25)

plt.show()
g = sns.FacetGrid(data, col = "target", height = 7)

g.map(sns.distplot, "Total_Proteins", bins = 25)

plt.show()



print("Mean of the total protein level in patiens:", float("{:.2f}".format( data['Total_Proteins'][data.target == 1].mean())))

print("Mean of the total protein level in healthy people:", float("{:.2f}".format(data['Total_Proteins'][data.target == 0].mean())))
#Importing necessary libraries



from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, confusion_matrix
#I want to store the scores in a dictionary to see which prediction method works best.



scores = {}
data = data.drop(columns = ['Total_Proteins', 'Age', 'Gender'])
y = data.target.values

x_ = data.drop(columns = ["target"])
#Normalisation

x = ((x_ - np.min(x_)) / (np.max(x_) - np.min(x_))).values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
x_train = x_train.T

y_train = y_train.T

x_test = x_test.T

y_test = y_test.T
def initialise(dimension):

    

    weight = np.full((dimension,1),0.01)

    bias = 0.0

    return weight,bias
def sigmoid(z):

    y_head = 1/(1 + np.exp(-z))

    return y_head
def forward_backward(weight, bias, x_train, y_train):

    z = np.dot(weight.T, x_train) + bias

    y_head = sigmoid(z)

    loss = -((y_train * np.log(y_head)) + ((1 - y_train) * np.log(1 - y_head)))

    cost = (np.sum(loss))/x_train.shape[1]

    

    derivative_weight = (np.dot(x_train,((y_head - y_train).T)))/x_train.shape[1]

    derivative_bias = np.sum(y_head - y_train)/x_train.shape[1]

    gradients = {"derivative_weight": derivative_weight,"derivative_bias": derivative_bias}

    

    return cost, gradients



def update(weight, bias, x_train, y_train, learning_rate, number_of_iterarion):

    cost_list = []

    cost_list2 = []

    index = []

    

    for i in range(number_of_iterarion):

        cost, gradients = forward_backward(weight, bias, x_train, y_train)

        cost_list.append(cost)

        

        weight = weight - learning_rate * gradients["derivative_weight"]

        bias = bias - learning_rate * gradients["derivative_bias"]

        

        if i % 10 == 0:

            cost_list2.append(cost)

            index.append(i)

            print ("Cost after iteration %i: %f" %(i, cost))

            

    parameters = {"weight": weight, "bias": bias}

    plt.plot(index, cost_list2)

    plt.xticks(index, rotation='vertical')

    plt.xlabel("Number of Iterarions")

    plt.ylabel("Cost")

    plt.show()

    

    return parameters, gradients, cost_list



def predict(weight, bias, x_test):

    z = sigmoid(np.dot(weight.T, x_test) + bias)

    Y_prediction = np.zeros((1, x_test.shape[1]))

    for i in range(z.shape[1]):

        if z[0,i]<= 0.5:

            Y_prediction[0,i] = 0

        else:

            Y_prediction[0,i] = 1



    return Y_prediction



def logistic_regression(x_train, y_train, x_test, y_test, learning_rate,  num_iterations):

    dimension =  x_train.shape[0]

    weight, bias = initialise(dimension)

    parameters, gradients, cost_list = update(weight, bias, x_train, y_train, learning_rate, num_iterations)

    

    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)

    

    print("test accuracy: {} %".format(float("{:.2f}".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))))

    scores['Logistic Regression with Functions'] = float("{:.2f}".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
logistic_regression(x_train, y_train, x_test, y_test, learning_rate = 0.01, num_iterations = 200)
lr = LogisticRegression()



lr.fit(x_train.T, y_train.T)

print("test accuracy = {}%" .format(float("{:.2f}".format(lr.score(x_test.T, y_test.T) * 100))))
scores['Logistic Regression Score'] = float("{:.2f}".format(lr.score(x_test.T, y_test) * 100))
x_train = x_train.T

x_test = x_test.T
knn_scores = []

for each in range(1, 15):

    knn2 = KNeighborsClassifier(n_neighbors = each)

    knn2.fit(x_train, y_train)

    knn_scores.append(knn2.score(x_test, y_test))



plt.figure(figsize = (10, 8))

plt.plot(range(1, 15), knn_scores)

plt.xlabel("k values")

plt.ylabel("accuracy")

plt.show()
knn = KNeighborsClassifier(n_neighbors = 9)

knn.fit(x_train, y_train)



prediction = knn.predict(x_test)
print("KNN score: {}" .format(float("{:.2f}".format(knn.score(x_test, y_test) * 100))))
scores['KNN Score'] = (float("{:.2f}".format(knn.score(x_test, y_test) * 100)))
svm = SVC(random_state = 1)

svm.fit(x_train, y_train)
print("SVM Score is: {}" .format(float("{:.2f}".format(svm.score(x_test, y_test) * 100))))
scores['SVM Score'] = (float("{:.2f}".format(svm.score(x_test, y_test) * 100)))
nb = GaussianNB()

nb.fit(x_train, y_train)
print("Naive Bayes Score is: {}" .format(float("{:.2f}".format(nb.score(x_test, y_test) * 100))))
scores['Naive Bayes Score'] = (float("{:.2f}".format(nb.score(x_test, y_test) * 100)))
dt = DecisionTreeClassifier()

dt.fit(x_train, y_train)
print("Decision Tree Score is: {}" .format(float("{:.2f}".format(dt.score(x_test, y_test) * 100))))
scores['Decision Tree Score'] = (float("{:.2f}".format(dt.score(x_test, y_test) * 100)))
rf = RandomForestClassifier(n_estimators = 100, random_state=1)

rf.fit(x_train, y_train)
print("Random Forest Score is: {}" .format(float("{:.2f}".format(rf.score(x_test, y_test) * 100))))
scores['Random Forest Score'] = (float("{:.2f}".format(rf.score(x_test, y_test) * 100)))
lists = sorted(scores.items())



x_axis, y_axis = zip(*lists)



plt.figure(figsize = (15, 10))

plt.plot(x_axis, y_axis)

plt.show()