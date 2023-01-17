# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.graph_objs as go
import plotly.express as px



import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")

import seaborn as sns

from collections import Counter

import warnings
warnings.filterwarnings("ignore")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/pima-indians-diabetes-database/diabetes.csv")
data.head()
data.describe()
data.info()
var_val = data.Pregnancies.values
var_ind = data.Pregnancies.index

var_data = pd.DataFrame({ "Index" : var_ind , "Values" : var_val})
var_data

trace1 =go.Scatter(
    x = var_data["Index"],
    y = var_data["Values"],
    name = "Pregnancies")

İlişki_data =[trace1]

layout = go. Layout(
    dict(title = 'Kişilere Göre "Pregnancies" Özelliğinin Değişimi',
              xaxis= dict(title= 'Kişi Sayısı',ticklen= 5,zeroline= False)) )


fig =go.Figure(data=İlişki_data , layout=layout)
fig
var_val = data.Glucose.values
var_ind = data.Glucose.index

var_data = pd.DataFrame({ "Index" : var_ind , "Values" : var_val})
var_data

trace1 =go.Scatter(
    x = var_data["Index"],
    y = var_data["Values"],
    name = "Glucose")

İlişki_data =[trace1]

layout = go. Layout(
    dict(title = 'Kişilere Göre "Glucose" Özelliğinin Değişimi',
              xaxis= dict(title= 'Kişi Sayısı',ticklen= 5,zeroline= False)) )


fig =go.Figure(data=İlişki_data , layout=layout)
fig
var_val = data.BloodPressure.values
var_ind = data.BloodPressure.index

var_data = pd.DataFrame({ "Index" : var_ind , "Values" : var_val})
var_data

trace1 =go.Scatter(
    x = var_data["Index"],
    y = var_data["Values"],
    name = "BloodPressure")

İlişki_data =[trace1]

layout = go. Layout(
    dict(title = 'Kişilere Göre "BloodPressure" Özelliğinin Değişimi',
              xaxis= dict(title= 'Kişi Sayısı',ticklen= 5,zeroline= False)) )


fig =go.Figure(data=İlişki_data , layout=layout)
fig
var_val = data.SkinThickness.values
var_ind = data.SkinThickness.index

var_data = pd.DataFrame({ "Index" : var_ind , "Values" : var_val})
var_data

trace1 =go.Scatter(
    x = var_data["Index"],
    y = var_data["Values"],
    name = "SkinThickness")

İlişki_data =[trace1]

layout = go. Layout(
    dict(title = 'Kişilere Göre "SkinThickness" Özelliğinin Değişimi',
              xaxis= dict(title= 'Kişi Sayısı',ticklen= 5,zeroline= False)) )


fig =go.Figure(data=İlişki_data , layout=layout)
fig
var_val = data.Insulin.values
var_ind = data.Insulin.index

var_data = pd.DataFrame({ "Index" : var_ind , "Values" : var_val})
var_data

trace1 =go.Scatter(
    x = var_data["Index"],
    y = var_data["Values"],
    name = "Insulin")

İlişki_data =[trace1]

layout = go. Layout(
    dict(title = 'Kişilere Göre "Insulin" Özelliğinin Değişimi',
              xaxis= dict(title= 'Kişi Sayısı',ticklen= 5,zeroline= False)) )


fig =go.Figure(data=İlişki_data , layout=layout)
fig
var_val = data.BMI.values
var_ind = data.BMI.index

var_data = pd.DataFrame({ "Index" : var_ind , "Values" : var_val})
var_data

trace1 =go.Scatter(
    x = var_data["Index"],
    y = var_data["Values"],
    name = "BMI")

İlişki_data =[trace1]

layout = go. Layout(
    dict(title = 'Kişilere Göre "BMI" Özelliğinin Değişimi',
              xaxis= dict(title= 'Kişi Sayısı',ticklen= 5,zeroline= False)) )


fig =go.Figure(data=İlişki_data , layout=layout)
fig
var_val = data.DiabetesPedigreeFunction.values
var_ind = data.DiabetesPedigreeFunction.index

var_data = pd.DataFrame({ "Index" : var_ind , "Values" : var_val})
var_data

trace1 =go.Scatter(
    x = var_data["Index"],
    y = var_data["Values"],
    name = "DiabetesPedigreeFunction")

İlişki_data =[trace1]

layout = go. Layout(
    dict(title = 'Kişilere Göre "DiabetesPedigreeFunction" Özelliğinin Değişimi',
              xaxis= dict(title= 'Kişi Sayısı',ticklen= 5,zeroline= False)) )


fig =go.Figure(data=İlişki_data , layout=layout)
fig
var_val = data.Age.values
var_ind = data.Age.index

var_data = pd.DataFrame({ "Index" : var_ind , "Values" : var_val})
var_data

trace1 =go.Scatter(
    x = var_data["Index"],
    y = var_data["Values"],
    name = "Age")

İlişki_data =[trace1]

layout = go. Layout(
    dict(title = 'Kişilere Göre "Age" Özelliğinin Değişimi',
              xaxis= dict(title= 'Kişi Sayısı',ticklen= 5,zeroline= False)) )


fig =go.Figure(data=İlişki_data , layout=layout)
fig
a = data[["Pregnancies","Outcome"]].groupby(["Pregnancies"], as_index = False).mean().sort_values(by="Outcome",ascending = False)
a
fig = px.bar(a , x = a.Pregnancies , y = a.Outcome)
fig.show()
x = data[["Glucose" , "Outcome"]].groupby(["Glucose"], as_index = False).mean().sort_values(by = "Glucose" , ascending = True)

x
fig = px.bar(x , x = x.Glucose , y = x.Outcome)
fig.show()
y = data[["BloodPressure" , "Outcome"]].groupby(["BloodPressure"] , as_index = False).mean().sort_values(by = "BloodPressure" , ascending = True)

y
fig = px.bar(y , x = y.BloodPressure , y = y.Outcome)
fig.show()
z = data[["SkinThickness" , "Outcome"]].groupby(["SkinThickness"] , as_index = False).mean().sort_values(by = "SkinThickness" , ascending = True)
z
fig = px.bar(z , x = z.SkinThickness , y = z.Outcome)
fig.show()
f = data[["Insulin" , "Outcome"]].groupby(["Insulin"] , as_index = False).mean().sort_values(by = "Insulin" , ascending = True)
f
fig = px.bar(f , x = f.Insulin , y = f.Outcome)
fig.show()
g = data[["BMI" , "Outcome"]].groupby(["BMI"] , as_index = False).mean().sort_values(by = "BMI" , ascending = True)
g
fig = px.bar(g , x = g.BMI , y = g.Outcome)
fig.show()
c = data[["DiabetesPedigreeFunction" , "Outcome"]].groupby(["DiabetesPedigreeFunction"] , as_index = False).mean().sort_values(by = "DiabetesPedigreeFunction" , ascending = True)
c
fig = px.bar(c , x = c.DiabetesPedigreeFunction , y = c.Outcome)
fig.show()
b = data[["Age" , "Outcome"]].groupby(["Age"] , as_index = False).mean().sort_values(by = "Age" , ascending = True)
b
fig = px.bar(b , x = b.Age , y = b.Outcome)
fig.show()
sns.heatmap(data.corr() , annot = True)
plt.show()
def outlier(df,features):
    outlier_indices = []
    
    for i in features:
        Q1 = np.percentile(df[i] , 25)
        Q3 = np.percentile(df[i] , 75)
        IQR = Q3-Q1
        outlier_step = IQR*1.5
        
        outlier_list_col = df[(df[i] < Q1 - outlier_step) | (df[i] > Q3 + outlier_step)].index
        outlier_indices.extend(outlier_list_col)
        
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)
    
    return multiple_outliers
data.loc[outlier(data,["Pregnancies" , "Glucose" , "BloodPressure" , "SkinThickness" , "Insulin" , "BMI" , "DiabetesPedigreeFunction" , "Age"])]

data.isnull().sum()
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

x = data.drop("Outcome" , axis = 1 )
y = data.Outcome

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 42)
print("X_train: {}".format(X_train.shape))
print("X_test: {}".format(X_test.shape))
print("y_train: {}".format(y_train.shape))
print("y_test: {}".format(y_test.shape))
X_train = (X_train- np.min(X_train))/ (np.max(X_train)-np.min(X_train)).values
X_test = (X_test- np.min(X_test))/ (np.max(X_test)-np.min(X_test)).values

#To run algorithm

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier


#To develop hyperparameter

from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV

#to find the accuracy rate

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score

dc = DecisionTreeClassifier()
dc.fit(X_train , y_train)
print("Score:" , dc.score(X_test , y_test) )
score = []
for i in range(1,50):
    rf = RandomForestClassifier(n_estimators = i , random_state = 42)
    rf.fit(X_train , y_train)
    score.append(rf.score(X_test , y_test))
    
plt.plot(score)
plt.show()
rf = RandomForestClassifier(n_estimators = 33 , random_state = 42)
rf.fit(X_train , y_train)
print("Random Forest Score: {}".format(rf.score(X_test ,y_test)))
model = LogisticRegression()
solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ['l1' , 'l2']
c_values = [100, 10, 1.0, 0.1, 0.01]

grid = dict(solver=solvers,penalty=penalty,C=c_values)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X_train, y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
model = KNeighborsClassifier()
n_neighbors = range(1, 21, 2)
weights = ['uniform', 'distance']
metric = ['euclidean', 'manhattan', 'minkowski']

grid = dict(n_neighbors=n_neighbors,weights=weights,metric=metric)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X_train, y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
model = SVC()
kernel = ['poly', 'rbf', 'sigmoid']
C = [50, 10, 1.0, 0.1, 0.01]
gamma = ['scale']
# define grid search
grid = dict(kernel=kernel,C=C,gamma=gamma)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X_train, y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
nb = GaussianNB()
nb.fit(X_train , y_train)

print("Print Accuracy Of Naive Bayes Algorithm:" , nb.score(X_test , y_test))