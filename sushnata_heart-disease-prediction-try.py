# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scikitplot as skplt

from sklearn.model_selection import GridSearchCV

import sklearn.metrics as m

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Read Data from csv

dataset = pd.read_csv("/kaggle/input/heart-disease-dataset/heart.csv")

dataset
features = ["age","sex","thalach","trestbps"]

features
pulseratenormal = dataset[dataset["target"]==0]['trestbps']

np.average(pulseratenormal)
#Setting X and y

X = dataset[features]

y = dataset["target"]

print(X)

print(y)
ds = dataset[features+["target"]]

corr =d.corr()

corr.style.background_gradient(cmap='coolwarm')
plt.figure(figsize=(6,7))

plt.bar(dataset['target'].unique(), dataset['target'].value_counts(), color = ['red', 'green'])

plt.xticks([0, 1])

plt.xlabel('Target Classes')

plt.ylabel('Count')

plt.title('Count of each Target Class')
g = sns.catplot(x="thalach", y="trestbps", hue="target", kind="swarm", data=dataset)#, height=50, aspect=100/20)



g.fig.set_figwidth(20.27)

g.fig.set_figheight(11.7)
X.hist()
df = dataset[features+["target"]]

df = df[df["target"]==1]

df = df[features]

df.hist(color="red");
df = dataset[features+["target"]]

df = df[df["target"]==0]

df = df[features]

df.hist(color="green");
g = sns.catplot(x="thalach", y="trestbps", hue="target", kind="swarm", data=dataset)#, height=50, aspect=100/20)



#g.fig.set_figwidth(20.27)

#g.fig.set_figheight(11.7)

plt.xticks(np.arange(0, max(dataset['thalach']), 30 ))
sns.catplot(x="thalach", y="trestbps", hue="sex", kind="swarm", data=dataset)

plt.xticks(np.arange(0, max(dataset['thalach']), 30 ))
y.value_counts()
from sklearn.model_selection import train_test_split

#Splitting the dataset into training and testing data

trainX,testX,trainY,testY = train_test_split(X,y,random_state=1,test_size=0.2)
from sklearn import preprocessing

scaler = preprocessing.StandardScaler()

trainX = scaler.fit_transform(trainX)

trainX =pd.DataFrame(trainX)

trainX.columns = X.columns

testX = scaler.transform(testX)

testX =pd.DataFrame(testX)

testX.columns = X.columns



mm_scaler = preprocessing.MinMaxScaler()

trainX = mm_scaler.fit_transform(trainX)

trainX =pd.DataFrame(trainX)

trainX.columns = X.columns

testX = mm_scaler.transform(testX)

testX =pd.DataFrame(testX)

testX.columns = X.columns
#fit the model with knn classifier

from sklearn.neighbors import KNeighborsClassifier

model1 = KNeighborsClassifier(n_neighbors=1)



model1.fit(trainX,trainY)
from sklearn.model_selection import GridSearchCV

parameters = {'n_neighbors': np.arange(1, 10)}

grid_search = GridSearchCV(estimator = model1, param_grid = parameters, scoring = 'f1', cv = 10, n_jobs = -1)

grid_search = grid_search.fit(trainX, trainY)

best_accuracy = grid_search.best_score_

best_parameters = grid_search.best_params_

print(best_accuracy)

print(best_parameters)
#skplt.estimators.plot_feature_importances(model,feature_names=features)
#prediction

predictY = pd.Series(model1.predict(testX))
#getting accuracy

import sklearn.metrics as m

accuracy = m.accuracy_score(predictY,testY)

confusion = m.confusion_matrix(predictY,testY)

precision = m.precision_score(predictY,testY)

recall = m.recall_score(predictY,testY)

f1 = m.f1_score(predictY,testY)

print("accuracy = ",accuracy)

print("confusion matrix : \n",confusion)

print("precision = ",precision)

print("recall = ",recall)

print("f1 score = ",f1)
skplt.metrics.plot_confusion_matrix(testY,predictY,normalize=True)
from sklearn.svm import SVC

model2 = SVC(kernel = 'rbf',C=10, random_state = 0, gamma=0.18)

model2.fit(trainX,trainY)
c = [1, 10, 100]

import sklearn.metrics as m

accuracy=0

f1=0

for ci in c:

    model2 = SVC(kernel = 'linear', random_state = 0, C=ci)

    model2.fit(trainX,trainY)

    predictY = pd.Series(model2.predict(testX))

    pa = accuracy

    accuracy = m.accuracy_score(predictY,testY)

    pa = accuracy-pa

    pf1 = f1

    f1 = m.f1_score(predictY,testY)

    pf1 = f1 - pf1

    print("c = ",model2.C," f1 = ",f1,"accuracy = ",accuracy)

    print("pa = ",pa," pf1 = ",pf1)
parameters =[{'C': [1, 10, 100], 'kernel': ['rbf'], 'gamma': [ 0.10,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2]}]

grid_search = GridSearchCV(estimator = model2, param_grid = parameters, scoring = 'f1', cv = 10, n_jobs = -1)

grid_search = grid_search.fit(X,y)

best_accuracy = grid_search.best_score_

best_parameters = grid_search.best_params_

print(best_accuracy)

print(best_parameters)
predictY = pd.Series(model2.predict(testX))
#getting accuracy

accuracy = m.accuracy_score(predictY,testY)

confusion = m.confusion_matrix(predictY,testY)

precision = m.precision_score(predictY,testY)

recall = m.recall_score(predictY,testY)

f1 = m.f1_score(predictY,testY)

print("accuracy = ",accuracy)

print("confusion matrix : \n",confusion)

print("precision = ",precision)

print("recall = ",recall)

print("f1 score = ",f1)
skplt.metrics.plot_confusion_matrix(testY,predictY,normalize=True)
from sklearn.ensemble import RandomForestClassifier

model3 = RandomForestClassifier(min_samples_leaf=0.1,min_samples_split=0.1,max_depth=24,n_estimators=8)

model3.fit(trainX,trainY)
parameters =[{"n_estimators" : [4,8,16,32],

              "max_depth":range(1,16),

              "min_samples_split":np.linspace(0.1, 1.0, 10, endpoint=True),

              "min_samples_leaf" : np.linspace(0.1, 0.5, 10, endpoint=True)}]

grid_search = GridSearchCV(estimator = model3, param_grid = parameters, scoring = 'f1', cv = 10, n_jobs = -1)

grid_search = grid_search.fit(X,y)

best_accuracy = grid_search.best_score_

best_parameters = grid_search.best_params_

print(best_accuracy)

print(best_parameters)
predictY = pd.Series(model3.predict(testX))
#getting accuracy

accuracy = m.accuracy_score(predictY,testY)

confusion = m.confusion_matrix(predictY,testY)

precision = m.precision_score(predictY,testY)

recall = m.recall_score(predictY,testY)

f1 = m.f1_score(predictY,testY)

print("accuracy = ",accuracy)

print("confusion matrix : \n",confusion)

print("precision = ",precision)

print("recall = ",recall)

print("f1 score = ",f1)
from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(hidden_layer_sizes=(20,20,20), max_iter=500, alpha=0.0001,

                     solver='sgd', verbose=10,  random_state=21,tol=0.000000001)

clf.fit(trainX,trainY)
parameters = {

    'hidden_layer_sizes': [(50,50,50), (50,100,50),(20,20,20)],

    'activation': ['tanh', 'relu'],

    'solver': ['sgd', 'adam'],

    'alpha': [0.0001, 0.05],

    'learning_rate': ['constant','adaptive'],

}

grid_search = GridSearchCV(estimator = clf, param_grid = parameters, scoring = 'f1', cv = 10, n_jobs = -1)

grid_search = grid_search.fit(X,y)

best_accuracy = grid_search.best_score_

best_parameters = grid_search.best_params_

print(best_accuracy)

print(best_parameters)
predictY = clf.predict(testX)
#getting accuracy

accuracy = m.accuracy_score(predictY,testY)

confusion = m.confusion_matrix(predictY,testY)

precision = m.precision_score(predictY,testY)

recall = m.recall_score(predictY,testY)

f1 = m.f1_score(predictY,testY)

print("accuracy = ",accuracy)

print("confusion matrix : \n",confusion)

print("precision = ",precision)

print("recall = ",recall)

print("f1 score = ",f1)
from sklearn.neural_network import MLPClassifier

model3 = MLPClassifier(hidden_layer_sizes=(20,20,20), max_iter=500, alpha=0.0001,

                     solver='sgd', verbose=10,  random_state=21,tol=0.000000001)

model3.fit(trainX,trainY)