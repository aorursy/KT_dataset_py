import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import neighbors, tree, ensemble
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')
data = pd.read_csv('../input/train.csv')
data = data.dropna()
data.head()
X = data['longitude']
Y = data['latitude']
Z = data['median_house_value']
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(X, Y, Z)
fig.show()
X = data['median_age']
Y = data['median_house_value']
fig = plt.figure()
plt.scatter(X,Y)
plt.show()
X = data['total_rooms']
Y = data['median_house_value']
fig = plt.figure()
plt.scatter(X,Y)
plt.show()
X = data['total_bedrooms']
Y = data['median_house_value']
fig = plt.figure()
plt.scatter(X,Y)
plt.show()
X = data['population']
Y = data['median_house_value']
fig = plt.figure()
plt.scatter(X,Y)
plt.show()
X = data['households']
Y = data['median_house_value']
fig = plt.figure()
plt.scatter(X,Y)
plt.show()
X = data['median_income']
Y = data['median_house_value']
fig = plt.figure()
plt.scatter(X,Y)
plt.show()
n_neighbors = 100
knn = neighbors.KNeighborsClassifier(n_neighbors)
scores = cross_val_score(knn, data.iloc[:,1:9], data.iloc[:,9:10], cv=5)
scores
np.mean(scores)
clf = tree.DecisionTreeClassifier()
scores = cross_val_score(clf, data.iloc[:,1:9], data.iloc[:,9:10], cv=5)
scores
np.mean(scores)
clf = ensemble.RandomForestClassifier()
scores = cross_val_score(clf, data.iloc[:,1:9], data.iloc[:,9:10], cv=5)
scores
np.mean(scores)
def create_model():
    model = Sequential()
    model.add(Dense(12, input_dim = 8, activation = 'relu'))
    model.add(Dense(8, activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
model = KerasClassifier(build_fn=create_model, epochs=150, batch_size=5, verbose=0)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=16)
scores = cross_val_score(clf, data.iloc[:,1:9], data.iloc[:,9:10], cv=kfold)
scores
np.mean(scores)
clf = Lasso()
scores = cross_val_score(clf, data.iloc[:,1:9], data.iloc[:,9:10], cv=5)
scores
np.mean(scores)
minPrice = min(data['median_house_value'])
maxPrice= max(data['median_house_value'])
print(minPrice)
print(maxPrice)
step = (maxPrice-minPrice)/10
print(step)
steps = [round(minPrice + i*step) for i in range(11)]
def closestTo(value, listOfValues):
    i = 0
    closest = 0
    if value <= listOfValues[0]:
        closest = listOfValues[0]
        return closest
    elif value > listOfValues[-1]:
        closest = listOfValues[-1]
        return closest
    else:
        for i in range(len(listOfValues)-1):
            smaller = listOfValues[i]
            greater = listOfValues[i+1]
            if value <= listOfValues[i+1] and value > listOfValues[i]:
                break
    if abs(value-listOfValues[i]) > abs(value-listOfValues[i+1]):
        return listOfValues[i+1]
    else:
        return listOfValues[i]
        
def replaceFunction(value):
    return closestTo(value, steps)

data['median_house_value'] = data['median_house_value'].apply(replaceFunction)
n_neighbors = 41
knn = neighbors.KNeighborsClassifier(n_neighbors)
scores = cross_val_score(knn, data.iloc[:,1:9], data.iloc[:,9:10], cv=5)
print(np.mean(scores))
clf = Lasso()
scores = cross_val_score(clf, data.iloc[:,1:9], data.iloc[:,9:10], cv=5)
print(np.mean(scores))
data = pd.read_csv('../input/train.csv')
data = data.dropna()
from sklearn import linear_model
mmq = linear_model.LinearRegression()
ridge = linear_model.Ridge()
lasso = linear_model.Lasso()
mtLasso = linear_model.MultiTaskLasso()
en = linear_model.ElasticNet()
mtEn = linear_model.MultiTaskElasticNet()
lar = linear_model.Lars()
larsLasso = linear_model.LassoLars()
omp = linear_model.OrthogonalMatchingPursuit()
rb = linear_model.BayesianRidge()
sgd = linear_model.SGDRegressor()
modelDict = {'mmq':mmq, 'ridge':ridge, 'lasso':lasso, 'mtLasso':mtLasso, 'en':en, 'mtEn':mtEn, 'lar':lar, 'larsLasso':larsLasso, 'omp':omp, 'rb':rb, 'sgd':sgd}
for key in modelDict:
    print('Calculando acurácia do modelo {0}'.format(key))
    acc = np.mean(cross_val_score(modelDict[key], data.iloc[:,1:9], data.iloc[:,9:10], cv=5))
    print('Acurácia: {0}'.format(acc))
    
model = linear_model.Lars()
model.fit(data.iloc[:,1:9], data.iloc[:,9:10])

testData = pd.read_csv('../input/train.csv')

prediction = model.predict(testData.iloc[:,1:])
exportData = []

for index,element in enumerate(testData['Id']):
    toAppend = prediction[index]
    if toAppend < 0:
        toAppend = -1*toAppend
    exportData.append([element,toAppend])

import csv
csvFile = open('prediction.csv', 'w')
csvWriter = csv.writer(csvFile)
csvWriter.writerow(['Id','median_house_value'])
for element in exportData:
    csvWriter.writerow(element)
csvFile.close()
