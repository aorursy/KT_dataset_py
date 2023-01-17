import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
data = pd.read_csv("../input/atividade-3-pmr3508/train.csv")
data.head()
data.shape
corrMatrix = np.array(data.corr())
features = data.columns.values.tolist()

fig, ax = plt.subplots(figsize=(9,9))
im = ax.imshow(corrMatrix)

ax.set_xticks(np.arange(len(features)))
ax.set_yticks(np.arange(len(features)))
ax.set_xticklabels(features)
ax.set_yticklabels(features)

plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")

for i in range(len(features)):
    for j in range(len(features)):
        text = ax.text(j,i,round(corrMatrix[i,j],2),ha="center",va="center",color="w",size=18)

ax.set_title("Matriz de correlação da base de dados")
fig.tight_layout()
plt.show()
newData = data.drop(["Id","latitude","longitude"],axis=1)
newData["avg_rooms"] = newData.total_rooms/newData.households
newData["avg_bedrooms"] = newData.total_bedrooms/newData.households
newData["avg_inhabitants"] = newData.population/newData.households
newCorrMatrix = np.array(newData.drop(["total_rooms","total_bedrooms","population"],axis=1).corr())

fig, ax = plt.subplots(figsize=(9,9))
im = ax.imshow(newCorrMatrix)
features = newData.drop(["total_rooms","total_bedrooms","population"],axis=1).columns.values.tolist()

ax.set_xticks(np.arange(len(features)))
ax.set_yticks(np.arange(len(features)))
ax.set_xticklabels(features)
ax.set_yticklabels(features)

plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")

for i in range(len(features)):
    for j in range(len(features)):
        text = ax.text(j,i,round(newCorrMatrix[i,j],2),ha="center",va="center",color="w",size=18)

ax.set_title("Matriz de correlação com as features auxiliares")
fig.tight_layout()
plt.show()
x = np.array(newData.median_age)[0:1000]
y = np.array(newData.median_house_value)[0:1000]
fig, ax = plt.subplots(figsize=(9,9))
ax.scatter(x,y)
fig, ax = plt.subplots(figsize=(9,9))
ax.hist(np.array(newData.avg_rooms),range=(0,20),color="xkcd:aquamarine")
roomData = newData[(newData.avg_rooms>=7)][(newData.avg_rooms<=20)]
x = np.array(roomData.avg_rooms)
y = np.array(roomData.median_house_value)
fig, ax = plt.subplots(figsize=(9,9))
ax.scatter(x,y,color="xkcd:aquamarine")
x = np.array(newData.median_income)[0:1000]
y = np.array(newData.median_house_value)[0:1000]
fig, ax = plt.subplots(figsize=(9,9))
ax.scatter(x,y,color="xkcd:plum")
x = np.array(newData.avg_rooms)
fig, ax = plt.subplots(figsize=(9,9))
ax.boxplot(x,vert=False)
trainData = newData[(newData.avg_rooms<=10)]
xTrain = trainData.drop(["median_house_value"],axis=1)
yTrain = trainData.median_house_value
xTrain.head()
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
knn = KNeighborsRegressor()
knn.get_params()
param_grid = {"n_neighbors":[i for i in range(1,31)],"weights":["uniform","distance"],"p":[1,2]}
grid = GridSearchCV(knn,param_grid,cv=10)
grid.fit(xTrain,yTrain)
print(grid.best_estimator_)
print(grid.best_score_)
knn = grid.best_estimator_
from sklearn.linear_model import Ridge
ridge = Ridge()
ridge.get_params()
param_grid2 = {"alpha":np.linspace(0.5,10.5,101).tolist()} 
grid2 = GridSearchCV(ridge,param_grid2,cv=10)
grid2.fit(xTrain,yTrain)
print(grid2.best_estimator_)
print(grid2.best_score_)
ridge = grid2.best_estimator_
from sklearn.linear_model import Lasso
lasso = Lasso()
lasso.get_params()
param_grid3 = {"alpha":np.linspace(0.5,5.5,51).tolist(),"normalize":[True,False]}
grid3 = GridSearchCV(lasso,param_grid3,cv=10)
grid3.fit(xTrain,yTrain)
print(grid3.best_estimator_)
print(grid3.best_score_)
lasso = grid3.best_estimator_
testRaw = pd.read_csv("../input/atividade-3-pmr3508/test.csv")
ID_list = testRaw.Id.tolist()
testRaw["avg_rooms"] = testRaw.total_rooms/testRaw.households
testRaw["avg_bedrooms"] = testRaw.total_bedrooms/testRaw.households
testRaw["avg_inhabitants"] = testRaw.population/testRaw.households
testData = testRaw.drop(["Id","latitude","longitude"],axis=1)
testData.head()
knn.fit(xTrain,yTrain)
pred_knn = knn.predict(testData).tolist()
pd.DataFrame({"Id":ID_list,"median_house_value":pred_knn}).to_csv("pred_knn.csv",index=False)
ridge.fit(xTrain,yTrain)
pred_ridge = ridge.predict(testData).tolist()
pd.DataFrame({"Id":ID_list,"median_house_value":pred_ridge}).to_csv("pred_ridge.csv",index=False)
lasso.fit(xTrain,yTrain)
pred_lasso = lasso.predict(testData).tolist()
pd.DataFrame({"Id":ID_list,"median_house_value":pred_lasso}).to_csv("pred_lasso.csv",index=False)