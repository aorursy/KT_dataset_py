#bliblioteca para facilitar as operações vetoriais e matriciais 
import numpy as np

#biblioteca para facilitar a organização do dataset
import pandas as pd

#blibliotecas de visualização gráfica
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
%matplotlib inline
import seaborn as sns

#biblioteca de machine learning
import sklearn
data_raw = pd.read_csv("../input/train.csv")
data_raw.head(10)
data_raw.info()
data_raw.describe()
len(data_raw.dropna()) == len(data_raw)
data_no_id = data_raw.drop(["Id"], axis=1)
data_no_id.head()
labels = data_no_id.drop(["median_house_value"], axis=1)
sns.violinplot(y=labels.longitude)
sns.violinplot(y=labels.latitude)
sns.jointplot(labels.longitude, labels.latitude, labels, kind='scatter', color=(0,0,1,0.2));
from IPython.display import Image
from IPython.core.display import HTML 
Image(url= "http://mala-beads.com/wp-content/uploads/2018/07/California-Lat-Long-Map-Photo-In-California-Latitude-Longitude-Map.jpg", height=600, width=540)
sns.jointplot(labels.longitude, labels.latitude, labels, kind='hex', color=(0.3,0.3,0.5,0.1));
sns.boxplot(y=labels["median_age"])
sns.violinplot(y=labels["median_age"])


sns.boxplot(y=labels.total_rooms)
sns.distplot(labels.total_rooms)
sns.violinplot(labels.total_bedrooms)
sns.distplot(labels.total_bedrooms)
sns.violinplot(labels.population)
sns.violinplot(labels.households)
sns.violinplot(labels.median_income)
target = data_no_id["median_house_value"]
sns.distplot(target)
sns.violinplot(y=target)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

xs = labels.longitude
ys = target
zs = labels.latitude
ax.scatter(xs, ys, zs, s=20, alpha=0.08)

ax.set_xlabel('Longitude')
ax.set_ylabel('Preço')
ax.set_zlabel('Latitude')

ax.view_init(45, 315)
plt.draw()
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

xs = labels.longitude
ys = target
zs = labels.latitude
ax.scatter(xs, ys, zs, s=20, alpha=0.08)

ax.set_xlabel('Longitude')
ax.set_ylabel('Preço')
ax.set_zlabel('Latitude')

ax.view_init(0, 270)
plt.draw()
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

xs = labels.longitude
ys = target
zs = labels.latitude
ax.scatter(xs, ys, zs, s=20, alpha=0.08)

ax.set_xlabel('Longitude')
ax.set_ylabel('Preço')
ax.set_zlabel('Latitude')

ax.view_init(0, 180)
plt.draw()
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

xs = labels.longitude
ys = target
zs = labels.latitude
ax.scatter(xs, ys, zs, s=20, alpha=0.08)

ax.set_xlabel('Longitude')
ax.set_ylabel('Preço')
ax.set_zlabel('Latitude')

ax.view_init(270, 90)
plt.draw()
sns.heatmap(data_no_id.corr(), annot=False, linewidth=0.3)
sns.jointplot(labels.median_income, target, kind="hex")
sns.jointplot(labels.population, target, kind="hex")
sns.jointplot(labels.median_age, target, kind="hex")
sns.jointplot(labels.total_rooms, target, kind="hex")
sns.jointplot(labels.longitude, target, kind="hex")
sns.jointplot(labels.latitude, target, kind="hex")
print("Los Angeles County:")
Image(url="https://upload.wikimedia.org/wikipedia/commons/thumb/3/33/Map_of_California_highlighting_Los_Angeles_County.svg/2000px-Map_of_California_highlighting_Los_Angeles_County.svg.png", width=250, height=286)
print("San Diego County:")
Image(url="https://www.familysearch.org/wiki/en/images/thumb/5/5f/California_San_Diego_Map.png/200px-California_San_Diego_Map.png")
print("Orange County:")
Image(url="data:image/jpeg;base64,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")
print("Sacramento County:")
Image(url="https://upload.wikimedia.org/wikipedia/commons/thumb/1/13/Map_of_California_highlighting_Sacramento_County.svg/1920px-Map_of_California_highlighting_Sacramento_County.svg.png", width=300, height=300)
SACRA_LAT = 38.575764
SACRA_LONG = -121.478851

LA_LAT = 34.053889
LA_LONG = -118.245

SD_LAT = 32.715736
SD_LONG = -117.161087

ANA_LAT = 33.835293
ANA_LONG = -117.914505
labels["dist_to_la"] = ((labels.latitude - LA_LAT)**2 + (labels.longitude - LA_LONG)**2)**0.5
labels["dist_to_sacramento"] = ((labels.latitude - SACRA_LAT)**2 + (labels.longitude - SACRA_LONG)**2)**0.5
labels["dist_to_sd"] = ((labels.latitude - SD_LAT)**2 + (labels.longitude - SD_LONG)**2)**0.5
labels["dist_to_ana"] = ((labels.latitude - ANA_LAT)**2 + (labels.longitude - ANA_LONG)**2)**0.5
labels.sample(3)
data_no_id["dist_to_la"] = ((data_no_id.latitude - LA_LAT)**2 + (data_no_id.longitude - LA_LONG)**2)**0.5
data_no_id["dist_to_sacramento"] = ((data_no_id.latitude - SACRA_LAT)**2 + (data_no_id.longitude - SACRA_LONG)**2)**0.5
data_no_id["dist_to_sd"] = ((data_no_id.latitude - SD_LAT)**2 + (data_no_id.longitude - SD_LONG)**2)**0.5
data_no_id["dist_to_ana"] = ((data_no_id.latitude - ANA_LAT)**2 + (data_no_id.longitude - ANA_LONG)**2)**0.5

sns.heatmap(data_no_id[["dist_to_la", "dist_to_sacramento", "dist_to_sd", "dist_to_ana", "median_house_value"]].corr(), annot=True, linewidths=.5)
teste_raw = pd.read_csv("../input/test.csv")
teste_raw["dist_to_la"] = ((teste_raw.latitude - LA_LAT)**2 + (teste_raw.longitude - LA_LONG)**2)**0.5
teste_raw["dist_to_sacramento"] = ((teste_raw.latitude - SACRA_LAT)**2 + (teste_raw.longitude - SACRA_LONG)**2)**0.5
teste_raw["dist_to_sd"] = ((teste_raw.latitude - SD_LAT)**2 + (teste_raw.longitude - SD_LONG)**2)**0.5
teste_raw["dist_to_ana"] = ((teste_raw.latitude - ANA_LAT)**2 + (teste_raw.longitude - ANA_LONG)**2)**0.5
teste_raw.sample(3)
teste = teste_raw.drop(["Id"], axis=1)
teste.sample(3)
# árvore de decisão
from sklearn.tree import DecisionTreeRegressor

# regressor com KNN
from sklearn.neighbors import KNeighborsRegressor

# regressor linear Ridge
from sklearn.linear_model import Ridge, RidgeCV

# regressor linear Lasso
from sklearn.linear_model import Lasso, LassoCV


# Biblioteca de validação cruzada
from sklearn.model_selection import cross_val_score
rcv = RidgeCV().fit(labels, target)
rcv.score(labels, target)
ridge = Ridge()
ridge.fit(labels, target)
pred_ridge = ridge.predict(teste)
resp_ridge = pd.DataFrame(list(zip(teste_raw.Id, pred_ridge)), columns= ['Id','median_house_value'])
resp_ridge = np.abs(resp_ridge.set_index("Id"))

resp_ridge.to_csv("resp_ridge.csv")
lcv = LassoCV().fit(labels, target)
lcv.score(labels, target)
lasso = Lasso(max_iter = 10000)
lasso.fit(labels, target)
pred_lasso = lasso.predict(teste)
resp_lasso = pd.DataFrame(list(zip(teste_raw.Id, pred_lasso)), columns= ['Id','median_house_value'])
resp_lasso = np.abs(resp_lasso.set_index("Id"))
resp_lasso.to_csv("resp_lasso.csv")
lista = []
for i in range(1,51):
    knnr = KNeighborsRegressor (n_neighbors=i)
    scores = cross_val_score(estimator=knnr, X=labels, y=target, cv=10)
    lista.append((i,scores.mean()))
print(lista)
best_n = 14
best_knnr = KNeighborsRegressor(n_neighbors=best_n)
best_knnr.fit(labels, target)
pred_knn = best_knnr.predict(teste)

resp_knnr = pd.DataFrame(list(zip(teste_raw.Id, pred_knn)), columns= ['Id','median_house_value'])
resp_knnr = np.abs(resp_knnr.set_index("Id"))

resp_knnr.to_csv("resp_knnr.csv")
depth = []
for i in range(3,20):
    tcv = DecisionTreeRegressor(max_depth=i) 
    scores = cross_val_score(estimator=tcv, X=labels, y=target, cv=10)
    depth.append((i,scores.mean()))
print(depth)
best_d = 9
best_tree = DecisionTreeRegressor(max_depth=best_d)
best_tree.fit(labels, target)
pred_tree = best_tree.predict(teste)
resp_tree = pd.DataFrame(list(zip(teste_raw.Id, pred_tree)), columns= ['Id','median_house_value'])
resp_tree = np.abs(resp_tree.set_index("Id"))

resp_tree.to_csv("resp_tree.csv")