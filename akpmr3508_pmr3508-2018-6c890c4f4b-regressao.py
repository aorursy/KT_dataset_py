import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from geopy.distance import distance
from geopy.distance import vincenty
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")
dfTrain = pd.read_csv("../input/californiahouse/train.csv",
          sep=r'\s*,\s*',
          engine='python',
          na_values="")
dfTest = pd.read_csv("../input/californiahouse/test.csv",
         sep=r'\s*,\s*',
         engine='python',
         na_values="")
dfTrain.shape
dfTest.shape
dfTrain = dfTrain.drop(columns=['Id'])
dfTrain.head()
dfTrain.describe()
dfTrain['median_house_value'].hist()
plt.figure(figsize=(6,6))
plt.title('Matriz de correlação')
sns.heatmap(dfTrain.corr(), annot=True, linewidths=0.1)
dfTrain = dfTrain.drop(['households', 'total_bedrooms'], axis=1)
plt.figure(figsize=(6,6))
plt.title('Matriz de correlação')
sns.heatmap(dfTrain.corr(), annot=True, linewidths=0.1)
SF_COORD = (37.7749, -122.4194)
LA_COORD = (34.0522, -118.2437)
SD_COORD = (32.7157, -117.1611)
def calc_dist_cities(row):
    local_coord = (row['latitude'], row['longitude'])
    row['min_dist'] = min(vincenty(local_coord, SF_COORD).km, vincenty(local_coord, LA_COORD).km, vincenty(local_coord, SD_COORD).km)
    return row

dfTrain = dfTrain.apply(calc_dist_cities, axis=1)
dfTrain = dfTrain.drop(['longitude', 'latitude'], axis=1)
plt.figure(figsize=(6,6))
plt.title('Matriz de correlação')
sns.heatmap(dfTrain.corr(), annot=True, linewidths=0.1)
def new_feat(row):
    row['rapp'] = row['total_rooms'] / row['population'] * row['median_age']
    return row

dfTrain = dfTrain.apply(new_feat, axis=1)
dfTrain = dfTrain.drop(['median_age', 'total_rooms', 'population'], axis=1)
plt.figure(figsize=(6,6))
plt.title('Matriz de correlação')
sns.heatmap(dfTrain.corr(), annot=True, linewidths=0.1)
scaler = MinMaxScaler()
selected_columns = ['median_income', 'min_dist', 'rapp']
dfScaled = scaler.fit_transform(dfTrain[selected_columns])
x_train, x_test, y_train, y_test = train_test_split(dfScaled, dfTrain['median_house_value'], test_size=0.20)
def rmsle(y_test, y_pred):
    return np.sqrt(np.mean((np.log(y_pred+1) - np.log(y_test+1))**2))

reg = LinearRegression()
scorer = make_scorer(rmsle, greater_is_better=False)
reg.fit(x_train, y_train)
y_pred = reg.predict(x_test)
print("RMSLE: " + str(rmsle(y_pred, y_test)))
param_grid = dict(n_neighbors=list(range(1,15)))
neigh = KNeighborsClassifier()
grid_obj = GridSearchCV(neigh, param_grid, scoring=scorer, cv=5)
grid_obj.fit(x_train, y_train)
grid_obj.best_params_
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(x_train, y_train)
y_pred = neigh.predict(x_test)
print("RMSLE: " + str(rmsle(y_pred, y_test)))
las = Lasso()
param_grid = dict(alpha=np.divide(list(range(1,100)),100))
grid_obj = GridSearchCV(las, param_grid, scoring=scorer, cv=5)
grid_obj.fit(x_train, y_train)
grid_obj.best_params_
las = Lasso(alpha=0.21)
las.fit(x_train, y_train)
y_pred = las.predict(x_test)
print("RMSLE: " + str(rmsle(y_pred, y_test)))
#rfc = RandomForestClassifier()
#param_grid = dict(n_estimators=np.multiply(list(range(1,12)), 5), max_depth=np.multiply(list(range(1,10)), 5))
#grid_obj = GridSearchCV(rfc, param_grid, scoring=scorer, cv=5)
#grid_obj.fit(x_train, y_train)
#grid_obj.best_params_
rfc = RandomForestClassifier(n_estimators=50, max_depth=35, random_state=0)
rfc.fit(x_train, y_train)
y_pred = rfc.predict(x_test)
print("RMSLE: " + str(rmsle(y_pred, y_test)))
dfTest = dfTest.apply(calc_dist_cities, axis=1)
dfTest = dfTest.apply(new_feat, axis=1)
dfTest = dfTest.drop(['longitude', 'latitude', 'median_age', 'total_rooms', 'population', 'households', 'total_bedrooms'], axis=1)
dfTest.head()
selected_model = las
x_val_test = scaler.transform(dfTest[selected_columns])
y_val_test = selected_model.predict(x_val_test)
dfSave = pd.DataFrame(data={"Id" : dfTest["Id"], "median_house_value" : y_val_test})
pd.DataFrame(dfSave[["Id", "median_house_value"]], columns = ["Id", "median_house_value"]).to_csv("Output.csv", index=False)