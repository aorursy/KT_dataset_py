# Import The Necessary Packages

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn import metrics
from sklearn import linear_model
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import GridSearchCV

from plotly.offline import init_notebook_mode,iplot
import plotly.graph_objs as go
init_notebook_mode(connected=True)

import warnings
warnings.filterwarnings("ignore")
data = "../input/housesalesprediction/kc_house_data.csv"
df = pd.read_csv(data)
df.head(10)
print("Data Shape: ", df.shape)
df.info()
df.describe().T
plt.subplots(figsize=(17,14))
sns.heatmap(df.corr(), annot=True, linewidth=0.5, linecolor="Black", fmt="1.1f")
plt.title("Attributes Correlation", fontsize=28)
plt.show()
hist1 = [go.Histogram(x=df.grade,marker=dict(color='rgb(102, 0, 102)'))]
histlayout1 = go.Layout(title="Grade Counts of Houses",xaxis=dict(title="Grades"),yaxis=dict(title="Counts"))
histfig1 = go.Figure(data=hist1,layout=histlayout1)
iplot(histfig1)
hist2 = [go.Histogram(x=df.yr_built,xbins=dict(start=np.min(df.yr_built),size=1,end=np.max(df.yr_built)),marker=dict(color='rgb(0,102,0)'))]

histlayout2 = go.Layout(title="Built Year Counts of Houses",xaxis=dict(title="Years"),yaxis=dict(title="Built Counts"))

histfig2 = go.Figure(data=hist2,layout=histlayout2)

iplot(histfig2)
v21 = [go.Box(y=df.bedrooms,name="Bedrooms",marker=dict(color="rgba(51,0,0,0.9)"),hoverinfo="name+y")]
v22 = [go.Box(y=df.bathrooms,name="Bathrooms",marker=dict(color="rgba(0,102,102,0.9)"),hoverinfo="name+y")]
v23 = [go.Box(y=df.floors,name="Floors",marker=dict(color="rgba(204,0,102,0.9)"),hoverinfo="name+y")]

layout2 = go.Layout(title="Bedrooms,Bathrooms and Floors",yaxis=dict(range=[0,13])) #I hate 33 bedroom

fig2 = go.Figure(data=v21+v22+v23,layout=layout2)
iplot(fig2)
import plotly.express as px

dataplus = df[np.logical_and(df.grade >= 7,df.yr_built >= 2000)] 
#list lat and long
lats = list(dataplus.lat.values)
longs = list(dataplus.long.values)

fig = px.scatter_mapbox(lat=lats, lon=longs, zoom=10, height=500)
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()
models_and_scores = []
X = df[['sqft_living']].values
y = df.price.values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=0.2,random_state=42)
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

model_score = lr.score(X_test,y_test)
r_square = metrics.r2_score(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
mae = metrics.median_absolute_error(y_test, y_pred)
ev = metrics.explained_variance_score(y_test, y_pred)

print("Model Score            : ", model_score*100)
print("R Square               : ", r_square*100)
print("Mean Squared Error     : ", mse)
print("Root Mean Squared Error: ", mse**(1/2))
print("Median Absolute Error  : ", mae)
print("Explained Variance     : ", ev)
new_df = df[['sqft_living15','lat','sqft_basement', 'sqft_above', 'grade','view','sqft_basement', 'sqft_living', 'bathrooms', 'floors', 'waterfront','yr_built']]
X = new_df.values
y = df.price.values

X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=0.2, random_state=42)
multi_lr_model = LinearRegression() # model

multi_lr_model.fit(X_train, y_train) # fit

y_pred = multi_lr_model.predict(X_test) # prediction

model_score = multi_lr_model.score(X_test,y_test)
r_square = metrics.r2_score(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
mae = metrics.median_absolute_error(y_test, y_pred)
ev = metrics.explained_variance_score(y_test, y_pred)
models_and_scores.append(["Multiple Linear",r_square])

print("Model Score            : ", model_score*100)
print("R Square               : ", r_square*100)
print("Mean Squared Error     : ", mse)
print("Root Mean Squared Error: ", mse**(1/2))
print("Median Absolute Error  : ", mae)
print("Explained Variance     : ", ev)
X1 = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y1 = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]

mymodel = np.poly1d(np.polyfit(X1, y1, 3))
myline = np.linspace(1, 22, 100)

plt.scatter(X1, y1)
plt.plot(myline, mymodel(myline))
plt.show()
speed = mymodel(17)
print(speed)
new_df = df[['sqft_living15','lat','sqft_basement', 'sqft_above', 'grade','view','sqft_basement', 'sqft_living', 'bathrooms', 'floors', 'waterfront','yr_built']]
X = new_df.values
y = df.price.values

X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=0.2, random_state=42)
from sklearn.tree import DecisionTreeRegressor

dtr = DecisionTreeRegressor(random_state=42)
dtr.fit(X_train, y_train)
y_pred = dtr.predict(X_test)

model_score = dtr.score(X_test,y_test)
r_square = metrics.r2_score(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
mae = metrics.median_absolute_error(y_test, y_pred)
ev = metrics.explained_variance_score(y_test, y_pred)

print("Model Score            : ", model_score*100)
print("R Square               : ", r_square*100)
print("Mean Squared Error     : ", mse)
print("Root Mean Squared Error: ", mse**(1/2))
print("Median Absolute Error  : ", mae)
print("Explained Variance     : ", ev)
# Model Tuning
param_grid = {
    'max_depth': list(np.arange(1,30)),
    'min_samples_split': list(np.arange(1,10))
}

dtr_model = DecisionTreeRegressor(random_state=42)
dtr_cv_model = GridSearchCV(dtr_model, param_grid, cv=10, n_jobs=-1, verbose=2)
dtr_cv_model.fit(X_train, y_train)
print("Best Params: ", dtr_cv_model.best_params_)
print("Best Score : ", dtr_cv_model.best_score_)

y_pred = dtr_cv_model.predict(X_test)

r_square = metrics.r2_score(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
mae = metrics.median_absolute_error(y_test, y_pred)
ev = metrics.explained_variance_score(y_test, y_pred)
models_and_scores.append(['Decision Tree', dtr_cv_model.best_score_])

print("R Square               : ", r_square*100)
print("Mean Squared Error     : ", mse)
print("Root Mean Squared Error: ", mse**(1/2))
print("Median Absolute Error  : ", mae)
print("Explained Variance     : ", ev)
columns = ['sqft_living15','lat','sqft_basement', 'sqft_above', 'grade','view','sqft_basement', 'sqft_living', 'bathrooms', 'floors', 'waterfront','yr_built']

Importance = pd.DataFrame({
    'Importance': dtr_cv_model.best_estimator_.feature_importances_*100}, index=columns)
Importance.sort_values(by="Importance", axis=0, ascending=True).plot(kind="barh", color="b")
plt.xlabel("Variable Importance")
plt.gca().legend_ = None
new_df = df[['sqft_living15','lat','sqft_basement', 'sqft_above', 'grade','view','sqft_basement', 'sqft_living', 'bathrooms', 'floors', 'waterfront','yr_built']]
X = new_df.values
y = df.price.values

X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=0.2, random_state=42)
from sklearn.ensemble import RandomForestRegressor


rfr = RandomForestRegressor(random_state=42)
rfr.fit(X_train, y_train)
y_pred = rfr.predict(X_test)

model_score = rfr.score(X_test,y_test)
r_square = metrics.r2_score(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
mae = metrics.median_absolute_error(y_test, y_pred)
ev = metrics.explained_variance_score(y_test, y_pred)

print("Model Score            : ", model_score*100)
print("R Square               : ", r_square*100)
print("Mean Squared Error     : ", mse)
print("Root Mean Squared Error: ", mse**(1/2))
print("Median Absolute Error  : ", mae)
print("Explained Variance     : ", ev)
# Model Tuning
param_grid = {
    'max_depth': [1,5,10,30,50,100]
}

rfr_model = RandomForestRegressor(random_state=42)
rfr_cv_model = GridSearchCV(rfr_model, param_grid, cv=10, n_jobs=-1, verbose=2)
rfr_cv_model.fit(X_train, y_train)
print("Best Params: ", rfr_cv_model.best_params_)
print("Best Score : ", rfr_cv_model.best_score_)

y_pred = rfr_cv_model.predict(X_test)

r_square = metrics.r2_score(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
mae = metrics.median_absolute_error(y_test, y_pred)
ev = metrics.explained_variance_score(y_test, y_pred)
models_and_scores.append(['Random Forest', rfr_cv_model.best_score_])


print("R Square               : ", r_square*100)
print("Mean Squared Error     : ", mse)
print("Root Mean Squared Error: ", mse**(1/2))
print("Median Absolute Error  : ", mae)
print("Explained Variance     : ", ev)
columns = ['sqft_living15','lat','sqft_basement', 'sqft_above', 'grade','view','sqft_basement', 'sqft_living', 'bathrooms', 'floors', 'waterfront','yr_built']

Importance = pd.DataFrame({
    'Importance': rfr_cv_model.best_estimator_.feature_importances_*100}, index=columns)
Importance.sort_values(by="Importance", axis=0, ascending=True).plot(kind="barh", color="b")
plt.xlabel("Variable Importance")
plt.gca().legend_ = None
new_df = df[['sqft_living15','lat','sqft_basement', 'sqft_above', 'grade','view','sqft_basement', 'sqft_living', 'bathrooms', 'floors', 'waterfront','yr_built']]
X = new_df.values
y = df.price.values

X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=0.2, random_state=42)
from sklearn.svm import SVR

svr = SVR(kernel='linear')
svr.fit(X_train, y_train)
y_pred = svr.predict(X_test)

model_score = svr.score(X_test,y_test)
r_square = metrics.r2_score(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
mae = metrics.median_absolute_error(y_test, y_pred)
ev = metrics.explained_variance_score(y_test, y_pred)

print("Model Score            : ", model_score*100)
print("R Square               : ", r_square*100)
print("Mean Squared Error     : ", mse)
print("Root Mean Squared Error: ", mse**(1/2))
print("Median Absolute Error  : ", mae)
print("Explained Variance     : ", ev)
# Model Tuning
param_grid = {
    'C': [0.1, 0.5, 1, 5]
}

svr = SVR(kernel='linear')
svr_cv_model = GridSearchCV(svr, param_grid, cv=5, n_jobs=-1, verbose=2)
svr_cv_model.fit(X_train, y_train)
print("Best Params: ", svr_cv_model.best_params_)
print("Best Score : ", svr_cv_model.best_score_)

y_pred = svr_cv_model.predict(X_test)

r_square = metrics.r2_score(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
mae = metrics.median_absolute_error(y_test, y_pred)
ev = metrics.explained_variance_score(y_test, y_pred)
models_and_scores.append(['SVM', svr_cv_model.best_score_])

print("R Square               : ", r_square*100)
print("Mean Squared Error     : ", mse)
print("Root Mean Squared Error: ", mse**(1/2))
print("Median Absolute Error  : ", mae)
print("Explained Variance     : ", ev)
new_df = df[['sqft_living15','lat','sqft_basement', 'sqft_above', 'grade','view','sqft_basement', 'sqft_living', 'bathrooms', 'floors', 'waterfront','yr_built']]
X = new_df.values
y = df.price.values

X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=0.2, random_state=42)
from xgboost import XGBRegressor

xgb = XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=7)
xgb.fit(X_train,y_train)
y_pred = xgb.predict(X_test)

model_score = xgb.score(X_test,y_test)
r_square = metrics.r2_score(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
mae = metrics.median_absolute_error(y_test, y_pred)
ev = metrics.explained_variance_score(y_test, y_pred)

print("Model Score            : ", model_score*100)
print("R Square               : ", r_square*100)
print("Mean Squared Error     : ", mse)
print("Root Mean Squared Error: ", mse**(1/2))
print("Median Absolute Error  : ", mae)
print("Explained Variance     : ", ev)
# Model Tuning
param_grid = {
    'learning_rate': [0.1, 0.01, 0.5],
    'max_depth': [2,3,5],
    'n_estimators': [100, 200, 300],
    'colsample_bytree': [0.4, 0.7, 1]
}

xgb_model = XGBRegressor()
xgb_cv_model = GridSearchCV(xgb_model, param_grid, cv=5, n_jobs=-1, verbose=2)
xgb_cv_model.fit(X_train, y_train)
print("Best Params: ", xgb_cv_model.best_params_)
print("Best Score : ", xgb_cv_model.best_score_)

y_pred = xgb_cv_model.predict(X_test)

r_square = metrics.r2_score(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
mae = metrics.median_absolute_error(y_test, y_pred)
ev = metrics.explained_variance_score(y_test, y_pred)
models_and_scores.append(['XGB', xgb_cv_model.best_score_])

print("R Square               : ", r_square*100)
print("Mean Squared Error     : ", mse)
print("Root Mean Squared Error: ", mse**(1/2))
print("Median Absolute Error  : ", mae)
print("Explained Variance     : ", ev)
columns = ['sqft_living15','lat','sqft_basement', 'sqft_above', 'grade','view','sqft_basement', 'sqft_living', 'bathrooms', 'floors', 'waterfront','yr_built']

Importance = pd.DataFrame({
    'Importance': xgb_cv_model.best_estimator_.feature_importances_*100}, index=columns)
Importance.sort_values(by="Importance", axis=0, ascending=True).plot(kind="barh", color="b")
plt.xlabel("Variable Importance")
plt.gca().legend_ = None
new_df = df[['sqft_living15','lat','sqft_basement', 'sqft_above', 'grade','view','sqft_basement', 'sqft_living', 'bathrooms', 'floors', 'waterfront','yr_built']]
X = new_df.values
y = df.price.values

X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=0.2, random_state=42)
from sklearn.linear_model import Lasso

lass = Lasso()
lass.fit(X_train, y_train)
y_pred = lass.predict(X_test)

model_score = lass.score(X_test,y_test)
r_square = metrics.r2_score(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
mae = metrics.median_absolute_error(y_test, y_pred)
ev = metrics.explained_variance_score(y_test, y_pred)

print("Model Score            : ", model_score*100)
print("R Square               : ", r_square*100)
print("Mean Squared Error     : ", mse)
print("Root Mean Squared Error: ", mse**(1/2))
print("Median Absolute Error  : ", mae)
print("Explained Variance     : ", ev)
# Model Tuning
from sklearn.linear_model import LassoCV

alphas = 10**np.linspace(10,-1,100)*0.5

lass_cv_model = LassoCV(alphas=alphas, cv=10, n_jobs=-1, verbose=2, max_iter=100000)
lass_cv_model.fit(X_train, y_train)

y_pred = lass_cv_model.predict(X_test)

r_square = metrics.r2_score(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
mae = metrics.median_absolute_error(y_test, y_pred)
ev = metrics.explained_variance_score(y_test, y_pred)
models_and_scores.append(['Lasso', r_square])

print("R Square               : ", r_square*100)
print("Mean Squared Error     : ", mse)
print("Root Mean Squared Error: ", mse**(1/2))
print("Median Absolute Error  : ", mae)
print("Explained Variance     : ", ev)
new_df = df[['sqft_living15','lat','sqft_basement', 'sqft_above', 'grade','view','sqft_basement', 'sqft_living', 'bathrooms', 'floors', 'waterfront','yr_built']]
X = new_df.values
y = df.price.values

X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=0.2, random_state=42)
from sklearn.linear_model import Ridge

ridge = Ridge()
ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)

model_score = ridge.score(X_test,y_test)
r_square = metrics.r2_score(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
mae = metrics.median_absolute_error(y_test, y_pred)
ev = metrics.explained_variance_score(y_test, y_pred)
models_and_scores.append(['Ridge', r_square])

print("Model Score            : ", model_score*100)
print("R Square               : ", r_square*100)
print("Mean Squared Error     : ", mse)
print("Root Mean Squared Error: ", mse**(1/2))
print("Median Absolute Error  : ", mae)
print("Explained Variance     : ", ev)
new_df = df[['sqft_living15','lat','sqft_basement', 'sqft_above', 'grade','view','sqft_basement', 'sqft_living', 'bathrooms', 'floors', 'waterfront','yr_built']]
X = new_df.values
y = df.price.values

X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=0.2, random_state=42)
from sklearn.linear_model import ElasticNet

elasticN = ElasticNet()
elasticN.fit(X_train, y_train)
y_pred = elasticN.predict(X_test)

model_score = elasticN.score(X_test,y_test)
r_square = metrics.r2_score(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
mae = metrics.median_absolute_error(y_test, y_pred)
ev = metrics.explained_variance_score(y_test, y_pred)
models_and_scores.append(['ElasticNet', r_square])

print("Model Score            : ", model_score*100)
print("R Square               : ", r_square*100)
print("Mean Squared Error     : ", mse)
print("Root Mean Squared Error: ", mse**(1/2))
print("Median Absolute Error  : ", mae)
print("Explained Variance     : ", ev)
new_df = df[['sqft_living15','lat','sqft_basement', 'sqft_above', 'grade','view','sqft_basement', 'sqft_living', 'bathrooms', 'floors', 'waterfront','yr_built']]
X = new_df.values
y = df.price.values

X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=0.2, random_state=42)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

model_score = knn.score(X_test,y_test)
r_square = metrics.r2_score(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
mae = metrics.median_absolute_error(y_test, y_pred)
ev = metrics.explained_variance_score(y_test, y_pred)
models_and_scores.append(['KNN', r_square])

print("Model Score            : ", model_score*100)
print("R Square               : ", r_square*100)
print("Mean Squared Error     : ", mse)
print("Root Mean Squared Error: ", mse**(1/2))
print("Median Absolute Error  : ", mae)
print("Explained Variance     : ", ev)
new_df = df[['sqft_living15','lat','sqft_basement', 'sqft_above', 'grade','view','sqft_basement', 'sqft_living', 'bathrooms', 'floors', 'waterfront','yr_built']]
X = new_df.values
y = df.price.values

X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=0.2, random_state=42)
from sklearn.ensemble import GradientBoostingRegressor

gbm = GradientBoostingRegressor(random_state=42)
gbm.fit(X_train, y_train)
y_pred = gbm.predict(X_test)

model_score = gbm.score(X_test,y_test)
r_square = metrics.r2_score(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
mae = metrics.median_absolute_error(y_test, y_pred)
ev = metrics.explained_variance_score(y_test, y_pred)

print("Model Score            : ", model_score*100)
print("R Square               : ", r_square*100)
print("Mean Squared Error     : ", mse)
print("Root Mean Squared Error: ", mse**(1/2))
print("Median Absolute Error  : ", mae)
print("Explained Variance     : ", ev)
# Model Tuning
"""
param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'max_depth': [3,5,8],
    'n_estimators': [100,200,500],
    'subsample': [1, 0.5, 0.8 ],
    'loss': ['ls', 'lad', 'quantile']
}
"""
param_grid = {
    'learning_rate': [0.001, 0.01],
    'max_depth': [3,5],
    'n_estimators': [150,200,250],
    'subsample': [1, 0.5]
}

gbm_model = GradientBoostingRegressor(random_state=42)
gbm_cv_model = GridSearchCV(gbm_model, param_grid, cv=10, n_jobs=-1, verbose=2)
gbm_cv_model.fit(X_train, y_train)
print("Best Params: ", gbm_cv_model.best_params_)
print("Best Score : ", gbm_cv_model.best_score_)

y_pred = gbm_cv_model.predict(X_test)

r_square = metrics.r2_score(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
mae = metrics.median_absolute_error(y_test, y_pred)
ev = metrics.explained_variance_score(y_test, y_pred)
models_and_scores.append(['GBM',  gbm_cv_model.best_score_])

print("Model Score            : ", model_score*100)
print("R Square               : ", r_square*100)
print("Mean Squared Error     : ", mse)
print("Root Mean Squared Error: ", mse**(1/2))
print("Median Absolute Error  : ", mae)
print("Explained Variance     : ", ev)
columns = ['sqft_living15','lat','sqft_basement', 'sqft_above', 'grade','view','sqft_basement', 'sqft_living', 'bathrooms', 'floors', 'waterfront','yr_built']

Importance = pd.DataFrame({
    'Importance': gbm_cv_model.best_estimator_.feature_importances_*100}, index=columns)
Importance.sort_values(by="Importance", axis=0, ascending=True).plot(kind="barh", color="b")
plt.xlabel("Variable Importance")
plt.gca().legend_ = None
new_df = df[['sqft_living15','lat','sqft_basement', 'sqft_above', 'grade','view','sqft_basement', 'sqft_living', 'bathrooms', 'floors', 'waterfront','yr_built']]
X = new_df.values
y = df.price.values

X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=0.2, random_state=42)
from lightgbm import LGBMRegressor

lgbm = LGBMRegressor(random_state=42)
lgbm.fit(X_train, y_train)
y_pred = lgbm.predict(X_test)

model_score = lgbm.score(X_test,y_test)
r_square = metrics.r2_score(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
mae = metrics.median_absolute_error(y_test, y_pred)
ev = metrics.explained_variance_score(y_test, y_pred)

print("Model Score            : ", model_score*100)
print("R Square               : ", r_square*100)
print("Mean Squared Error     : ", mse)
print("Root Mean Squared Error: ", mse**(1/2))
print("Median Absolute Error  : ", mae)
print("Explained Variance     : ", ev)
# Model Tuning
param_grid = {
    "learning_rate": [0.01, 0.1, 0.5, 1],
    "n_estimators": [20, 40, 100, 500, 1000],
    "max_depth": [-1,1,3,5]
}

lgbm_model = LGBMRegressor(random_state=42)
lgbm_cv_model = GridSearchCV(lgbm_model, param_grid, cv=10, n_jobs=-1, verbose=2)
lgbm_cv_model.fit(X_train, y_train)
print("Best Params: ", lgbm_cv_model.best_params_)
print("Best Score : ", lgbm_cv_model.best_score_)

y_pred = lgbm_cv_model.predict(X_test)

r_square = metrics.r2_score(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
mae = metrics.median_absolute_error(y_test, y_pred)
ev = metrics.explained_variance_score(y_test, y_pred)
models_and_scores.append(['LightGBM',  lgbm_cv_model.best_score_])

print("Model Score            : ", model_score*100)
print("R Square               : ", r_square*100)
print("Mean Squared Error     : ", mse)
print("Root Mean Squared Error: ", mse**(1/2))
print("Median Absolute Error  : ", mae)
print("Explained Variance     : ", ev)
columns = ['sqft_living15','lat','sqft_basement', 'sqft_above', 'grade','view','sqft_basement', 'sqft_living', 'bathrooms', 'floors', 'waterfront','yr_built']

Importance = pd.DataFrame({
    'Importance': lgbm_cv_model.best_estimator_.feature_importances_*100}, index=columns)
Importance.sort_values(by="Importance", axis=0, ascending=True).plot(kind="barh", color="b")
plt.xlabel("Variable Importance")
plt.gca().legend_ = None
from catboost import CatBoostRegressor

catb = CatBoostRegressor(random_state=42)
catb.fit(X_train, y_train)
y_pred = catb.predict(X_test)

model_score = catb.score(X_test,y_test)
r_square = metrics.r2_score(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
mae = metrics.median_absolute_error(y_test, y_pred)
ev = metrics.explained_variance_score(y_test, y_pred)

print("Model Score            : ", model_score*100)
print("R Square               : ", r_square*100)
print("Mean Squared Error     : ", mse)
print("Root Mean Squared Error: ", mse**(1/2))
print("Median Absolute Error  : ", mae)
print("Explained Variance     : ", ev)
# Model Tuning
param_grid = {
    "iterations": [200, 500, 1000],
    'learning_rate': [0.01, 0.1],
    "depth": [3,6,8]
}

catb_model = LGBMRegressor(random_state=42)
catb_cv_model = GridSearchCV(catb_model, param_grid, cv=10, n_jobs=-1, verbose=2)
catb_cv_model.fit(X_train, y_train)
print("Best Params: ", catb_cv_model.best_params_)
print("Best Score : ", catb_cv_model.best_score_)

y_pred = catb_cv_model.predict(X_test)

r_square = metrics.r2_score(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
mae = metrics.median_absolute_error(y_test, y_pred)
ev = metrics.explained_variance_score(y_test, y_pred)
models_and_scores.append(['CatBoost',  catb_cv_model.best_score_])

print("Model Score            : ", model_score*100)
print("R Square               : ", r_square*100)
print("Mean Squared Error     : ", mse)
print("Root Mean Squared Error: ", mse**(1/2))
print("Median Absolute Error  : ", mae)
print("Explained Variance     : ", ev)
models, scores = [], []

for x in models_and_scores:
    models.append(x[0])
    scores.append(x[1])

plt.figure(figsize=(15,10))
ax = sns.barplot(x=scores, y=models, palette="Blues_d")
ax.set_title("Models And Scores - Comparison")
ax.set_ylabel("Models")
ax.set_ylabel("Scores")
plt.show()