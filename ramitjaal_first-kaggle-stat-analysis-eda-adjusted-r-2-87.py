import numpy as np
import pandas as pd
import plotly.plotly as py
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.offline as ply
import plotly.graph_objs as go
from plotly import tools
init_notebook_mode(connected=True)
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn import utils
import statsmodels.formula.api as sm
sns.set(style= "whitegrid")
from  plotly.offline import plot
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()
from sklearn.neighbors import KNeighborsRegressor
df = pd.read_csv("../input/kc_house_data.csv")

df.head(5).transpose()
df.info()
df.describe().transpose()
corr_mat = df.corr()
plt.figure(figsize=(20,10))
sns.heatmap(corr_mat, cmap = 'coolwarm', linecolor = 'white', linewidth = 1, annot=True)
plt.figure(figsize=(12,5))
sns.distplot(df['price'])
plt.figure(figsize = (5,5))
sns.distplot(df['grade'] , kde = False)
bb = df['bedrooms'].value_counts()
index = [3,4,2,5,6,1,7,8,0,9,10,11,33]
trace = go.Pie(labels = index, values=bb.values)
ply.iplot([trace])
bbath = df['bathrooms'].value_counts()
indexbath =[2.5,1,1.75,2.25,2,1.5,2.75,3,3.5,3.25,3.75,4,4.5,4.25,0.75,4.75,5,5.25,0,5.5,1.25,6,.5,5.75,8,6.25,6.5,6.75
,7.5,7.75]
tracebath = go.Pie(labels = indexbath, values=bbath.values)
ply.iplot([tracebath])
trace1 = go.Scattergl(x=df['sqft_living'], y=df['price'], mode='markers', name='sqft_living')
trace2 = go.Scattergl(x=df['bedrooms'], y=df['price'], mode = 'markers', name = 'bedrooms')
trace3 = go.Scattergl(x=df['bathrooms'], y=df['price'],mode = 'markers', name = 'bathrooms')
trace4 = go.Scattergl(x=df['grade'], y=df['price'],mode = 'markers', name = 'grade')
fig = tools.make_subplots(rows=2, cols=2, subplot_titles=('sqft_living vs Price', 'bedrooms vs Price',
'bathrooms vs Price', 'grade vs Price'))
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
fig.append_trace(trace3, 2, 1)
fig.append_trace(trace4, 2, 2)
fig['layout'].update(height=800, width=800, title='Price Subplots')
ply.iplot(fig)
plt.figure(figsize=(20,10))
x = sns.scatterplot(x=df['price'], y=df["sqft_living"],
hue=df['condition'],
palette="coolwarm",
sizes=(4, 16), linewidth=0)
plt.setp(x.get_legend().get_texts(), fontsize='22')
plt.setp(x.get_legend().get_title(), fontsize='32')
plt.show()
df.isnull().values.any()
df.drop(["id"], axis = 1, inplace=True)
df.drop(["date"], axis = 1, inplace=True)
df = df.drop(df[df["bedrooms"]>10].index )
bed = pd.get_dummies(df["bedrooms"])
df.columns
df_pre1 = df[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'grade', 'sqft_above',
'sqft_basement', 'yr_built', 'zipcode', 'lat', 'long','sqft_living15', 'sqft_lot15', 'price']]
ss = StandardScaler()
df_pre = ss.fit_transform(df_pre1)
df_pre = pd.DataFrame(df_pre, columns= df_pre1.columns)
eigen_value, eigen_vector = np.linalg.eig(corr_mat)
print(eigen_value.round(3))
X_s = np.array(df_pre['sqft_living']).reshape(-1,1)
y_s = np.array(df_pre['price']).reshape(-1,1)
X_trains , X_tests, y_trains, y_tests = train_test_split(X_s, y_s , test_size = 0.3, random_state = 101)
lms = LinearRegression()
lms.fit(X_trains, y_trains)
per_lms = lms.predict(X_tests)
residual = (X_tests - per_lms )
sns.distplot(residual)
print(lms.coef_)
rsquared_sl = metrics.r2_score(y_tests,per_lms)
adjusted_r_squared_sl = 1 - (1-rsquared_sl)*(len(y_tests)-1)/(len(y_tests)-X_tests.shape[1]-1)
print('Mean absolute error: {}'.format(metrics.mean_absolute_error(y_tests, per_lms)))
print('Mean squared error: {}'.format(metrics.mean_squared_error(y_tests, per_lms)))
print('Root mean squared error: {}'.format(np.sqrt(metrics.mean_squared_error(y_tests, per_lms))))
print('R Squared value: {}'.format(rsquared_sl))
print('Adjusted R Squared Value: {}'.format(adjusted_r_squared_sl))
X = df_pre[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'grade', 'sqft_above',
'sqft_basement', 'yr_built', 'zipcode', 'lat', 'long','sqft_living15', 'sqft_lot15']]
y = df_pre["price"]
X_train , X_test, y_train, y_test = train_test_split(X, y , test_size = 0.3, random_state = 101)
lm = LinearRegression()
lm.fit(X_train, y_train)
per_lm = lm.predict(X_test)
residuals = (y_test- per_lm)
sns.distplot(residuals)
rsquared_ml = metrics.r2_score(y_test,per_lm)
adjusted_r_squared_ml = 1 - (1-rsquared_ml)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
print('Mean absolute error: {}'.format(metrics.mean_absolute_error(y_test, per_lm)))
print('Mean squared error: {}'.format(metrics.mean_squared_error(y_test, per_lm)))
print('Root mean squared error: {}'.format(np.sqrt(metrics.mean_squared_error(y_test, per_lm))))
print('R Squared value: {}'.format(rsquared_ml))
print('Adjusted R Squared Value: {}'.format(adjusted_r_squared_ml))
trace0 = go.Scatter(x = X_test['sqft_living'],y = y_test,mode = 'markers',name = 'Test Set')
trace1 = go.Scatter(x = X_test['sqft_living'],y = per_lm,opacity = 0.75,mode = 'markers',name = 'Predictions',marker = dict(line = dict(color = 'black', width = 0.5)))
data = [trace0, trace1]
ply.iplot(data)
lm2 = sm.ols(formula = 'price ~ bedrooms+bathrooms+sqft_living+sqft_lot+waterfront+view+condition+grade+sqft_above+sqft_basement+yr_built+yr_renovated+zipcode+lat+long+sqft_living15+sqft_lot15', data = df).fit()
lm2.summary()
Xb = df[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'waterfront','view','condition', 'grade',
'sqft_above',
'sqft_basement', 'yr_built','yr_renovated', 'zipcode', 'lat', 'long','sqft_living15', 'sqft_lot15']]
Xb = ss.fit_transform(Xb)
yb = df_pre["price"]
Xb_train , Xb_test, yb_train, yb_test = train_test_split(Xb, yb , test_size = 0.3, random_state = 101)
lmb = LinearRegression()
lmb.fit(Xb_train, yb_train)
per_lmb = lmb.predict(Xb_test)
residuals = (yb_test- per_lmb)
sns.distplot(residuals)
rsquared_mlb = metrics.r2_score(yb_test,per_lmb)
adjusted_r_squared_mlb = 1 - (1-rsquared_mlb)*(len(yb_test)-1)/(len(yb_test)-Xb_test.shape[1]-1)
print('Mean absolute error: {}'.format(metrics.mean_absolute_error(yb_test, per_lmb)))
print('Mean squared error: {}'.format(metrics.mean_squared_error(yb_test, per_lmb)))
print('Root mean squared error: {}'.format(np.sqrt(metrics.mean_squared_error(yb_test, per_lmb))))
print('R Squared value: {}'.format(rsquared_mlb))
print('Adjusted R Squared Value: {}'.format(adjusted_r_squared_mlb))
trace0 = go.Scatter(
x = Xb_test[:,2],
y = yb_test,
mode = 'markers',
name = 'Test Set'
)
trace1 = go.Scatter(
x = Xb_test[:,2],
y = per_lmb,
opacity = 0.75,
mode = 'markers',
name = 'Predictions',
marker = dict(line = dict(color = 'black', width = 0.5))
)
data = [trace0, trace1]
ply.iplot(data)
knn = KNeighborsRegressor(n_neighbors=1)
knn.fit(X_train,y_train)
knn_pred = knn.predict(X_test)
print('Mean absolute error: {}'.format(metrics.mean_absolute_error(y_test, knn_pred)))
print('Mean squared error: {}'.format(metrics.mean_squared_error(y_test, knn_pred)))
print('Root mean squared error: {}'.format(np.sqrt(metrics.mean_squared_error(y_test, knn_pred))))
rsquared_knn = metrics.r2_score(y_test,knn_pred)
adjusted_r_squared_knn = 1 - (1-rsquared_knn)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
print('R Squared: {}'.format(rsquared_knn))
print('Adjusted R Squared: {}'.format(adjusted_r_squared_knn))
RMSE = []
for i in range(1,20):
    knn = KNeighborsRegressor(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    RMSE.append((np.sqrt(metrics.mean_squared_error(y_test, pred_i))))
print('Minimum Root Mean Squared Error is {} with {} neighbors'.format(round(min(RMSE),3),RMSE.index(min(RMSE))+1))
trace = go.Scatter(
x = np.arange(1,20),
y = np.round(RMSE,3),
marker = dict(
size = 10,
color = 'rgba(255, 182, 193, .9)'),
line = dict( color = 'blue'),
mode = 'lines+markers'
)
layout = dict(title = 'RMSE vs Neighbors',
xaxis = dict(title = 'Number of neighbors',zeroline = False),
yaxis = dict(title = 'RMSE',zeroline = False)
)
data = [trace]
fig = dict(data = data, layout = layout)
ply.iplot(fig)
knn = KNeighborsRegressor(n_neighbors=8)
knn.fit(X_train,y_train)
pred_8 = knn.predict(X_test)
rsquared_8 = metrics.r2_score(y_test,pred_8)
adjusted_r_squared_8 = 1 - (1-rsquared_8)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
print('Mean absolute error: {}'.format(metrics.mean_absolute_error(y_test, pred_8)))
print('Mean squared error: {}'.format(metrics.mean_squared_error(y_test, pred_8)))
print('Root mean squared error: {}'.format(np.sqrt(metrics.mean_squared_error(y_test, pred_8))))
print('R squared value with 8 neighbors: {}'.format(rsquared_8))
print('Adjusted squared value with 8 neighbors: {}'.format(adjusted_r_squared_8))

residualk = (y_test- pred_8)
sns.distplot(residualk)
trace0 = go.Scatter(
x = X_test['sqft_living'],
y = y_test,
mode = 'markers',
name = 'Test Set'
)
trace1 = go.Scatter(
x = X_test['sqft_living'],
y = pred_8,
opacity = 0.75,
mode = 'markers',
name = 'Predictions',
marker = dict(line = dict(color = 'black', width = 0.5))
)
data = [trace0, trace1]
ply.iplot(data)
Xr = df[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'waterfront','view','condition', 'grade',
'sqft_above',
'sqft_basement', 'yr_built','yr_renovated', 'zipcode', 'lat', 'long','sqft_living15', 'sqft_lot15']]
Xr = ss.fit_transform(Xr)
yr = df_pre["price"]
Xr_train , Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size = 0.3, random_state = 101)
from sklearn.ensemble import RandomForestRegressor
rfc = RandomForestRegressor(n_estimators = 1)
rfc.fit(Xr_train,yr_train)
rfc_pred = rfc.predict(Xr_test)
rsquared = metrics.r2_score(yr_test,rfc_pred)
adjusted_r_squared = 1 - (1-rsquared)*(len(yr_test)-1)/(len(yr_test)-Xr_test.shape[1]-1)
print('Mean absolute error: {}'.format(metrics.mean_absolute_error(yr_test, rfc_pred)))
print('Mean squared error: {}'.format(metrics.mean_squared_error(yr_test, rfc_pred)))
print('Root mean squared error: {}'.format(np.sqrt(metrics.mean_squared_error(yr_test, rfc_pred))))
print('R Squared value: {}'.format(rsquared))
print('Adjusted R Squared Value: {}'.format(adjusted_r_squared))
RMSE_rfc = []
for i in range(1,50):
    rfc = RandomForestRegressor(n_estimators=i)
    rfc.fit(Xr_train,yr_train)
    pred_i = rfc.predict(Xr_test)
    RMSE_rfc.append((np.sqrt(metrics.mean_squared_error(yr_test, pred_i))))
print('Minimum Root Mean Squared Error is {} with {} estimators'.format(round(min(RMSE_rfc),3),RMSE_rfc.index(min(RMSE_rfc))+1))
trace = go.Scatter(
x = np.arange(1,100),
y = np.round(RMSE_rfc,3),
marker = dict(
size = 10,
color = 'rgba(255, 182, 193, .9)'),
line = dict( color = 'blue'),
mode = 'lines+markers'
)
layout = dict(title = 'RMSE vs Estimators',
xaxis = dict(title = 'Number of estimators',zeroline = False),
yaxis = dict(title = 'RMSE',zeroline = False)
)
data = [trace]
fig = dict(data = data, layout = layout)
ply.iplot(fig)
rfc = RandomForestRegressor(n_estimators=40)
rfc.fit(Xr_train,yr_train)
pred_p = rfc.predict(Xr_test)
rsquared_p = metrics.r2_score(yr_test,pred_p)
adjusted_r_squared_p = 1 - (1-rsquared_p)*(len(yr_test)-1)/(len(yr_test)-Xr_test.shape[1]-1)
print('Mean absolute error: {}'.format(metrics.mean_absolute_error(yr_test, pred_p)))
print('Mean squared error: {}'.format(metrics.mean_squared_error(yr_test, pred_p)))
print('Root mean squared error: {}'.format(np.sqrt(metrics.mean_squared_error(yr_test, pred_p))))
print('R squared value: {}'.format(rsquared_p))
print('Adjusted squared value: {}'.format(adjusted_r_squared_p))
residualr = (yr_test- pred_p)
sns.distplot(residualr)
trace0 = go.Scatter(
x = Xr_test[:,2],
y = yr_test,
mode = 'markers',
name = 'Test Set'
)
trace1 = go.Scatter(
x = Xr_test[:,2],
y = pred_8,
opacity = 0.75,
mode = 'markers',
name = 'Predictions',
marker = dict(line = dict(color = 'black', width = 0.5))
)
data = [trace0, trace1]
ply.iplot(data)