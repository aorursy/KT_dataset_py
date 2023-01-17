!pip install pycaret
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import pycaret
import pickle

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor
from xgboost import Booster
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate, GridSearchCV, RandomizedSearchCV
df = pd.read_csv('/kaggle/input/df-edited/dfnew.csv')
df.drop(['Unnamed: 0'],axis=1,inplace=True)
fig, ax = plt.subplots(2,3,figsize=(18,10))

sns.scatterplot(x='temp',y='cum_power',data=df,ax=ax[0,0],color='g')
sns.scatterplot(x='weather',y='cum_power',data=df,ax=ax[0,1],color='g')
sns.scatterplot(x='wind',y='cum_power',data=df,ax=ax[0,2],color='g')
sns.scatterplot(x='humidity',y='cum_power',data=df,ax=ax[1,0],color='g')
sns.scatterplot(x='barometer',y='cum_power',data=df,ax=ax[1,1],color='g')
sns.scatterplot(x='visibility',y='cum_power',data=df,ax=ax[1,2],color='g')

plt.show()
a = sns.jointplot(x='temp',y='cum_power',data=df,kind='kde',color='g')
b = sns.jointplot(x='weather',y='cum_power',data=df,kind='kde',color='g')
c = sns.jointplot(x='wind',y='cum_power',data=df,kind='kde',color='g')
d = sns.jointplot(x='humidity',y='cum_power',data=df,kind='kde',color='g')
e = sns.jointplot(x='barometer',y='cum_power',data=df,kind='kde',color='g')
f = sns.jointplot(x='visibility',y='cum_power',data=df,kind='kde',color='g')

plots = [a,b,c,d,e,f]

for plot in plots:
  plt.show()
plt.figure(figsize=(18,10))
sns.heatmap(df.corr(),
            vmin=-1,
            cmap='coolwarm',
            annot=True)
plt.show()
# Creating figure 
fig = plt.figure(figsize = (16, 9)) 
ax = plt.axes(projection ="3d")

# Add x, y gridlines  
ax.grid(b = True, color ='grey',  
        linestyle ='-.', linewidth = 0.3,  
        alpha = 0.2)  
  
# Creating color map 
my_cmap = plt.get_cmap('hsv') 
  
# Creating plot 
sctt = ax.scatter3D(df['temp'], df['humidity'], df['cum_power'], 
                    alpha = 0.8, 
                    c = df['cum_power']) 
  
plt.title('3D plot of Weather vs Visinilty vs Cum_Power with color legend of cum_power') 
ax.set_xlabel('temp', fontweight ='bold')  
ax.set_ylabel('humidity', fontweight ='bold')  
ax.set_zlabel('cum_power', fontweight ='bold') 
fig.colorbar(sctt, ax = ax, shrink = 0.5, aspect = 5) 
  
# show plot 
plt.show() 
# Creating figure 
fig = plt.figure(figsize = (16, 9)) 
ax = plt.axes(projection ="3d")

# Add x, y gridlines  
ax.grid(b = True, color ='grey',  
        linestyle ='-.', linewidth = 0.3,  
        alpha = 0.2)  
  
# Creating color map 
my_cmap = plt.get_cmap('hsv') 
  
# Creating plot 
sctt = ax.scatter3D(df['temp'], df['month'], df['humidity'], 
                    alpha = 0.8, 
                    c = df['cum_power']) 
  
plt.title('3D plot of Weather vs Month vs Visibility with colour legend of Cum_Power') 
ax.set_xlabel('temp', fontweight ='bold')  
ax.set_ylabel('month', fontweight ='bold')  
ax.set_zlabel('humidity', fontweight ='bold') 
fig.colorbar(sctt, ax = ax, shrink = 0.5, aspect = 5) 
  
# show plot 
plt.show() 
# Creating figure 
fig = plt.figure(figsize = (16, 9)) 
ax = plt.axes(projection ="3d")

# Add x, y gridlines  
ax.grid(b = True, color ='grey',  
        linestyle ='-.', linewidth = 0.3,  
        alpha = 0.2)  
  
# Creating color map 
my_cmap = plt.get_cmap('hsv') 
  
# Creating plot 
sctt = ax.scatter3D(df['weather'], df['visibility'], df['cum_power'], 
                    alpha = 0.8, 
                    c = df['cum_power']) 
  
plt.title('3D plot of Weather vs Visinilty vs Cum_Power with color legend of cum_power') 
ax.set_xlabel('humidity', fontweight ='bold')  
ax.set_ylabel('visibility', fontweight ='bold')  
ax.set_zlabel('cum_power', fontweight ='bold') 
fig.colorbar(sctt, ax = ax, shrink = 0.5, aspect = 5) 
  
# show plot 
plt.show() 
# Creating figure 
fig = plt.figure(figsize = (16, 9)) 
ax = plt.axes(projection ="3d")

# Add x, y gridlines  
ax.grid(b = True, color ='grey',  
        linestyle ='-.', linewidth = 0.3,  
        alpha = 0.2)  
  
# Creating color map 
my_cmap = plt.get_cmap('hsv') 
  
# Creating plot 
sctt = ax.scatter3D(df['humidity'], df['visibility'], df['cum_power'], 
                    alpha = 0.8, 
                    c = df['cum_power']) 
  
plt.title('3D plot of Humidity vs Visibility with colour legend of Cum_Power') 
ax.set_xlabel('weather', fontweight ='bold')  
ax.set_ylabel('month', fontweight ='bold')  
ax.set_zlabel('visibility', fontweight ='bold') 
fig.colorbar(sctt, ax = ax, shrink = 0.5, aspect = 5) 
  
# show plot 
plt.show() 
y = df[['cum_power']]
xt = df[['year','month','day','temp','weather','wind','humidity','barometer','visibility']] 

scaler = StandardScaler()
x = scaler.fit_transform(xt)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.30, random_state=42)
def rmse_test(model, X=xtrain):
    rmse = np.sqrt(-cross_val_score(model, xtrain, ytrain, scoring="neg_mean_squared_error", cv=5))
    return (rmse)
lasso = LassoCV(random_state=42, cv=5)
print("RMSE score for Lasso:")
print(rmse_test(lasso).mean())
lasso_model = lasso.fit(xtrain, ytrain)
ypred = lasso_model.predict(xtest)
np.sqrt(mean_squared_error(ytest, ypred))
elastic_net = ElasticNetCV(random_state=42, cv=5)
print("RMSE score for Elastic Net:")
print(rmse_test(elastic_net).mean())
elastic_model = elastic_net.fit(xtrain, ytrain)
ypred = elastic_model.predict(xtest)
np.sqrt(mean_squared_error(ytest, ypred))
rf = RandomForestRegressor(random_state=42)
print("RMSE score for Random Forest:")
print(rmse_test(rf).mean())
rf_model = rf.fit(xtrain,ytrain)
ypred = rf_model.predict(xtest)
np.sqrt(mean_squared_error(ytest, ypred))
xgboost = XGBRegressor(random_state=42)
print("RMSE score for XGBoost:")
print(rmse_test(xgboost).mean())
xgb_model = xgboost.fit(xtrain, ytrain)
ypred = xgb_model.predict(xtest)
np.sqrt(mean_squared_error(ytest, ypred))
xgboost = XGBRegressor(learning_rate=0.1,n_estimators=200,random_state=42)
print("RMSE score for XGBoost:")
print(rmse_test(xgboost).mean())
xgb_model = xgboost.fit(xtrain, ytrain)
ypred = xgb_model.predict(xtest)
np.sqrt(mean_squared_error(ytest, ypred))
xgb_model.feature_importances_
features = np.reshape(xgb_model.feature_importances_,(1,9))
featuredf = pd.DataFrame(features,columns=xt.columns)
# FEATURE COEEFICIENTS
featuredf.head()
featuredf.plot(kind='bar',title='Feature Coefficients',figsize=(15,6))
plt.show()
from pycaret.regression import *
setup = setup(df, target = 'cum_power', session_id = 123, normalize = True,
              numeric_features = ['year','month','day','temp','weather','wind','humidity','barometer','visibility'],
              polynomial_features = True, trigonometry_features = True, feature_interaction=True,
              bin_numeric_features = ['weather'])
setup[0].columns
regressor = create_model('gbr')
plot_model(regressor)
top3 = compare_models(n_select = 3)
