import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
%matplotlib inline
'''Display markdown formatted output like bold, italic bold etc.'''
from IPython.display import Markdown
def bold(string):
    display(Markdown(string))
dc_dat = pd.read_csv("../input/dc-properties/DC_Properties_trimmed.csv")
bold('**Preview of Data:**')
display(dc_dat.head(3))
bold('**Data types of our variables:**')
display(dc_dat.dtypes.value_counts())

dc_dat = dc_dat[dc_dat.GRADE != 'No Data']
for col in dc_dat.columns:
    if len(dc_dat[col].unique()) == 1:
        dc_dat.drop(col,inplace=True,axis=1)
# making a new  df with reduced number of variables after removing redundant variables
dc_dat1 = dc_dat.drop(['X','Y','QUADRANT', 'ASSESSMENT_NBHD', 'CENSUS_TRACT', 'CENSUS_BLOCK','ASSESSMENT_SUBNBHD', 'ZIPCODE', 'LATITUDE', 'LONGITUDE', 'FULLADDRESS', 'NATIONALGRID'],axis =1)

dc_dat1 = dc_dat1.drop(['SALEDATE','AYB'],axis = 1)
dc_dat1.dtypes.value_counts()
num_dat = dc_dat1.select_dtypes(include = ['int64', 'float64'])
bold('**Numerical variables:**')
display(num_dat.head(3))
#dc_dat1.hist('EYB',range=[1930, 2018])
sns.scatterplot(x = 'EYB',y = 'PRICE',data = dc_dat1)

category = pd.cut(dc_dat1.EYB,bins=[0,1950,1970,1990,2005,2018],labels=['Before1950','1951-1970','1971-1990','1991-2005','2006-2018'])
dc_dat1.insert(8,'EYB_group',category)
#dc_dat1.hist('YR_RMDL',range=[1935, 2020])
sns.scatterplot(x = 'YR_RMDL',y = 'PRICE',data = dc_dat1)
plt.xlim(1930, 2018)
category1 = pd.cut(dc_dat1.YR_RMDL,bins=[0,1950,1970,2000,2018],labels=['Before1950','1951-1970','1971-2000','2001-2018'])
dc_dat1.insert(7,'YR_RMDL_group',category1)
dc_dat1.loc[:,['YR_RMDL_group', 'EYB_group', 'BLDG_NUM',  'USECODE']] = dc_dat1.loc[:,['YR_RMDL_group', 'EYB_group', 'BLDG_NUM', 'USECODE']].astype('object')
# Dropping old yr_rmdl and EYB variables 
dc_dat1 = dc_dat1.drop(['EYB','YR_RMDL'], axis = 1)
dc_dat1.info()
cat_dat = dc_dat1.select_dtypes(include = ['object'])
bold('**Categorical variables:**')
display(cat_dat.head(3))
 #'YR_RMDL_group', 'EYB_group', 'QUALIFIED', 'GRADE', 'CNDTN'
    
    

plt.figure(figsize=(20,5))
sns.boxplot(
    data=dc_dat1,
    x='USECODE',
    y='PRICE',
    color='red')
dc_dat1.YR_RMDL_group.replace(to_replace = [ 'Before1950', '1951-1970','1971-2000', '2001-2018'], value = [0, 1, 2, 3], inplace = True)
dc_dat1.EYB_group.replace(to_replace = [ 'Before1950', '1951-1970','1971-1990', '1991-2005', '2006-2018'], value = [0, 1, 2, 3,4], inplace = True)
dc_dat1.QUALIFIED.replace(to_replace = ['U','Q'], value = [0, 1], inplace = True)
dc_dat1.GRADE.replace(to_replace = [ 'Fair Quality','Average','Above Average','Good Quality','Very Good', 'Excellent','Superior' ,'Exceptional-A', 'Exceptional-B', 'Exceptional-D','Exceptional-C'], value = [0, 1, 2, 3,4,5,6,7,8,9,10], inplace = True)
dc_dat1.CNDTN.replace(to_replace = ['Fair', 'Poor','Average','Good', 'Very Good', 'Excellent' ], value = [0, 1,2,3,4,5], inplace = True)


cat_dat = dc_dat1.select_dtypes(include = ['object'])
bold('**Categorical variables:**')
display(cat_dat.head(3))
cat_dat_one_hot = pd.get_dummies(cat_dat)
cat_dat_one_hot.head()
dc_dat1 = dc_dat1.drop(['HEAT', 'AC', 'BLDG_NUM', 'STYLE', 'STRUCT', 'EXTWALL', 'ROOF','INTWALL', 'USECODE', 'WARD'], axis = 1)

dc_dat1 = pd.concat([dc_dat1, cat_dat_one_hot], axis = 1)
dc_dat1.head()
from sklearn.model_selection import train_test_split
dc_train,dc_test = train_test_split(dc_dat1 , test_size = 0.25, random_state = 101)
x_train = dc_train.drop(['PRICE'], axis = 1)
x_test = dc_test.drop(['PRICE'], axis = 1)
y_train = dc_train['PRICE']
y_test = dc_test['PRICE']
bold('**Preview of Train Data:**')
display(dc_train.head(3))
bold('**Preview of Test Data:**')
display(dc_test.head(3))
'''Dimensions of train and test data'''
bold('**Shape of our train and test data:**')
display(dc_train.shape, dc_test.shape)
plt.hist(y_train,bins = 100)
plt.show()
y_train = np.log1p(y_train)
plt.hist(y_train,bins = 100)
plt.show()
y_test = np.log1p(y_test)
plt.hist(y_test,bins = 100)
plt.show()
'''Standarize numeric features with RobustScaler'''
from sklearn.preprocessing import RobustScaler

'''Initialize robust scaler object.'''
robust_scl = RobustScaler()


cols_to_norm = ['BATHRM', 'HF_BATHRM', 'NUM_UNITS', 'ROOMS', 'BEDRM','STORIES', 'SALE_NUM', 'GBA', 'KITCHENS','FIREPLACES', 'LANDAREA', 'SQUARE']

x_train[cols_to_norm] = robust_scl.fit_transform(x_train[cols_to_norm])
x_test[cols_to_norm] = robust_scl.fit_transform(x_test[cols_to_norm])
bold('**Preview of Train Data:**')
display(x_train.head(3))
display(y_train.head(3))

bold('**Preview of Test Data:**')
display(x_test.head(3))
display(y_test.head(3))
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
'''We are interested in the following 14 regression models.
All initialized with default parameters except random_state and n_jobs.'''

seed = 103 

linear = LinearRegression(n_jobs = -1)
lasso = Lasso(random_state = seed)
ridge = Ridge(random_state = seed)
kr = KernelRidge()
elnt = ElasticNet(random_state = seed)
dt = DecisionTreeRegressor(random_state = seed)
svm = SVR()
knn = KNeighborsRegressor(n_jobs = -1)
rf =  RandomForestRegressor(n_jobs = -1, random_state = seed)
et = ExtraTreesRegressor(n_jobs = -1, random_state = seed)
ab = AdaBoostRegressor(random_state = seed)
gb = GradientBoostingRegressor(random_state = seed)
xgb = XGBRegressor(random_state = seed, n_jobs = -1)
lgb = LGBMRegressor(random_state = seed, n_jobs = -1)
'''Evaluate models on the test dataset.'''
def model_score(model):
    from sklearn.metrics import mean_squared_error
    X_train = x_train
    X_test = x_test
    Y_train = y_train
    Y_test = y_test
    model.fit(X_train, Y_train)
    prediction = model.predict(X_test)
    mse = mean_squared_error(prediction, Y_test)
    rmse = np.sqrt(mse)
    Y_test_e = np.expm1( Y_test )
    prediction_e = np.expm1(prediction)
    mse_e = mean_squared_error(prediction_e, Y_test_e)
    rmse_e = np.sqrt(mse_e) 
    return rmse, rmse_e

'''Calculate train_test_split score of differnt models and plot them.'''
models = [lasso, ridge, kr, elnt, dt, svm, knn, rf, et, ab, gb, xgb, lgb]
model_rmse = []
for model in models:
    model_rmse.append(model_score(model))

'''Function to plot scatter plot'''
import plotly.offline as py
from plotly.offline import iplot, init_notebook_mode
import plotly.graph_objs as go
init_notebook_mode(connected = True) # Required to use plotly offline in jupyter notebook

def scatter_plot(x, y, title, xaxis, yaxis, size, c_scale):
    trace = go.Scatter(
    x = x,
    y = y,
    mode = 'markers',
    marker = dict(color = y, size = size, showscale = True, colorscale = c_scale))
    layout = go.Layout(hovermode= 'closest', title = title, xaxis = dict(title = xaxis), yaxis = dict(title = yaxis))
    fig = go.Figure(data = [trace], layout = layout)
    return iplot(fig) 
'''Plot data frame of train test rmse'''
train_test_score = pd.DataFrame(data = model_rmse, columns = ['RMSE','RMSE_target_unit'])
train_test_score.index = ['LSO', 'RIDGE', 'KR', 'ELNT', 'DT', 'SVM', 'KNN', 'RF', 'ET', 'AB', 'GB', 'XGB', 'LGB']
train_test_score = train_test_score.round(5)
x = train_test_score.index
y = train_test_score['RMSE']
z = train_test_score['RMSE_target_unit']
title = "Models' Test Score (RMSE)"
scatter_plot(x, y, title, 'Models','RMSE', 30, 'RdBu')
title1 = "Models' Test Score (RMSE_target_unit)"
scatter_plot(x, z, title1, 'Models','RMSE', 30, 'RdBu')
model_rmse
lm = LinearRegression()
lm.fit(x_train, y_train)
predictions = lm.predict(x_test)
plt.scatter(y_test, predictions)
sns.distplot((y_test-predictions), bins = 50)
y_test_e = np.expm1( y_test )
predictions_e = np.expm1(predictions)

from sklearn import metrics
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test_e, predictions_e)))