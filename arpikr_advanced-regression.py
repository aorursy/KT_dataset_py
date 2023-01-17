pwd
import warnings
warnings.filterwarnings('ignore')
import os
import pandas as pd
from pandas import DataFrame
import pylab as pl
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
%matplotlib inline
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.colors import n_colors
from plotly.subplots import make_subplots
kaggle=pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
#kaggle_test=pd.read_csv("C:\\Users\\ARPIT\\Desktop\\New folder\\Kaggle Housing\\test_housing.csv")
kaggle.head()
kaggle.info()
numeric_data = kaggle.select_dtypes(include=[np.number])
categorical_data = kaggle.select_dtypes(exclude=[np.number])
print("Numeric_Column_Count =", numeric_data.shape)
print("Categorical_Column_Count =", categorical_data.shape)
kag=categorical_data.nunique()
fig = go.Figure(go.Bar(x=categorical_data,y=kag,))
fig.update_layout(title_text='Unique values in Categorical Columns',xaxis_title="Category",yaxis_title="Count of Unique Values")
fig.show()
for k, v in numeric_data.items():
    q1 = v.quantile(0.25)
    q3 = v.quantile(0.75)
    irq = q3 - q1
    v_col = v[(v <= q1 - 1.5 * irq) | (v >= q3 + 1.5 * irq)]
    perc = np.shape(v_col)[0] * 100.0 / np.shape(numeric_data)[0]
    print("Column %s outliers = %.2f%%" % (k, perc))
plt.figure(figsize=(15,10))
kaggle.boxplot(patch_artist=True,vert=False)
import missingno as msno
msno.bar(kaggle)
total = kaggle.isnull().sum().sort_values(ascending=False)
percent_1 = kaggle.isnull().sum()/kaggle.isnull().count()*100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, percent_2], axis=1, keys=['Total','%'])
def msv1(data, thresh=20, color='black', edgecolor='black', width=15, height=5):
    plt.figure(figsize=(width,height))
    percentage=(kaggle.isnull().mean())*100
    percentage.sort_values(ascending=False).plot.bar(color=color, edgecolor=edgecolor)
    plt.axhline(y=thresh, color='r', linestyle='-')
    plt.title('Missing values percentage per column', fontsize=20, weight='bold' )
    plt.text(len(data.isnull().sum()/len(data))/1.7, thresh+12.5, f'Columns with more than {thresh}% missing values', fontsize=12, color='crimson',
         ha='left' ,va='top')
    plt.text(len(data.isnull().sum()/len(data))/1.7, thresh - 5, f'Columns with less than {thresh}% missing values', fontsize=12, color='green',
         ha='left' ,va='top')
    plt.xlabel('Columns', size=15, weight='bold')
    plt.ylabel('Missing values percentage')
    plt.yticks(weight ='bold')
    
    return plt.show()
msv1(kaggle, 20, color=('black', 'deeppink'))
### 
total_missing=kaggle.isnull().sum().sort_values(ascending=False)
percent=(kaggle.isnull().sum()/kaggle.isnull().count()).sort_values(ascending=False)
missing_data=pd.concat([total_missing,percent],axis=1,keys=['Missing_Total','Percent'])
missing_data.head(5)
allna = (kaggle.isnull().sum() / len(kaggle))*100
allna = allna.drop(allna[allna == 0].index).sort_values()
NA=kaggle[allna.index.to_list()]
NAcat=NA.select_dtypes(include='object')
NAnum=NA.select_dtypes(exclude='object')
print(f'We have :{NAcat.shape[1]} categorical features with missing values')
print(f'We have :{NAnum.shape[1]} numerical features with missing values')
#all_data_na = (kaggle.isnull().sum() / len(kaggle)) * 100
#all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
#missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
#missing_data.head(20)
my_corr=kaggle.corr()
plt.figure(figsize=(20,20))
sns.set(font_scale=0.8)
sns.heatmap(my_corr, cbar=False, annot=True, square=True, fmt='.2f', annot_kws={'size': 10},linewidth=0.8)
plt.show()
num=kaggle.select_dtypes(exclude='object')
numcorr=num.corr()
f,ax=plt.subplots(figsize=(20,1))
sns.heatmap(numcorr.sort_values(by=['SalePrice'], ascending=False).head(1), cmap='Blues')
plt.title(" Numerical features correlation with the sale price", weight='bold', fontsize=18)
plt.xticks(weight='bold')
plt.yticks(weight='bold', color='dodgerblue', rotation=0)
plt.show()
cor_target = abs(my_corr["SalePrice"])
#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.5]
relevant_features
cor_target =kaggle.corr().abs()
Target_Corr = cor_target.corr()['SalePrice'].to_frame().reset_index() #Feature Correlation related to SalePrice
Feature_corr =cor_target.unstack().to_frame(name='Correlation') # Feature Relation
Feature = Feature_corr[(Feature_corr['Correlation']>=0.8)&(Feature_corr['Correlation']<1)].sort_values(by='Correlation', ascending = False).reset_index()
Feature.head(10)

kaggle.groupby('Neighborhood', as_index=True)['SalePrice'].mean()
plt.figure(figsize=(20,12))
plt.yticks(weight='bold')
kaggle['Neighborhood'].value_counts().plot(kind="barh",color='coral', fontsize=16)
plt.title('Most frequent neighborhoods',fontsize=22, loc='center')
import plotly.graph_objects as go
from plotly.subplots import make_subplots

labels = kaggle['BldgType']
fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
values = kaggle['LotFrontage']
values1= kaggle['MSSubClass']
fig.add_trace(go.Pie(labels=labels, values=values, name="Lot Frontage"),1, 1)
fig.add_trace(go.Pie(labels=labels, values=values1, name="MSSubClass"),1, 2)
fig.update_traces(hole=.5, hoverinfo="label+percent+name")
fig.update_layout(title_text="Class& Frontage dependency on Building Type")
annotations=([dict(text='LotFrontage', x=0.18, y=0.3, font_size=20, showarrow=False),
                 dict(text='MSSubClass', x=0.82, y=0.3, font_size=20, showarrow=False)])
fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
fig.show()
#fig = go.Figure(data=[go.Pie(labels=labels, values=values,hole=.5)])

import plotly.graph_objects as go
labels = kaggle['SaleType']
values = kaggle['GrLivArea']
fig = go.Figure(data=[go.Pie(labels=labels, values=values,hole=0.0)])
fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
fig.show()
import plotly.graph_objects as go
labels = kaggle['Foundation']
values = kaggle['SalePrice']
fig = go.Figure(data=[go.Pie(labels=labels, values=values,hole=0.0)])
fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
fig.show()
import plotly.graph_objects as go
labels = kaggle['MoSold']
values = kaggle['SalePrice']
fig = go.Figure(data=[go.Pie(labels=labels, values=values,hole=0.5)])
fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
fig.show()
pd.crosstab([kaggle.Heating,kaggle.Electrical],kaggle.KitchenAbvGr,margins=True).style.background_gradient(cmap='Wistia')
pearson_coef, p_value = stats.pearsonr(kaggle['LotArea'], kaggle['SalePrice'])
print("The Pearson Correlation Coefficient of LotArea is", pearson_coef, " with a P-value of P =", p_value)  
sns.regplot(x="LotArea", y="SalePrice", data=kaggle)
plt.ylim(0,)
pearson_coef, p_value = stats.pearsonr(kaggle['GarageArea'], kaggle['SalePrice'])
print("The Pearson Correlation Coefficient of GarageArea is", pearson_coef, " with a P-value of P =", p_value)  
sns.regplot(x="GarageArea", y="SalePrice", data=kaggle)
plt.ylim(0,)
pearson_coef, p_value = stats.pearsonr(kaggle['GrLivArea'], kaggle['SalePrice'])
print("The Pearson Correlation Coefficient of GrLivArea is", pearson_coef, " with a P-value of P =", p_value)  
sns.regplot(x="GrLivArea", y="SalePrice", data=kaggle)
plt.ylim(0,)
import matplotlib.gridspec as gridspec
def multi_plotting (kaggle, feature): 

    fig = plt.figure(constrained_layout=True, figsize=(12,8))
    grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)

    ax1 = fig.add_subplot(grid[0, :2])
    ax1.set_title('Histogram')
    sns.distplot(kaggle.loc[:,feature], norm_hist=True, ax = ax1)

    ax2 = fig.add_subplot(grid[1, :2])
    ax2.set_title('QQ_plot')
    stats.probplot(kaggle.loc[:,feature], plot = ax2)

    ax3 = fig.add_subplot(grid[:, 2])
    ax3.set_title('Box Plot')
    sns.boxplot(kaggle.loc[:,feature], orient='v', ax = ax3 );

    print("Skewness: "+ str(kaggle['SalePrice'].skew().round(3))) 
    print("Kurtosis: " + str(kaggle['SalePrice'].kurt().round(3)))
multi_plotting (kaggle,'SalePrice')
kaggle["SalePrice"] = np.log1p(kaggle["SalePrice"])
multi_plotting (kaggle,'SalePrice')
from scipy import stats
import numpy as np
z = np.abs(stats.zscore(numeric_data))
print(z)
threshold = 3
#print(np.where(z > 3))
kaggle['SalePrice'].quantile([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
import matplotlib.gridspec as gridspec
def multi_plotting (kaggle, feature): 

    fig = plt.figure(constrained_layout=True, figsize=(12,8))
    grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)

    ax1 = fig.add_subplot(grid[0, :2])
    ax1.set_title('Histogram')
    sns.distplot(kaggle.loc[:,feature], norm_hist=True, ax = ax1)

    ax2 = fig.add_subplot(grid[1, :2])
    ax2.set_title('QQ_plot')
    stats.probplot(kaggle.loc[:,feature], plot = ax2)

    ax3 = fig.add_subplot(grid[:, 2])
    ax3.set_title('Box Plot')
    sns.boxplot(kaggle.loc[:,feature], orient='v', ax = ax3 );

    print("Skewness: "+ str(kaggle['LotArea'].skew().round(3))) 
    print("Kurtosis: " + str(kaggle['LotArea'].kurt().round(3)))
multi_plotting (kaggle,'LotArea')
from yellowbrick.regressor import CooksDistance
# Load the regression dataset
X = kaggle["LotArea"].values.reshape(-1,1)
Y = kaggle["SalePrice"]
# Instantiate and fit the visualizer
visualizer = CooksDistance()
visualizer.fit(X, Y)
visualizer.show()
#kaggle['LotArea'].quantile([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
q = kaggle['LotArea'].quantile(0.97)
q
kaggle[kaggle.LotArea > q]
avg =kaggle['LotArea'].mean()
r=kaggle.loc[(kaggle["LotArea"]>q), 'LotArea'] = avg # Replacing the Row with Base Price less than q with Average Price
#r 
#sns.boxplot( y=kaggle["LotArea"])
fig = go.Figure(go.Box(y=kaggle["LotArea"],name="Lot Area")) # to get Horizonal plot change axis :  x=germany_score
fig.update_layout(title="Distribution of Datapoints")
fig.show()
kaggle=kaggle.drop(['PoolQC','MiscFeature','Alley','Fence','LotFrontage'],axis=1,inplace=False)
kaggle=kaggle.drop(['GarageCars','GarageYrBlt','TotRmsAbvGrd','YearRemodAdd'],axis=1,inplace=False) #Not of USe
kaggle.shape
kaggle['House_Age']=kaggle['YrSold']-kaggle['YearBuilt']
kaggle=kaggle.drop(['YrSold','YearBuilt'],axis=1,inplace=False)
kaggle.shape
kaggle['MasVnrArea']=kaggle.MasVnrArea.fillna(0)
kaggle.head()
kaggle['TotalLivingSF'] = kaggle['GrLivArea'] + kaggle['TotalBsmtSF'] - kaggle['LowQualFinSF']
kaggle=kaggle.drop(['BsmtFinSF1','BsmtFinSF2','GrLivArea','TotalBsmtSF','LowQualFinSF','FireplaceQu'],axis=1,inplace=False)
kaggle.shape
kaggle=kaggle.drop(['1stFlrSF','2ndFlrSF'],axis=1,inplace=False)
kaggle.shape
from yellowbrick.regressor import CooksDistance
# Load the regression dataset
X = kaggle["TotalLivingSF"].values.reshape(-1,1)
Y = kaggle["SalePrice"]
# Instantiate and fit the visualizer
visualizer = CooksDistance()
visualizer.fit(X, Y)
visualizer.show()
fig = go.Figure(data=go.Violin(y=kaggle["TotalLivingSF"], box_visible=True, line_color='black',
                               meanline_visible=True, fillcolor='lightseagreen', opacity=0.8,
                               x0='Total Living Area'))
fig.update_layout(yaxis_zeroline=False,title="Distribution of Derived Feature-Total Living Area ")
fig.show()
q = kaggle['TotalLivingSF'].quantile(0.98)
q
kaggle[kaggle.TotalLivingSF > q]
avg =kaggle['TotalLivingSF'].mean()
r=kaggle.loc[(kaggle["TotalLivingSF"]>q), 'TotalLivingSF'] = avg # Replacing the Row with Base Price less than q with Average Price
r 
fig = go.Figure(go.Box(y=kaggle["TotalLivingSF"],name="Lot Area")) # to get Horizonal plot change axis :  x=germany_score
fig.update_layout(title="Distribution of Datapoints")
fig.show()
kaggle['MSZoning'] = kaggle.groupby(['Neighborhood', 'MSSubClass'])['MSZoning'].apply(lambda x: x.fillna(x.value_counts().index[0]))
kaggle['Utilities'] = kaggle.groupby(['Neighborhood', 'MSSubClass'])['Utilities'].apply(lambda x: x.fillna(x.value_counts().index[0]))
kaggle['Exterior1st'] = kaggle.groupby(['Neighborhood', 'MSSubClass'])['Exterior1st'].apply(lambda x: x.fillna(x.value_counts().index[0]))
kaggle['Exterior2nd'] = kaggle.groupby(['Neighborhood', 'MSSubClass'])['Exterior2nd'].apply(lambda x: x.fillna(x.value_counts().index[0]))
kaggle['MasVnrType'] = kaggle.groupby(['Neighborhood', 'MSSubClass'])['MasVnrType'].apply(lambda x: x.fillna(x.value_counts().index[0]))
for col in ['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','GarageType','GarageFinish','GarageQual','GarageCond']:
    kaggle[col] = kaggle[col].fillna('None')
kaggle['Electrical'] = kaggle.groupby(['Neighborhood','MSSubClass' ])['Electrical'].apply(lambda x: x.fillna(x.value_counts().index[0]))
kaggle['KitchenQual'] = kaggle.groupby(['Neighborhood','MSSubClass' ])['KitchenQual'].apply(lambda x: x.fillna(x.value_counts().index[0]))
kaggle['Functional'] = kaggle.groupby(['Neighborhood', 'MSSubClass'])['Functional'].apply(lambda x: x.fillna(x.value_counts().index[0]))
kaggle['SaleType'] = kaggle.groupby(['Neighborhood', 'MSSubClass'])['SaleType'].apply(lambda x: x.fillna(x.value_counts().index[0]))
numeric_data = kaggle.select_dtypes(include=[np.number])
categorical_data = kaggle.select_dtypes(exclude=[np.number])
print("Numeric_Column_Count =", numeric_data.shape)
print("Categorical_Column_Count =", categorical_data.shape)
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
categorical_data=categorical_data.apply(LabelEncoder().fit_transform)
#X=X.apply(LabelEncoder().fit_transform)
categorical_data.head()
categorical_data=categorical_data.astype(int)
#categorical_data.dtypes
Kag_new = pd.concat([numeric_data, categorical_data],axis=1)
Kag_new.info()
Kag_new=Kag_new.drop(['Id'],axis=1,inplace=False)
X=Kag_new.drop(['SalePrice'],axis=1,inplace=False)
Y=Kag_new['SalePrice']
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
X_vif = add_constant(X)
vif = pd.Series([variance_inflation_factor(X_vif.values, i) 
               for i in range(X_vif.shape[1])], 
              index=X_vif.columns)
print(vif.sort_values(ascending = False).head(20))
from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X)
X
import sklearn.linear_model as linear_model
import sklearn.model_selection as model_selection
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=123)
from sklearn.linear_model import LinearRegression
my_model = LinearRegression(normalize=True)  #Create an object of LinearRegression class.
my_model.fit(X_train, Y_train)               #Fitting the linear regression model to our training set.
predictions = my_model.predict(X_test)       #Make predictions on the test set
pd.DataFrame({'actual value': Y_test, 'predictions':predictions}).sample(5)   #Compare a sample of 5 actual Y values from test set and corresponding predicted values 
linear=my_model.score(X_test, Y_test)           #Check the  R2  value
linear
my_model.coef_
my_model.intercept_
from sklearn import metrics
print('MAE',metrics.mean_absolute_error(Y_test,predictions))
print('MSE',metrics.mean_squared_error(Y_test,predictions))
print('RMSE',np.sqrt(metrics.mean_squared_error(Y_test,predictions)))
actuals = Y_test.values
## Actual vs Predicted plot
plt.plot(actuals,"b")
plt.plot(predictions,"g")
plt.title('Actual vs Predicted')
plt.show()

# Visualzing actual vs predicted 
fig, ax = plt.subplots()
ax.scatter(Y_test, predictions, edgecolors=(0, 0, 0))
ax.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=4)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title("Actual vs Predicted")
plt.show()
from yellowbrick.regressor import ResidualsPlot
visualizer = ResidualsPlot(my_model,hist=False)
visualizer.fit(X_train, Y_train)  # Fit the training data to the visualizer
visualizer.score(X_test,Y_test)  # Evaluate the model on the test data
visualizer.show() 
regR=linear_model.Ridge(normalize=True)
regR=regR.fit(X_train,Y_train)
GregR=model_selection.GridSearchCV(regR,param_grid={'alpha':np.arange(1,1000,100)})
GregR=GregR.fit(X_train,Y_train)
GregR.best_params_
metrics.r2_score(Y_test,GregR.predict(X_test))
print(metrics.mean_squared_error(Y_test,GregR.predict(X_test)))
reg=linear_model.Lasso(max_iter=1000,normalize=True)
reg=reg.fit(X_train,Y_train)
Greg=model_selection.GridSearchCV(reg,param_grid={'alpha':np.arange(0.1,100,1).tolist()})
Greg=Greg.fit(X_train,Y_train)
Greg.best_params_
Greg.cv_results_
print(metrics.mean_squared_error(Y_test,GregR.predict(X_test)))
nEstimator = [140,160,180,200,220,240,260,280,300,320,340,360]
depth = [10,15,20,25,30,35,40,45,50,55,60]
RF = RandomForestRegressor()
hyperParam = [{'n_estimators':nEstimator,'max_depth': depth}]
gsv = GridSearchCV(RF,hyperParam,cv=5,verbose=1,scoring='r2',n_jobs=-1)
gsv.fit(X_train, Y_train)
print("Best HyperParameter: ",gsv.best_params_)
print(gsv.best_score_)
scores = gsv.cv_results_['mean_test_score'].reshape(len(nEstimator),len(depth))
plt.figure(figsize=(8, 8))
plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot)
plt.xlabel('n_estimators')
plt.ylabel('max_depth')
plt.colorbar()
plt.xticks(np.arange(len(nEstimator)), nEstimator)
plt.yticks(np.arange(len(depth)), depth)
plt.title('Grid Search r^2 Score')
plt.show()
maxDepth=gsv.best_params_['max_depth']
nEstimators=gsv.best_params_['n_estimators']
model = RandomForestRegressor(n_estimators = nEstimators,max_depth=maxDepth)
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)
print('Root means score', np.sqrt(mean_squared_error(Y_test, y_pred)))
print('Variance score: %.2f' % r2_score(Y_test, y_pred))
print("Result :",model.score(X_test, Y_test))
RF_HyperScore = r2_score(Y_test, y_pred)
RF_HyperScore
gb_model = GradientBoostingRegressor(max_depth=30, n_estimators=250, learning_rate=0.01, random_state=123)
gb_model.fit(X_train, Y_train)
predictions = gb_model.predict(X_test)
Ensemble_GB_Score = r2_score(Y_test, predictions)
Ensemble_GB_Score
print('RMSE',np.sqrt(metrics.mean_squared_error(Y_test,predictions)))
# Visualzing actual vs predicted 
fig, ax = plt.subplots()
ax.scatter(Y_test, predictions, edgecolors=(0, 0, 0))
ax.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=4)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title("Actual vs Predicted")
plt.show()
gb_model.feature_importances_
ada_model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=8), n_estimators=200, learning_rate=0.01, random_state=123)
ada_model.fit(X_train, Y_train)
predictions = ada_model.predict(X_test)
AdaBoostScore=r2_score(Y_test, predictions)
AdaBoostScore
print('RMSE',np.sqrt(metrics.mean_squared_error(Y_test,predictions)))
tree = DecisionTreeRegressor(max_depth=10)
tree.fit(X_train, Y_train)
predictions = tree.predict(X_test)
actuals = Y_test.values
## Actual vs Predicted plot
plt.plot(actuals,"b")
plt.plot(predictions,"g")
plt.title('Actual vs Predicted')
plt.show()

# Visualzing actual vs predicted 
fig, ax = plt.subplots()
ax.scatter(Y_test, predictions, edgecolors=(0, 0, 0))
ax.plot([Y_test.min(), Y_test.max()], [Y_test.min(),Y_test.max()], 'k--', lw=4)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title("Actual vs Predicted")
plt.show()
DecisonTreeRegressor_Score = r2_score(Y_test, predictions)
DecisonTreeRegressor_Score
print('RMSE',np.sqrt(metrics.mean_squared_error(Y_test,predictions)))
tree_score = tree.score(X_train,Y_train)
tree_score
param_grid = [{"max_depth":[3, 4, 5,6,7,8,9,10, None], "max_features":[5,6,7,8,9,10,11,12,13,14,15,16]}]
gs = GridSearchCV(estimator=DecisionTreeRegressor(random_state=123),\
                 param_grid = param_grid,\
                 cv=15)
gs.fit(X_train, Y_train)
gs.cv_results_['params']
gs.best_params_
gs.cv_results_['rank_test_score']
gs.best_estimator_
predictions2 = gs.predict(X_test)
DTGridSearchCGScore = r2_score(Y_test, predictions2)
DTGridSearchCGScore
np.any(np.isnan(Test))
Test.fillna(0,inplace=True)
np.all(np.isfinite(Test))
