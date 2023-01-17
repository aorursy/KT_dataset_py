# Import Libraries
import pandas as pd
import numpy as np
from scipy import stats
import urllib
import os
import time

import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
%matplotlib inline
# Fetch data from book's github repository and dump into disk, and load it into dataframe
data_url = 'https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv'
default_data_file_path = os.path.join('datasets','housing')
def fetch_housing_data(data_url=data_url,data_file_path=default_data_file_path):
    os.makedirs(data_file_path,exist_ok=True)
    csv_file_path = os.path.join(data_file_path,'housing.csv')
    urllib.request.urlretrieve(data_url,csv_file_path)
    df = pd.read_csv(csv_file_path)
    return df
# Load Data And view the sample
run_dir = os.path.join(default_data_file_path,time.strftime('run_%Y_%m_%d-%H_%M_%S'))
df = fetch_housing_data(data_file_path=run_dir)
print(f'Shape of is {df.shape[0]} rows and {df.shape[1]} columns')
df.sample(5)
# Check the data count and data type for all the available columns
df.info()
# Check  the distribution of categorical variable
df['ocean_proximity'].value_counts()
# Check the distribution of continues variables
df.drop(['ocean_proximity'],axis=1).describe()
# Plot the histogram of all the continues variables
hist_plot = df.drop(['ocean_proximity'],axis=1).hist(bins=50,figsize=(25,21))
plt.savefig(os.path.join(run_dir,'Initial_Histograms.jpg'))
# Bin the Median Income variable to so that we can split  data without creating sampling bias
df['median_income_cut'] = pd.cut(df['median_income'],bins=[0,1.5,3,4.5,6,np.inf],labels=[1,2,3,4,5])
hist_plot = df['median_income_cut'].hist()
df['median_income_cut'].value_counts()/len(df) * 100
# Stratified Sampling is done, so that both train & test samples will have similar distribution of important variables
split = StratifiedShuffleSplit(n_splits=10,test_size=0.2,random_state=42)
train = test = 0
for train_loc,test_loc in split.split(df,df['median_income_cut']):
    train = df.loc[train_loc]
    test = df.loc[test_loc]
# Check if the distribution is similar to full dataset after spliting
cut_check = pd.DataFrame([df['median_income_cut'].value_counts()/len(df),train['median_income_cut'].value_counts()/len(train),
                          test['median_income_cut'].value_counts()/len(test)]).T
cut_check.columns = ['Overall','Training Set','Test Set']
print('As we can see, using Stratified, sampling we could make sure that the distribution is equal in both sets')
cut_check.sort_index()
# Drop the binned column we created and dump the data from train & test set into files
train = train.drop(['median_income_cut'],axis=1)
test = test.drop(['median_income_cut'],axis=1)
train.to_csv(os.path.join(run_dir,'train.csv'),index=False)
test.to_csv(os.path.join(run_dir,'test.csv'),index=False)
# Create a copy training data set for further analysis, this gives us freedom to play around with data set
df = train.copy()
# Plot the Latitude & Longitude to visualize the distribution of population and median house value
sct_plot = df.plot(kind='scatter',x='longitude',y='latitude',alpha=0.4,s=df['population']/100,
                   label='popilation',c='median_house_value',cmap=plt.get_cmap('jet'),figsize=(12,8),colorbar=True)
sct_plot = plt.legend()
plt.savefig(os.path.join(run_dir,'GeoPlot_Expo.jpg'))
# Calculate the correlation of numerical variables with target variable 'median house price'
scat_mat = df.drop(['ocean_proximity'],axis=1).corr()
scat_mat['median_house_value'].sort_values(ascending=False)
# generate random data
x,y = np.zeros((3,100)),np.zeros((3,100))
x[0,:] = np.array([np.random.randint(0,100) for _ in range(100)])
y[0,:] = 3.5 * x[0,:] + np.random.randn(100)*20
x[1,:] = np.array([np.random.randint(0,100) for _ in range(100)])
y[1,:] = -2 * x[1,:]**3  + np.random.randn(100)
x[2,:] = np.array([np.random.randint(0,100) for _ in range(100)])
y[2,:] = 2 * x[2,:]**3  + np.random.randn(100)
fig,ax = plt.subplots(figsize=(16,4),ncols=3,dpi=100)
for i in range(3):
    coff = np.round(np.corrcoef(x[i,:],y[i,:])[0,1],5)
    ax[i].scatter(x[i,:],y[i,:])
    ax[i].title.set_text(f'Correlation value: {coff}')
# Plot scater plot matrix to visualize the correlation of variabes with each other
scat_plot = pd.plotting.scatter_matrix(df[['median_house_value','median_income','total_rooms',
                                           'housing_median_age']],figsize=(12,10))
# Build 3 new variables as per our intution and business understanding
df['rooms_per_household'] = df['total_rooms']/df['households']
df['bedrooms_per_room'] = df['total_bedrooms']/df['total_rooms']
df['population_per_household'] = df['population']/df['households']
# Again the check the correlation to evaluate the effect of new features built
scat_mat = df.drop(['ocean_proximity'],axis=1).corr()
scat_mat['median_house_value'].sort_values(ascending=False)
# Build box plots of categorical feature to evalutae its effect on house values
box_plot = df.boxplot(column='median_house_value',by='ocean_proximity',figsize=(8,8))
plt.savefig(os.path.join(run_dir,"Ocean_Proximity_Box_Plot.jpg"))
# Seprate training data and labels
X_train = train.drop('median_house_value',axis=1)
y_train = train['median_house_value']
# Building transformer to create new features on dataset
class CombineAttributes(BaseEstimator,TransformerMixin):
    def __init__(self,ttl_room_idx=3,ttl_bed_idx=4,pop_idx=5,hh_idx=6):
        self.ttl_room_idx = ttl_room_idx
        self.ttl_bed_idx = ttl_bed_idx
        self.pop_idx = pop_idx
        self.hh_idx = hh_idx
    def fit(self,X, y=None):
        return self
    def transform(self,X,y=None):
        rooms_per_household = X[:,self.ttl_room_idx]/X[:,self.hh_idx]
        bedrooms_per_room = X[:,self.ttl_bed_idx]/X[:,self.ttl_room_idx]
        population_per_household = X[:,self.pop_idx]/X[:,self.hh_idx]
        return np.c_[X,rooms_per_household,bedrooms_per_room,population_per_household]
# Build the pipeline to perform all major transformations to fit on training data,
#this will help us apply consistent transformations to test datasets and any new dataset we get for scoring.
pipe_num = Pipeline([('imputer',SimpleImputer(strategy='median')),('attr_com',CombineAttributes()),
                     ('scaler',StandardScaler())])
num_attr = list(X_train.drop('ocean_proximity',axis=1).columns)
full_pipeline = ColumnTransformer([('num_pipe',pipe_num,num_attr),
                                   ('encoder',OneHotEncoder(),['ocean_proximity'])])
# Transform training dataset using the pipeine
X_train = full_pipeline.fit_transform(X_train)
# Build a function to train any given model, and give results, this will save a lot of time
def single_model_trainer(model,X,y,validation_set=True):
    if validation_set == True:
        train_X,test_X,train_y,test_y = train_test_split(X,y,random_state=42)
        model.fit(train_X,train_y)
        y_pred = model.predict(train_X)
        mse = mean_squared_error(train_y,y_pred)
        train_rmse = np.sqrt(mse)
        y_pred = model.predict(test_X)
        mse = mean_squared_error(test_y,y_pred)
        test_rmse = np.sqrt(mse)
        return (model,train_rmse,test_rmse)
    else:
        model.fit(X,y)
        y_pred = model.predict(X)
        mse = mean_squared_error(y,y_pred)
        rmse = np.sqrt(mse)
        return (model,rmse)
# Training Linear Regression Model
model,train_rmse,test_rmse = single_model_trainer(LinearRegression(),X_train,y_train)
print(f'Training Error : {train_rmse}')
print(f'Validation Error : {test_rmse}')
# Training Decision Tree Model
model,train_rmse,test_rmse = single_model_trainer(DecisionTreeRegressor(),X_train,y_train)
print(f'Training Error : {train_rmse}')
print(f'Validation Error : {test_rmse}')
# Training Random Forest Model
model,train_rmse,test_rmse = single_model_trainer(RandomForestRegressor(n_estimators=100),X_train,y_train)
print(f'Training Error : {train_rmse}')
print(f'Validation Error : {test_rmse}')
# Training Gradient Boosting Model
model,train_rmse,test_rmse = single_model_trainer(GradientBoostingRegressor(n_estimators=100),X_train,y_train)
print(f'Training Error : {train_rmse}')
print(f'Validation Error : {test_rmse}')
# training Support Vector Model
model,train_rmse,test_rmse = single_model_trainer(SVR(gamma='scale'),X_train,y_train)
print(f'Training Error : {train_rmse}')
print(f'Validation Error : {test_rmse}')
# Training Model using KFold Cross Validation approch
models = {'LinearRegression':LinearRegression(),'DecisionTreeRegressor':DecisionTreeRegressor(),
          'RandomForestRegressor':RandomForestRegressor(n_estimators=100),'GradientBoostingRegressor':
          GradientBoostingRegressor(n_estimators=100),'SVR':SVR(gamma='scale')}
results = {}
for name,model in models.items():
    cv = KFold(n_splits=10, shuffle=False, random_state=42)
    scores = cross_val_score(model,X_train,y_train,cv=cv,scoring='neg_mean_squared_error')
    print(f'Cross Validation Completed for {name}')
    rmse_scores = np.sqrt(-scores)
    results[name] = rmse_scores
results_df = pd.DataFrame.from_dict(results)
results_df
# Looking at the distribution of model results, we dont want a model which is very good at certain subset but bad at others.
results_df.describe()
# Train Random Forest Model using Grid Search to find best hyperparameters
forest_reg = RandomForestRegressor()
param_grid = {'bootstrap':[False],'n_estimators':(100,150),'max_depth':(10,15,None),'min_samples_split':(2,5),
          'max_features':(6,8,10,'auto'),'random_state':[42]}
grid_search_results = GridSearchCV(forest_reg,param_grid,scoring='neg_mean_squared_error',cv=5,return_train_score=True,
                                   n_jobs=-1,verbose=1)
# Run the Grid Search
grid_search_results.fit(X_train,y_train)
# Print Results of Grid Search
cv_res = grid_search_results.cv_results_
rows = []

for score,params in zip(cv_res['mean_test_score'],cv_res['params']):
    rmse = np.sqrt(-score)
    params['RMSE'] = rmse
    rows.append(params)
pd.DataFrame.from_dict(rows).sort_values(by='RMSE').head(10)
#  Select the best performing model to final training
reg_forest = grid_search_results.best_estimator_
print('Here is our best performing model on cross validation')
print('-'*50)
reg_forest
# Training Final model on full dataset
model,rmse = single_model_trainer(reg_forest,X_train,y_train,validation_set=False)
print(f'Final Model Training Error : {rmse}')
# Transform Tests Data Set for final evaluation
X_test = test.drop('median_house_value',axis=1)
y_test = test['median_house_value']
X_test = full_pipeline.transform(X_test)
# Evaluate the model on test set
y_pred = reg_forest.predict(X_test)
mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mse)
print(f'Final Model Testing Error : {rmse}')
error = y_test-y_pred
fig,ax = plt.subplots(figsize=(10,8))
ax = ax.hist(error,bins=100)
plt.savefig(os.path.join(run_dir,'Error_Distribution.jpg'))
# Get the 95% confidence interval of the testing set error
confidence = 0.95
squared_error = (y_pred - y_test)**2
CI = np.sqrt(stats.t.interval(confidence,len(squared_error)-1,loc=squared_error.mean(),scale=stats.sem(squared_error)))
print(f'Testing Error 95% Confidence Interval is Between {CI[0]} - {CI[1]}')
plt.figure(figsize=(12,8))
plot = plt.hist([test['median_house_value'],y_pred],bins=20,label=['True Value','Predicted Value'])
plt.xlabel('House Price')
plt.ylabel('Distribution')
plot = plt.legend()
plot = plt.title('True Value v/s  Predicted Value Distribution Comparision',fontsize=18)
plt.savefig(os.path.join(run_dir,'True_vs_Predicted_Distro.jpg'))
# Plot the Latitude & Longitude to visualize the distribution of population and median house value
sct_plot = test.plot(kind='scatter',x='longitude',y='latitude',alpha=0.6,s=test['population']/100,label='popilation',c=y_pred,cmap=plt.get_cmap('jet'),figsize=(12,8),colorbar=True)
sct_plot = plt.legend()
sct_plot = plt.title('Latitude | Longitude Distribution of House Price over population')
plt.savefig(os.path.join(run_dir,'Predicted_Geo_Distro.jpg'))
