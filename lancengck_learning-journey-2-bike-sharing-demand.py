#STEP 1: IMPORTING LIBRARIES AND DATASET

# Importing the libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_log_error


# Importing the dataset from Kaggle
traindf = pd.read_csv('../input/bike-sharing-demand/train.csv')
testdf = pd.read_csv('../input/bike-sharing-demand/test.csv')
# Print first 5 rows of traindf 
traindf.head()
# Print first 5 rows of testdf 
testdf.head()
print("Brief look at train set")
traindf.info()
traindf['datetime'] = pd.to_datetime(traindf['datetime'])
traindf = traindf.set_index('datetime')

# Creating relevant datetime columns
traindf['year'] = traindf.index.year
traindf['month'] = traindf.index.month
traindf['hour'] = traindf.index.hour

traindf.info()
# Encoding categorical data 

traindf['spring'] = (traindf['season']==1)*1
traindf['summer'] = (traindf['season']==2)*1
traindf['fall'] = (traindf['season']==3)*1
traindf['winter'] = (traindf['season']==4)*1

traindf['clear'] = (traindf['weather']==1)*1
traindf['cloudy'] = (traindf['weather']==2)*1
traindf['light_snow'] = (traindf['weather']==3)*1
traindf['heavy_snow'] = (traindf['weather']==4)*1

traindf = traindf.drop(['season'],axis=1)
traindf = traindf.drop(['weather'],axis=1)

traindf.head()
# Checking for Correlation
cor = traindf.corr()
sns.set(font_scale=1.25)
f, ax = plt.subplots(figsize=(15, 15))
sns.heatmap(cor, cmap="YlGnBu", annot=True, fmt='.2f', square =True, cbar=False);

# Dropping redundant columns
cols_to_drop = ['holiday', 'atemp', 'summer', 'winter','cloudy','heavy_snow','casual','registered']
traindf = traindf.drop(cols_to_drop, axis=1)

# Check for missing data
total = traindf.isnull().sum().sort_values(ascending=False)
percent = (traindf.isnull().sum()/traindf.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data)
# Train_test_split using traindf
traindf_X = traindf.drop(['count'],axis=1)
traindf_y = traindf[['count']]
X_train, X_test, y_train, y_test = train_test_split(traindf_X,traindf_y,test_size=.2, random_state=8)

# Initial testing based on several models
models=[RandomForestRegressor(),AdaBoostRegressor(),SVR(),KNeighborsRegressor()]
model_names=['RandomForestRegressor','AdaBoostRegressor','SVR','KNeighborsRegressor']

# Compiling initial base results using RMSLE
rmsle=[]
model_result={}
for model in range (len(models)):
    clf=models[model]
    clf.fit(X_train,y_train.values.ravel())
    test_pred=clf.predict(X_test)
    rmsle.append(np.sqrt(mean_squared_log_error(test_pred,y_test)))
model_result={'Modelling Algo':model_names,'RMSLE':rmsle}   
rmsle_frame=pd.DataFrame(model_result)
rmsle_frame
# Fitting the best parameters to traindf
params_dict={'n_estimators':[300],'bootstrap':[True],'max_depth':[50],'min_samples_leaf':[2],'min_samples_split':[2],'n_jobs':[-1],'max_features':['auto']}
clf_rf=GridSearchCV(estimator=RandomForestRegressor(),param_grid=params_dict,scoring='neg_mean_squared_log_error',cv=5)
clf_rf.fit(X_train,y_train.values.ravel())
pred=clf_rf.predict(X_test)
print((np.sqrt(mean_squared_log_error(pred,y_test))))
# Converting 'datetime' datatype and set as index
class DatetimeConverter(BaseEstimator):
    def __init__(self):
        pass
    def fit(self, documents, y=None):
        return self
    def transform(self, x_dataset):
        x_dataset['datetime'] = pd.to_datetime(x_dataset['datetime'])
        x_dataset = x_dataset.set_index('datetime')
        x_dataset['year'] = x_dataset.index.year
        x_dataset['month'] = x_dataset.index.month
        x_dataset['hour'] = x_dataset.index.hour        
        
        return x_dataset

# Creating custom class for binary encoding
class BinaryEncoder(BaseEstimator):
    def __init__(self):
        pass
    def fit(self, documents, y=None):
        return self
    def transform(self, x_dataset):
        x_dataset['spring'] = (x_dataset['season'] == 1)*1
        x_dataset['fall'] = (x_dataset['season'] == 3)*1
        x_dataset['clear'] = (x_dataset['weather'] == 1)*1
        x_dataset['light_snow'] = (x_dataset['weather'] == 3)*1
        
        return x_dataset

# Create transformer to drop irrelevant columns
drop_col = ColumnTransformer(remainder='passthrough',
                                transformers=[('drop_columns', 'drop', ['holiday', 'atemp', 'weather', 'season'])])

model_pipeline = Pipeline(steps=[('converting_datetime', DatetimeConverter()),
                                 ('create_binary_columns', BinaryEncoder()),
                                 ('drop_columns', drop_col),
                                 ('random_forest_regressor', RandomForestRegressor(n_estimators=300,
                                                                                   bootstrap=True,
                                                                                   max_depth=50,
                                                                                   min_samples_leaf=2,
                                                                                   min_samples_split=2,
                                                                                   max_features='auto'))])


# Re-importing the dataset from Kaggle
traindf = pd.read_csv('../input/bike-sharing-demand/train.csv')
testdf = pd.read_csv('../input/bike-sharing-demand/test.csv')
traindf_X = traindf.drop(['count','casual','registered'],axis=1)
traindf_y = traindf[['count']]
model_pipeline.fit(traindf_X,traindf_y.values.ravel())
submission=pd.DataFrame(model_pipeline.predict(testdf), index=testdf['datetime'])
submission.rename(columns={0:'count'}, inplace=True)
submission.to_csv('submission.csv', index=True)
print(submission)