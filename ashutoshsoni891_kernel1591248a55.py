# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/train.csv', parse_dates = ['Date'])
test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/test.csv' , parse_dates = ['Date'])
train.shape , test.shape
train[:5]
test[:5]
#all possible features that could be generated from date
def generate_date_features(df):
    df['day'] = df['Date'].dt.day
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    df['day_of_year'] = df['Date'].dt.dayofyear
    df['dayofyear'] = df['Date'].dt.dayofyear
    df['weekofyear'] = df['Date'].dt.weekofyear
    df['week'] = df['Date'].dt.week
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['weekday'] = df['Date'].dt.weekday
    df['quarter'] = df['Date'].dt.quarter
    df['daysinmonth'] = df['Date'].dt.daysinmonth
    df['is_month_start'] = df['Date'].dt.is_month_start
    df['is_month_end'] = df['Date'].dt.is_month_end
    df['is_quarter_start'] = df['Date'].dt.is_quarter_start
    df['is_quarter_end'] = df['Date'].dt.is_quarter_end
    df['is_year_start'] = df['Date'].dt.is_year_start
    df['is_year_end'] = df['Date'].dt.is_year_end
    df['is_leap_year'] = df['Date'].dt.is_leap_year
    df = df.replace(False , 0).replace(True , 1)
    
    return df
train.info()
train = generate_date_features(train)
train[:5]
train['Target'].value_counts().plot(kind = 'pie',autopct='%1.0f%%' , figsize = (15 ,8))
import seaborn as sns

plt.figure(figsize=(15,8))

sns.scatterplot(x = 'dayofyear' , y = 'TargetValue',
               hue ='Target',
               data = train)
train.plot(kind = 'scatter',x = 'dayofyear' , y = 'TargetValue' , figsize = (20,8) , c= 'Id' , colormap = 'viridis')
train.isna().sum()
train['County'].value_counts()
train['Province_State'].value_counts()
for cols in train.columns:
    if len(train[cols].value_counts()) == 1:
        print(cols)
        train = train.drop(cols , axis = 1)
train[:5]
train.plot(x = 'Date' , y = 'TargetValue' , kind = 'scatter' , figsize = (20 , 8))
train.info()
split = int(len(train) * 0.8)

trainX , testX , trainY , testY = train.drop('TargetValue' , axis = 1)[:split] ,train.drop('TargetValue' , axis = 1)[:-split]  , train['TargetValue'][:split] , train['TargetValue'][:-split]

trainX = trainX.drop(['Id', 'Date' , 'County', 'Province_State', 'Country_Region'],axis = 1)
testX = testX.drop(['Id','Date' ,'County', 'Province_State', 'Country_Region'],axis = 1)
trainX.shape , testX.shape , trainY.shape , testY.shape
from sklearn.preprocessing import StandardScaler

cat_features = trainX.select_dtypes(include = ['object']).columns
num_features = trainX.select_dtypes(include = ['int64' , 'float64']).columns

num_features
cat_features
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler , OneHotEncoder
from sklearn.compose import ColumnTransformer

numeric_transformer = Pipeline(steps =[
    ('imputer' , SimpleImputer(strategy = 'median')),
    ('scaler' , StandardScaler())
])

categorical_transformer =  Pipeline(steps = [
    ('imputer' ,SimpleImputer(strategy = 'most_frequent')),
    ('onehot' , OneHotEncoder(handle_unknown = 'ignore'))
])


preprocessor = ColumnTransformer(transformers = [
    ('num' , numeric_transformer , num_features),
    ('cat' , categorical_transformer , cat_features)
])




from sklearn.ensemble import RandomForestRegressor

rf = Pipeline(steps = [('preprocessor' , preprocessor),
                      ('regressor' , RandomForestRegressor())])


rf.fit(trainX , trainY)
rf.score(testX , testY)
test = generate_date_features(test)
forecastid = test['ForecastId']
test = test.drop(['Date','ForecastId' ,'County' , 'Province_State' , 'Country_Region','year','is_year_start','is_year_end','is_leap_year'] , axis= 1)
test[:5]
test.shape
TargetValue = rf.predict(test)
submit = pd.DataFrame({'ForecastId_Quantile' : forecastid , 
             'TargetValue' : TargetValue})
submit.to_csv('submission.csv' , index = False)
# regressors = []

# for regressor in regressors:
#     pipe = Pipeline(steps=[('preprocessor' , preprocessor),
#                           ('regressor' ,regressor)])
#     pipe.fit(trainX , trainY)
#     print(regressor)
#      print('Model Score: ', pipe.score(testX , testY))
    
    
# param_grid = { 
#     'classifier__n_estimators': [200, 500],
#     'classifier__max_features': ['auto', 'sqrt', 'log2'],
#     'classifier__max_depth' : [4,5,6,7,8],
#     'classifier__criterion' :['gini', 'entropy']}
# from sklearn.model_selection import GridSearchCV
# CV = GridSearchCV(rf, param_grid, n_jobs= 1)
                  
# CV.fit(X_train, y_train)  
# print(CV.best_params_)    
# print(CV.best_score_)