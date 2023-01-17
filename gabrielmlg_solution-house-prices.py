# Import packages

import pandas as pd
import numpy as np

# Machine Learning
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error


# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')
%matplotlib inline
# extract datasets (train and test)

train_v1 = pd.read_csv('../input/train.csv')
test_v1 = pd.read_csv('../input/test.csv')
# Show dataset

train_v1.head()
# Shape
train_v1.shape
train_v1.dtypes
# Sale price distribution
sns.distplot(train_v1.SalePrice)
# Missing values
train_v1.isnull().sum().sort_values(ascending=False).head(20)
# Preprocessing
# 1. Transform missing values and factorize

def preProcessing(df):
    le = LabelEncoder()
    df_result = df.copy()
    
    for column in df_result:
        if type(df_result[column][0]) == str:
            #print('Col: %s' %column)
            df_result[column].fillna('N/A', inplace=True) # Missing values
            df_result[column] = le.fit_transform(df_result[column]) # Factorize
        else:
            if column == 'Alley' or column == 'MiscFeature' or column == 'Fence' or column == 'ExterQual' \
                or column == 'ExterCond' or column == 'BsmtQual' or column == 'BsmtCond' or column == 'BsmtExposure' :
                    df_result[column].fillna('N/A', inplace=True) # Missing values
                    df_result[column] = le.fit_transform(df_result[column]) # Factorize
            else:
                df_result[column].fillna(0, inplace=True) # Missing values
                
    df_result.drop(['3SsnPorch', 'Street', 'LandContour', 'Condition2', 
                    'BsmtFinSF2', 'Utilities', 'BsmtHalfBath', 'BsmtCond', 'MoSold', 'MiscVal'], axis=1, inplace=True)
      
    return df_result
        
            
train_v2 = preProcessing(train_v1)            
test_v2 = preProcessing(test_v1)
# Correlation
df_corr = abs(pd.DataFrame(train_v2.corr()['SalePrice']))
# List features sort by correlation score

df_corr.sort_values('SalePrice', ascending=False)
# Filter by numerics features

print(train_v1.shape)

train_v2 = train_v2.select_dtypes(include=[np.number])
test_v2 = test_v2.select_dtypes(include=[np.number])

print(train_v2.shape)
# Create/split dataset train and test

train_v3 = train_v2.sample(frac=0.7, random_state=101)
test_v3 = train_v2.loc[~train_v2.index.isin(train_v3.index)]
# Split features and target 

train_x = train_v3.drop(['Id', 'SalePrice'], axis=1)
train_y = train_v3.SalePrice

test_x = test_v3.drop(['Id', 'SalePrice'], axis=1)
test_y = test_v3.SalePrice
# Linear Regression

model_lr = LinearRegression()

# Fit model
model_lr = model_lr.fit(train_x, train_y)

# Predict
pred_lr = model_lr.predict(test_x)
# Avaliable result
mean_squared_error(pred_lr, test_y)
# Create data frame compare
df = pd.DataFrame()
df['Id'] = test_v3['Id']
df['SalePriceReal'] = test_y 
df['Predict'] = pred_lr
df['diff'] = df['Predict']-df['SalePriceReal']
df.head()
# Plot real and predict values

import numpy as np
from numpy import ma
import matplotlib.pyplot as plt

y = df['SalePriceReal']
x = df.index

plt.step(x, y, label='Sale Price')

y = df['Predict']
plt.step(x, y, label='Predict')

plt.legend()


plt.show()
# Create CSV result - Linear Regression.
#dfResult = pd.DataFrame()

#dfResult['Id'] = test_v2['Id']
#dfResult['SalePrice'] = model_lr.predict(test_v2.drop('Id', axis=1))
#dfResult.to_csv('result_v2.csv', index=False)
# Test solution with Random Forest Regressor

model_rf = RandomForestRegressor(n_estimators = 100, min_samples_leaf = 5, random_state = 101)

# Create model
modelo_v2 = model_rf.fit(train_x, train_y)

# Test predict
previsoes = modelo_v2.predict(test_x)
# Check 

mean_squared_error(previsoes, test_y)
scores = cross_val_score(modelo_v2, train_x, train_y)
mean = scores.mean()
print(scores)
print(mean)
# Test solution with Extra Tree Regressor

model_extraTreeRegressor = ExtraTreesRegressor(n_estimators=100, min_samples_leaf=5, random_state=101)

# Create model
modelo_v3 = model_extraTreeRegressor.fit(train_x, train_y)

# Test predict
pred_v3 = model_extraTreeRegressor.predict(test_x)
# Check
mean_squared_error(pred_v3, test_y)
# Use GridSearch to optimize Random Forest Regressor

# Create param list
param_grid = {"n_estimators": [10,100],
              "max_features": [1, 3, 10, 'auto', None],
              "min_samples_leaf": [1, 2, 3, 5],
              "bootstrap": [True, False],
              "criterion": ["mse", "mae"]}

# Executing ...
grid_search = GridSearchCV(modelo_v2, param_grid = param_grid)
grid_search.fit(train_x, train_y)
# Print scores

grid_search.grid_scores_
# Create model optimized
model_rf = RandomForestRegressor(n_estimators = 100, min_samples_leaf = 1, random_state = 101, bootstrap=False, 
                                 criterion='mse', max_features=10)

# Model
modelo_v4 = model_rf.fit(train_x, train_y)

# Predict test dataset
pred_v4 = modelo_v4.predict(test_x)
# Check error
mean_squared_error(pred_v4, test_y)
# Create CSV result.
dfResult = pd.DataFrame()

dfResult['Id'] = test_v2['Id']
dfResult['SalePrice'] = modelo_v4.predict(test_v2.drop('Id', axis=1))
dfResult.to_csv('result_v5.csv', index=False)