import os
from zipfile import ZipFile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import missingno as msm
import pandas as pd
from scipy import stats

from sklearn.feature_selection import SelectKBest, f_regression, SelectFromModel, mutual_info_regression
from sklearn.model_selection import cross_val_score, cross_val_predict, RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow import keras
data = pd.read_csv('../input/trainset/Train.csv', low_memory=False, parse_dates=['saledate'])
data.head()
data.describe()
df_eda = data.copy()
plt.figure(figsize=(13, 5))
i = 0
for attribute in ('Engine_Horsepower', 'Drive_System'):
  i += 1
  plt.subplot(1, 2, i)
  sns.boxplot(df_eda[attribute], df_eda['SalePrice'])
plt.show()
# Missing value
msm.bar(df_eda, figsize=(24, 5), fontsize=16, labels=True, log=False )
plt.show()
(df_eda.isnull().sum()/len(df_eda)).sort_values()
# Let's identify the which feature contains more than 90% missing value
heigh_nan_features = [feature for feature in df_eda.columns  if (df_eda[feature].isna().sum()/len(df_eda)) > 0.9]
heigh_nan_features
# Categorical features
categorical_feature_list = [feature for feature in df_eda.columns if feature != 'saledate' and df_eda[feature].dtype == 'O']

for feature in categorical_feature_list:
  print(f'{feature :-<50} {df_eda[feature].nunique()}')
# Numerical categories analysis
numerical_feature_list = [feature for feature in df_eda.columns if feature not in ('SalePrice', 'saledate') and df_eda[feature].dtype != 'O']
discret_value_feature = [feature for feature in numerical_feature_list if len(df_eda[feature].unique()) < 25]
sns.countplot(x='datasource', data=df_eda[['datasource']]);
continuous_value_feature = [feature for feature in numerical_feature_list if df_eda[feature].nunique() > 25]
def plot_hist (nrow=1, ncol=1, feature_list=None, figsize=(24, 10)):
  plt.figure(figsize=figsize)
  i = 0
  for feature in feature_list:
    i += 1
    plt.subplot(nrow, ncol, i)
    sns.distplot(df_eda[feature], bins=50)
  plt.show()
plot_hist(nrow=2, ncol=3, feature_list=continuous_value_feature)
#2 SalePrice
df_eda['SalePrice'].describe()
sns.distplot(df_eda['SalePrice'], bins=50);
df_eda['SalePrice'].skew(), df_eda['SalePrice'].kurt()
sns.boxplot(df_eda['SalePrice']);
# Descret value features
sns.boxplot(x='datasource', y='SalePrice', data=df_eda);
# continuous value feature
def scatter_plot (nrow=1, ncol=1, feature_list=None, figsize=(24, 10), target='SalePrice'):
  plt.figure(figsize=figsize)
  i = 0
  for feature in feature_list:
    i += 1
    plt.subplot(nrow, ncol, i)
    sns.scatterplot(df_eda[feature], df_eda[target])
  plt.show()
scatter_plot(nrow=2, ncol=3, feature_list=continuous_value_feature)

corr = df_eda.corr()
plt.figure(figsize=(7, 7))
sns.heatmap(corr, cmap='YlGn_r', cbar=False, annot=True);
# Let's plot what we have seen in heatmap using pairplot
sns.pairplot(df_eda[numerical_feature_list + ['SalePrice']], height = 2.5);
saleprice_scaled = StandardScaler().fit_transform(df_eda['SalePrice'].to_numpy().reshape(-1, 1));
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)
def statical_analysis (df, attribute_name, figsize=(15, 5)):

  plt.figure(figsize=figsize)
  i = 0
  for item in (1, 2):
    i +=1
    plt.subplot(1, 2, i)
    if i == 1: sns.distplot(df[attribute_name], bins=50, fit=stats.norm)
    else:  stats.probplot(df[attribute_name], plot=plt)
    
  plt.show()
# For sale price 
statical_analysis(df_eda, 'SalePrice')
# let's apply log transform to SalePrice
df_log = df_eda.copy()
df_log['SalePrice'] = np.log1p(df_log['SalePrice'])
statical_analysis(df_log, 'SalePrice')
numerical_feature_list
statical_analysis(df_eda, 'MachineHoursCurrentMeter')
#apply log transform
df_log['MachineHoursCurrentMeter'] = np.log1p(df_log['MachineHoursCurrentMeter'])
statical_analysis(df_log, 'MachineHoursCurrentMeter')
df = data.copy()
# let's first work on sale date 
def date_preprocessing (dataFrame, feature='saledate'):
  dataFrame['saleYear'] = dataFrame[feature].dt.year
  dataFrame['saleMonth'] = dataFrame[feature].dt.month
  dataFrame['saleDay'] = dataFrame[feature].dt.day
  dataFrame['saleDayOfWeek'] = dataFrame[feature].dt.dayofweek
  dataFrame['saleDayOfYear'] = dataFrame[feature].dt.dayofyear
  dataFrame.drop(feature, inplace=True, axis=1)
date_preprocessing(df)
def Ordinal_encoder (dataFrame, feature_list):
  mask = {'Mini': 1, 'Small': 1, 'Medium': 2, 'Large / Medium': 3,  'Large': 3, 'Low': 1, 'High': 3}
  for label in feature_list:
    dataFrame[label] = dataFrame[label].map(mask)
Ordinal_encoder(df, ['ProductSize', 'UsageBand'])
# Now let's handle Missing value
class FillMissing ():

  def fill_categorical (self,  dataFrame):
    for label, content in  dataFrame.items():
      if pd.api.types.is_string_dtype(content):
         dataFrame[label].fillna('missing', inplace=True)
    # return df

  def fill_numerical (self,  dataFrame):
    for label, content in  dataFrame.items():
     if pd.api.types.is_numeric_dtype(content):
        dataFrame[label] = content.fillna(content.median())
    # return df
fill_missing_value = FillMissing()
fill_missing_value.fill_categorical(df)
fill_missing_value.fill_numerical(df)
def nominal_encoder (dataFrame, label_list):

  for label, content in  dataFrame.items():
    if pd.api.types.is_string_dtype(content):
       dataFrame[label] = content.astype('category').cat.as_ordered()
       dataFrame[label] = pd.Categorical(content).codes + 3
label_list = [feature for feature in df.columns if df[feature].dtype == 'O']
nominal_encoder(df, label_list)
#drop IDs 
df =df.drop(['SalesID', 'MachineID','ModelID','auctioneerID'], axis=1)
# We will be using decision to simply findout which value of k that gives small MSE.
feature_selection = make_pipeline(SelectKBest(score_func=f_regression, k=52), DecisionTreeRegressor(max_depth=10, random_state=0))
# For the purpose will split the training set into train and test. It is important to bear in mind that the mentioned splited dataset will be use only for finding best K value
X_1, y_1 = df.drop('SalePrice', axis=1), df['SalePrice']
X_1, X_2, y_1, y_2 = train_test_split(X_1, y_1, test_size=.2)
feature_selection.fit(X_1, y_1)
y_1_pred, y_2_pred= feature_selection.predict(X_1),  feature_selection.predict(X_2)
print(f'mse_1: {np.sqrt(mean_squared_error(y_1, y_1_pred))}, mse_2:{np.sqrt(mean_squared_error(y_2, y_2_pred))}')
def scale_feature (X):
  scaled = StandardScaler().fit_transform(X)
  return scaled
X_train, y_train = df.drop('SalePrice', axis=1), df['SalePrice']
X_train = scale_feature(X_train)
def model_training (models, X_train, y_train, cv=3):
  for name, model in models.items():
    print(name)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')
    print({'score': np.sqrt(-scores).mean(), 'std': np.sqrt(-scores).std() })
models = {'KNeighborsRegressor': KNeighborsRegressor(),
          'RandomForestRegressor': RandomForestRegressor(), 
          'ExtraTreesRegressor': ExtraTreesRegressor(),  
          'AdaBoostRegressor': AdaBoostRegressor(), 
          'GradientBoostingRegressor':  GradientBoostingRegressor(), 
          'XGBRegressor':  XGBRegressor()
          }

model_training(models, X_train, y_train)
def hyperparameters_tuning (model, X_train, y_train, param_grid, cv=5):
  grid = RandomizedSearchCV(model, param_grid, n_iter=10, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42, verbose=2, cv=cv)
  grid.fit(X_train, y_train)
  
  return grid.best_estimator_
extra_forest_param_grid = {'n_estimators':[150, 300, 600], 
                          'min_samples_leaf':[2, 3],  
                          'min_samples_split':[14, 12], 
                          'max_features':[0.7, 1],
                          'max_samples':[10000], # 10000 has been used due to the limited available memory of notebook (in Kaggle).
                          'bootstrap':[None, True]
              }
extra_forest = hyperparameters_tuning(ExtraTreesRegressor(), X_train, y_train, extra_forest_param_grid)
extra_forest
# let's prepare validation data 
X_valid = pd.read_csv('../input/bluebook-for-bulldozers/Valid.csv', parse_dates=['saledate'])
y_valid = pd.read_csv('../input/bluebook-for-bulldozers/ValidSolution.csv')
y_valid = y_valid['SalePrice']
#drop IDs 
X_valid =X_valid.drop(['SalesID', 'MachineID','ModelID','auctioneerID'], axis=1)
date_preprocessing(X_valid)
Ordinal_encoder(X_valid, ['ProductSize', 'UsageBand'])
fill_missing_value.fill_categorical(X_valid)
fill_missing_value.fill_numerical(X_valid)
nominal_encoder(X_valid, label_list)
X_valid = scale_feature(X_valid)
def evaluate_best_model (X_train, y_train, X_valid, y_valid):

    estimator = ExtraTreesRegressor(bootstrap=None, ccp_alpha=0.0, criterion='mse',
                    max_depth=None, max_features=0.7, max_leaf_nodes=None,
                    max_samples=None, min_impurity_decrease=0.0,
                    min_impurity_split=None, min_samples_leaf=2,
                    min_samples_split=12, min_weight_fraction_leaf=0.0,
                    n_estimators=300, n_jobs=None, oob_score=False,
                    random_state=None, verbose=0, warm_start=False)
    
    estimator.fit(X_train, y_train)
    y_train_pred = estimator.predict(X_train)
    y_valid_pred = estimator.predict(X_valid)

    return {
          'MSE_train': np.sqrt(mean_squared_error(y_train, y_train_pred)),
          'MSLE_train':np.sqrt( mean_squared_log_error(y_train, y_train_pred)),
          'MSE_valid': np.sqrt(mean_squared_error(y_valid, y_valid_pred)),
          'MSLE_valid': np.sqrt(mean_squared_log_error(y_valid, y_valid_pred))
      }
evaluate_best_model(X_train, y_train, X_valid, y_valid)
FNN = keras.models.Sequential([
                               Flatten(input_shape=[52]),
                               Dense(350, activation='relu', kernel_initializer='lecun_normal'),
                               keras.layers.Dropout(.3),
                               Dense(250, activation='relu', kernel_initializer='lecun_normal'),
                               keras.layers.Dropout(.3),
                               Dense(150, activation='relu', kernel_initializer='lecun_normal'),
                               keras.layers.Dropout(.4),
                               Dense(100, activation='relu', kernel_initializer='lecun_normal'),
                               keras.layers.Dropout(.5),
                               Dense(70, activation='relu', kernel_initializer='lecun_normal'),
                               keras.layers.Dropout(.5),
                               Dense(1)
])
FNN.compile(optimizer='adam', loss='mse', metrics=['mse'])
history = FNN.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=100, batch_size=10, callbacks=[keras.callbacks.EarlyStopping(patience=10)])
plt.plot(history.history['loss'], label='Training')
plt.plot(history.history['val_loss'], label='Validation')
plt.legend();
def evaluate_feedforwardNN (X_train, y_train, X_valid, y_valid):
    y_train_pred = FNN.predict(X_train)
    y_valid_pred = FNN.predict(X_valid)

    
    return {
          'MSE_train': np.sqrt(mean_squared_error(y_train, y_train_pred)),
          'MSLE_train':np.sqrt( mean_squared_log_error(y_train, y_train_pred)),
          'MSE_valid': np.sqrt(mean_squared_error(y_valid, y_valid_pred)),
          'MSLE_valid': np.sqrt(mean_squared_log_error(y_valid, y_valid_pred))
      }
evaluate_feedforwardNN (X_train, y_train, X_valid, y_valid)
# Let's prepare the test data for prediction
testset =  pd.read_csv('../input/bluebook-for-bulldozers/Test.csv', parse_dates=['saledate'])
#drop IDs 
X_test =testset.drop(['SalesID', 'MachineID','ModelID','auctioneerID'], axis=1)
date_preprocessing(X_test)
Ordinal_encoder(X_test, ['ProductSize', 'UsageBand'])
fill_missing_value.fill_categorical(X_test)
fill_missing_value.fill_numerical(X_test)
nominal_encoder(X_test, label_list)
X_test = scale_feature(X_test)
X_test_predict = FNN.predict(X_test)
prediction = pd.DataFrame()
prediction['SalesID'] = testset['SalesID']
prediction['SalePrice']= X_test_predict
prediction
