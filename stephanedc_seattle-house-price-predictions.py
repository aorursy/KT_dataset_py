# Some interesting Lecture and Concepts
# https://www.coursera.org/learn/ml-foundations/notebook/et8PR/predicting-house-prices-assignment
# https://www.kaggle.com/mattcarter865/boston-house-prices
# http://www.neural.cz/dataset-exploration-boston-house-pricing.html
# https://jovianlin.io/data-visualization-seaborn-part-3/
# http://www.bigendiandata.com/2017-06-27-Mapping_in_Jupyter/
# https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
# https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python
# https://www.kaggle.com/darquesm/house-prices-ensemble-regressors
# Importing the Required Librairies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns ; sns.set()
import sklearn
import math
import warnings

from sklearn import preprocessing
from scipy.stats import norm, skew #for some statistics
from sklearn.preprocessing import QuantileTransformer, quantile_transform
from sklearn.compose import TransformedTargetRegressor
from sklearn import linear_model
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn import ensemble
from sklearn import model_selection
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score
from sklearn.metrics import r2_score
from mpl_toolkits.mplot3d import Axes3D

pd.options.display.float_format = '{:.3f}'.format
%matplotlib inline
print('The scikit-learn version is {}.'.format(sklearn.__version__))
# Scaling data with a Standard Scaler
USE_SCALED_DATA = False                       # Bool : Scale or not the data                  

# Filtering data parameters
SUPPRESS_TOP_OUTLIERS = 10                     # Supressing top 2N outliers (Low & High)
FILTER_ZIPCODE_EXCLUDE_TOP_DEVIANCE = 2        # Exclude zipcode with most deviance in price
FILTER_SQFT_LIVING = {'min':20, 'max':7000}    # Filter by living surface
FILTER_PRICE = {'min':2000, 'max':6000000}     # Filter by house price
FILTER_YEAR_BUILT = {'min':1950, 'max':2012}   # Filter by built year
FILTER_GRADE = {'min':-1, 'max':15}            # Filter by grade

# Dataset Testing Ratio
RATIO_TRAIN_TEST = 0.20                        # Exemple : 0.20 => 80% Training / 20ù Testing

# xgboost parameters
XGBOOST_PERFORM_GRID = False                   # Perform grid Search CV optimisation
XGBOOST_CROSS_VALIDATION = False               # Perform cross validation of the model
def compute_score(y_pred, y_test): 
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squarred Error: %.4f" % mse)
    print("Root Mean Squarred Error: %.4f" % math.sqrt(mse))
    print("Mean Absolute Error: %.4f" % mean_absolute_error(y_test, y_pred))
    print("Variance Score (Best possible score is 1): %.4f" % explained_variance_score(y_test, y_pred))
    print("R2Score (Best possible score is 1): %.4f" % r2_score(y_test, y_pred))
    
def plot_prediction(y_pred, y_test):
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, edgecolors=(0, 0, 0))
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
    ax.set_xlabel('Measured Price')
    ax.set_ylabel('Predicted Price')
    ax.set_title('Price Prediction')
    plt.show()
    
def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    compute_score(y_pred, y_test)
    plot_prediction(y_pred, y_test)
    try: 
        print(model.feature_importances_)
    except AttributeError: 
        print("No feature_importances_ for this model")
dataset = pd.read_csv("../input/home_data(3).csv", delimiter=",")        # Read dataset
dataset = dataset.drop(columns=['id', 'date'])                           # Suppress unused columns
print("The Dataset shape is (row, col): {}".format(dataset.shape))       # print the shape of the Dataset
dataset.head()                                                           # Showing some lines of original datas
# Plotting all the features in histogram
ax = dataset.hist(figsize=(18,20))
# Price stats
y = dataset['price']       # Get the target price (y) 
y.describe().astype(int)   # Print some stats about price
# Plotting price distribution
with warnings.catch_warnings():
    ax = sns.distplot(dataset['price'], bins=200, fit=norm)
    sns.set(rc={'figure.figsize':(7,5)})
    ax.set_title('Price Target distribution')
    ax.set(xlim=(50000, 1800000))
# find correlation between atributes
pearson = dataset.corr(method='pearson')

# Draw a heatmap
sns.set(rc={'figure.figsize':(30,12)})
mask = np.zeros_like(pearson)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    ax = sns.heatmap(data=pearson, cmap="YlGnBu", mask=mask, vmax=1, annot=True, square=True)
# Plotting between year of built and number of floors
ax = sns.jointplot(dataset['yr_built'], dataset['floors'], kind='kde', 
                   joint_kws={'alpha':0.5}, 
                   xlim=(FILTER_YEAR_BUILT['min'], FILTER_YEAR_BUILT['max']), 
                   ylim=(0,4), 
                   height=6)
# Plotting between price of built and surface
# NB: Max price & max sqft are fixed for better plotting
ax = sns.jointplot(dataset['price'], dataset['sqft_living'], kind='kde', 
                   joint_kws={'alpha':0.5}, 
                   xlim=(FILTER_PRICE['min'],1000000), 
                   ylim=(FILTER_SQFT_LIVING['min'], 4000),
                   height=6)
# Plot price repartition
ax = sns.jointplot(dataset['price'], dataset['sqft_living'], kind='scatter', 
              joint_kws={'alpha':0.5}, xlim=(0,6000000), ylim=(0,9000), height=6)
# Print top outliers
saleprice = dataset['price'][:,np.newaxis]
low_range = saleprice[saleprice[:,0].argsort()][:SUPPRESS_TOP_OUTLIERS]
high_range= saleprice[saleprice[:,0].argsort()][-SUPPRESS_TOP_OUTLIERS:]
print('outer range (low) of the distribution:\n {}'.format(low_range))
print('\nouter range (high) of the distribution:\n {}'.format(high_range))
# Suppress top outliers
if SUPPRESS_TOP_OUTLIERS > 0:
    print("Suppressing Outliers...")
    dataset=dataset.loc[~dataset['price'].isin(list(high_range))]
    dataset=dataset.loc[~dataset['price'].isin(list(low_range))]
    print("New Shape : {} ".format(dataset.shape))
    print("Done")
# Filtering on different criteria
# Surface
if len(FILTER_SQFT_LIVING) > 0:
    print("Filtering by Sqft min {}, max {} ..".format(FILTER_SQFT_LIVING['min'], FILTER_SQFT_LIVING['max']))
    filtered_sqft = dataset.loc[(dataset['sqft_living'] > FILTER_SQFT_LIVING['min']) & 
                                (dataset['sqft_living'] < FILTER_SQFT_LIVING['max'])]
    ratio = len(filtered_sqft) / len(dataset)
    print("New Shape : {} , Ratio of Selected Sqft {}".format(filtered_sqft.shape, ratio))
    dataset = filtered_sqft.copy()

# Zipcode with deviance
if FILTER_ZIPCODE_EXCLUDE_TOP_DEVIANCE > 0:
    print("Filtering by Zipcode Deviance : Top {}".format(FILTER_ZIPCODE_EXCLUDE_TOP_DEVIANCE))
    ds = dataset.groupby(['zipcode']).std()
    zipcode_exclude_list = ds['price'].astype(int).sort_values(ascending=False).nlargest(FILTER_ZIPCODE_EXCLUDE_TOP_DEVIANCE)
    zipcode_exclude_list = zipcode_exclude_list.index.tolist()
    dataset = dataset.loc[(~dataset.zipcode.isin(list(zipcode_exclude_list)))]
    print("New Shape : {} ".format(dataset.shape))

# Grade of House
if len(FILTER_GRADE) > 0:
    print("Filtering by Grade : min {} , max {}".format(FILTER_GRADE['min'], FILTER_GRADE['max']))
    
    filtered_dataset = dataset[(dataset.grade < FILTER_GRADE['max']) & 
                               (dataset.grade > FILTER_GRADE['min'])]
        
    print("Filtering Grade : We exclude {} rows of data".format(len(dataset) - len(filtered_dataset)))    
    print("New Shape : {} ".format(filtered_dataset.shape))    
    dataset = filtered_dataset.copy()

# Built year
if len(FILTER_YEAR_BUILT) > 0:
    print("Filtering by Year built : min {} , max {}".format(FILTER_YEAR_BUILT['min'], FILTER_YEAR_BUILT['max']))

    filtered_dataset = dataset[(dataset.yr_built < FILTER_YEAR_BUILT['max']) & 
                               (dataset.yr_built > FILTER_YEAR_BUILT['min'])]    
    
    print("Filtering Yr_built : We exclude {} rows of data".format(len(dataset) - len(filtered_dataset)))
    print("New Shape : {} ".format(filtered_dataset.shape))        
    dataset = filtered_dataset.copy()

# House Price
if len(FILTER_PRICE) > 0:

    print("Filtering by Price : min {} , max {}".format(FILTER_PRICE['min'], FILTER_PRICE['max']))
    filtered_dataset = dataset[(dataset.price < FILTER_PRICE['max']) & 
                               (dataset.price > FILTER_PRICE['min'])]
    print("Filtering Price : We exclude {} rows of data".format(len(dataset) - len(filtered_dataset)))
    print("New Shape : {} ".format(filtered_dataset.shape)) 
    dataset = filtered_dataset.copy()      
# Plot Infos by ZipCode
dataset.astype('int').groupby(['zipcode'])['price'].describe().plot()
zip_mean_max = dataset.astype('int').groupby(['zipcode'])['price'].mean().idxmax()         # zipcode with the max of mean price
zip_mean = dataset.loc[(dataset['zipcode'] == zip_mean_max)]['price'].mean()               # mean price
print("Zipcode with maximum mean is {} with value: {}".format(zip_mean_max, int(zip_mean)))
# Finding the most expensive zipcode regarding price/m²
moy = dataset.groupby(['zipcode']).mean()
moy['sqm2_living'] = moy['sqft_living'] / 3.28
moy['m2'] = moy['price'] / moy['sqm2_living']
moy[['price','sqft_living', 'sqm2_living', 'm2']].nlargest(5, columns='m2')
ax = sns.jointplot(dataset['long'], dataset['lat'], kind='scatter', 
                   joint_kws={'alpha':0.03}, xlim=(-122.6, -121.6), ylim=(47.2,47.8), height=7)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
xs = dataset['lat']
ys = dataset['long']
zs = dataset['price']
ax.scatter(xs, ys, zs, s=50, alpha=0.1, edgecolors='w')
ax.set_xlabel('Lattitude')
ax.set_ylabel('Longitude')
ax.set_zlabel('Price')
ax.set(zlim=(FILTER_PRICE['min'], 2500000))
plt.show()
# Preparation for Scaling
scaler = preprocessing.StandardScaler()                                               # Init Data scaler
numpy_scaled = scaler.fit_transform(dataset.astype("float64"))                        # return a numpy array
df_scaled = pd.DataFrame.from_records(data=numpy_scaled, columns=dataset.columns)     # get a dataframe
df_scaled.head()                                                                      # show some lines

if USE_SCALED_DATA:
    print("Using Scaled Data")
    print("Shape : {} ".format(df_scaled.shape))
    dataset = df_scaled.copy()
    
dataset.isnull().sum()
# Dataset X, y and splitting test / train
y = dataset.price   
X = dataset.loc[:, ~dataset.columns.isin(['price', 'y_trans'])]
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=RATIO_TRAIN_TEST, 
                                                                        random_state=5, shuffle=True) 
# Gradient Booster & Features Importance
params = {'n_estimators': 650, 'max_depth': 6, 'min_samples_split': 2, 'learning_rate': 0.01, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)
train_and_evaluate(clf, X_train, X_test, y_train, y_test)
test_score = np.zeros((params['n_estimators'],), dtype=np.float64)
featureslist = np.array(X.columns.tolist())

for i, y_pred in enumerate(clf.staged_predict(X_test)):
    test_score[i] = clf.loss_(y_test, y_pred)

# Plotting Deviance
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-', label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-', label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')

# Feature importance
feature_importance = clf.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, featureslist[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()
# Gradient Booster with regularised y distribution
params = {'n_estimators': 700, 'max_depth': 7, 'min_samples_split': 2, 'learning_rate': 0.01, 'loss': 'ls'}
gradboost = TransformedTargetRegressor(
    regressor=ensemble.GradientBoostingRegressor(**params),
    transformer=QuantileTransformer(output_distribution='normal'))
train_and_evaluate(gradboost, X_train, X_test, y_train, y_test)
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
# Regressor parameters
xg_reg = xgb.XGBRegressor(booster="gbtree", objective="reg:linear",colsample_bytree=0.8, min_child_weight=1, 
          learning_rate=0.1, max_depth=6, alpha=2, n_estimators=350, subsample=1, reg_alpha=1e-05)
train_and_evaluate(xg_reg, X_train, X_test, y_train, y_test)
if XGBOOST_PERFORM_GRID:

    # Grid Search for optimizing
    param_test1 = {
     'max_depth': [5,6,7],
     'n_estimators': [250,350,450],
     'min_child_weight': [20],
     'reg_alpha':[1e-5],
     'colsample_bytree': [0.7,0.8]
    }

    gsearch1 = GridSearchCV(estimator = xgb.XGBRegressor(
                            learning_rate =0.1, alpha=1, objective='reg:linear'), 
               param_grid = param_test1, scoring='r2', cv=3, verbose=3)

    gsearch1.fit(X_train, y_train)
    gsearch1.best_params_, gsearch1.best_score_
# Cross Validation of the defined Model
if XGBOOST_CROSS_VALIDATION: 
    
    params = {"objective":"reg:linear",'colsample_bytree': 0.8, 'min_child_weight': 20, 'reg_alpha': 1e-05,
              'learning_rate': 0.1, 'max_depth': 6, 'alpha': 2, 'n_estimators': 350}

    data_dmatrix = xgb.DMatrix(data=X_train, label=y_train)

    cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=5,
                        num_boost_round=100,early_stopping_rounds=10,metrics="rmse", 
                        as_pandas=True, seed=123)

    print((cv_results["test-rmse-mean"]).tail(1))