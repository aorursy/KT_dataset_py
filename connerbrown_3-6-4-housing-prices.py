# tools
import numpy as np # linear algebra
np.random.seed(42)
import pandas as pd # data processing
import seaborn as sns # plotting
import matplotlib.pyplot as plt # plotting
from scipy import stats # t-test
import time # runtime
from sklearn import metrics # roc, auc score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# models
from sklearn.linear_model import LinearRegression # Linear(fit_intercept=True, normalize=False, n_jobs=1)
from sklearn.neighbors import KNeighborsRegressor # KNN(n_neighbors=5, weights='uniform', n_jobs=1)
from sklearn.tree import DecisionTreeRegressor # DTR(max_depth=None, min_samples_split=2, max_features=None, max_leaf_nodes=None)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor # RFR(n_estimators=10, max_depth=None, min_samples_split=2, max_features='auto', max_leaf_nodes=None, n_jobs=1), GBR(learning_rate=0.1, n_estimators=100, subsample=1.0, max_depth=3, max_features=None, max_leaf_nodes=None)
from sklearn.svm import SVR # SVR(kernel='rbf')
data_path = '../input/Melbourne_housing_FULL.csv'
df = pd.read_csv(data_path)
display(df.shape)
display(df.head(3))
df_null = df.isnull().sum()
display(df_null[df_null > 0])
# drop Landsize, BuildingArea, YearBuilt, Latitude, Longitude
df_feat = df.drop(['Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude'], axis = 1)
df_feat = df_feat.dropna(axis = 0, subset = ['Distance', 'Postcode', 'CouncilArea', 'Regionname', 'Propertycount'])
df_feat = df_feat.reset_index()
# impute 
print ("New size of Dataset {}".format(df_feat.shape))
# % overlap of missing Prices
df_overlap = df_feat.isnull()
display(df_overlap.sum()[df_overlap.sum() > 0])
# Price + Bedroom2
price_bedroom2_nans = df_overlap[['Price', 'Bedroom2']].all(axis = 1).sum()
print ("Price + Bedroom2 overlapping NaNs: %0.0f" % (price_bedroom2_nans))
# Price + Bathroom
price_bathroom_nans = (df_overlap[['Price', 'Bathroom']]).all(axis = 1).sum()
print ("Price + Bathroom overlapping NaNs: %0.0f" % (price_bathroom_nans))
# Price + Car
price_car_nans = (df_overlap[['Price','Car']]).all(axis = 1).sum()
print ("Price + Car overlapping NaNs: %0.0f" % (price_car_nans))
obj_cols = list(df_feat.dtypes[df_feat.dtypes == 'object'].index)
for col in obj_cols:
    print ("{} has {} unique values".format(col, (len(df_feat[col].value_counts().index))))
# top Suburb (if Suburb in top 20 set to 1)
top_suburbs = list(df_feat['Suburb'].value_counts().index)[:20]
df_feat['is_top_suburb'] = df_feat['Suburb'].isin(top_suburbs)
# grab st/av/dr/rd from Address, make dummies for top 20
# split and take the last value in array
# df_feat["street_address"] = df_feat["Address"].str.split().str[-1]
# get_dummies Type, Method, CouncilArea, Regionname
#df_feat = pd.get_dummies(df_feat, columns = ['Type', 'Method', 'CouncilArea', 'Regionname'])
# top SellerG (if sellerg in top 20 set to 1)
top_sellerg = list(df_feat['SellerG'].value_counts().index)[:20]
df_feat['is_top_sellerg'] = df_feat['SellerG'].isin(top_sellerg)
# grab month/year from Date (last 7 characters)
month_year = df_feat["Date"].str[-7:]
df_feat['month'] = month_year.str[:2]
df_feat['year'] = month_year.str[-4:]
# drop feature engineered columns
df_feat = df_feat.drop(['index', 'Suburb','Address', 'SellerG', 'Date'], axis = 1)
'''# check distribution of numeric variab
for col in ['Price','Rooms','Distance','Bedroom2','Bathroom','Car','Propertycount','month','year']:
    df_feat[col].hist()
    plt.title(col)
    plt.show()
    '''
# transformations dataframe
df_trans = pd.DataFrame()
df_trans = np.log(df_feat[['Price','Distance','Bedroom2','Bathroom','Car']] + 1)
df_trans['Propertycount'] = df_feat['Propertycount'].apply(np.sqrt)
df_trans['Rooms'] = df_feat['Rooms'].apply(np.sqrt)
df_trans['rooms_bedroom2'] = df_trans['Rooms'] / (df_trans['Bedroom2'] + 1)
df_trans.drop(['Rooms', 'Bedroom2'], inplace = True, axis = 1)
df_trans['month'] = df_feat['month'].astype(int)
df_trans['year'] = df_feat['year'].astype(int)
'''
# make sure the transformations made normal distributions
for col in df_trans.columns:
    df_trans[col].hist()
    plt.title(col)
    plt.show()
    '''
# include binary features
df_trans['is_top_sellerg'] = df_feat['is_top_sellerg'].astype(int)
df_trans = pd.concat([df_trans, df_feat['Type'], df_feat['CouncilArea'], df_feat['Regionname']], axis = 1)
df_trans = pd.get_dummies(df_trans, columns = ['Type', 'CouncilArea', 'Regionname'])
df_trans['is_top_suburb'] = df_feat['is_top_suburb'].astype(int)
# dropna
df_dropna = df_trans.dropna()
X = df_dropna.drop('Price', axis = 1)
X_norm = (X - X.min()) / (X.max() - X.min())
Y = df_dropna['Price']
# correlation matrix - transformed
plt.figure(figsize = (20,20))
sns.heatmap(df_dropna.corr(), vmin = .3)
#PCA, n_components = 4 from scree plot
n_components = 22
pca = PCA(n_components=n_components)
X_PCA = pca.fit_transform(X_norm)

print('The percentage of total variance in the dataset explained by each',
    'component from Sklearn PCA.\n',
    pca.explained_variance_ratio_)
print ('Sum of variance explained by {} components\n%0.4f'.format(n_components) % (pca.explained_variance_ratio_.sum()))
# scree plot
# train_test_split on X_norm or X_PCA
X_train, X_test, Y_train, Y_test = train_test_split(X_norm, Y)
df_eval = pd.DataFrame()
def evaluate(model_name, Y_pred):
    evals = np.zeros(3)
    # Mean Absolute Error
    mae_log = metrics.mean_absolute_error(Y_test, Y_pred)
    evals[0] = np.mean(abs(np.e**(Y_test) - np.e**(Y_pred)))
    # R-squared
    evals[1] = metrics.r2_score(Y_test, Y_pred)
    # Explained variance
    evals[2] = metrics.explained_variance_score(Y_test, Y_pred)
    df_eval[model_name] = evals
# runtime ~ 0.15 seconds (default train_test_split)
#### Linear Regressor
start = time.time()
linear = LinearRegression(fit_intercept=True, normalize=False, n_jobs=1)
linear.fit(X_train, Y_train)
linear_pred = linear.predict(X_test)
print ("Runtime %0.2f" % (time.time() - start))
evaluate('Linear Regression',linear_pred)
# runtime ~ 0.56 seconds (default train_test_split)
#### KNN Regressor
start = time.time()
knn_model = KNeighborsRegressor(n_neighbors=5, algorithm='ball_tree', n_jobs = 3)
knn_model.fit(X_train, Y_train)
knn_pred = knn_model.predict(X_test)
print ("Runtime %0.2f" % (time.time() - start))
evaluate('KNN',knn_pred)
# runtime ~ 0.21 seconds (default train_test_split)
#### RandomForestClassifier 
start = time.time()
DTR = DecisionTreeRegressor(max_depth=None, min_samples_split=2, max_features=None, max_leaf_nodes=None)
DTR.fit(X_train, Y_train)
DTR_pred = DTR.predict(X_test)
print ("Runtime %0.2f" % (time.time() - start))
evaluate('Decision Tree',DTR_pred)
# runtime ~ 0.95 seconds (default train_test_split)
#### RandomForestRegressor
start = time.time()
RFR = RandomForestRegressor(n_estimators=10, max_depth=None, min_samples_split=2, max_features='auto', max_leaf_nodes=None, n_jobs=1)
RFR.fit(X_train, Y_train)
RFR_pred = RFR.predict(X_test)
print ("Runtime %0.2f" % (time.time() - start))
evaluate('Random Forest', RFR_pred)
# runtime ~ 47.82 seconds (default train_test_split)
#### SVR
start = time.time()
SVR_model = SVR(kernel='rbf')
SVR_model.fit(X_train, Y_train)
SVR_pred = SVR_model.predict(X_test)
print ("Runtime %0.2f" % (time.time() - start))
evaluate('SVM', SVR_pred)
# runtime ~ 3.19 seconds (default train_test_split)
# n_estimators = 1000, max_depth = 4, subsample, 0.5, learning_rate = 0.001
#### Gradient Boost Regressor
start = time.time()
GBR = GradientBoostingRegressor(learning_rate=0.1, n_estimators=1000, subsample=1.0, max_depth=3, max_features=None, max_leaf_nodes=None)
GBR.fit(X_train, Y_train)
GBR_pred = GBR.predict(X_test)
print ("Runtime %0.2f" % (time.time() - start))
evaluate('Gradient Boost', GBR_pred)
df_eval.rename(index={0: 'MAE Score', 
                      1: 'R-squared', 
                      2: 'Explained Variance'
                     }, inplace = True)
# Accuracy scores, highlight highest in each row
df_eval.style.highlight_max(axis = 1)
# print out the top features from Gradient Boosting
feature_importance = GBR.feature_importances_

# Make importances relative to max importance.
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.figure(figsize=(20,20))
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, X_train.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance from Gradient Bosting')
plt.show()


