# Import libraries
from scipy import stats
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("white")
sns.set(style="ticks", color_codes=True)
%matplotlib inline
from sklearn.model_selection import learning_curve, validation_curve, cross_val_score

# Functions
def find_missing_data(df):# missing data
  #missing data
  total = df.isnull().sum().sort_values(ascending=False)
  percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
  missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
  print (missing_data.head(20))


housing = pd.read_csv("../input/housing.csv")
housing.info()
find_missing_data(housing)
## Fill missing data 'total_bedrooms'
g = sns.pairplot(housing,
                 x_vars=["total_bedrooms"],
                 y_vars=["total_rooms","population","households","housing_median_age"])
## Check relationship between features
df_train = housing.loc[:, housing.columns != 'ocean_proximity']
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, square=True,annot=True);
plt.scatter(housing["median_income"],housing["median_house_value"])
plt.xlabel("median income");plt.ylabel("house value")
# 1. Fill missing data
## Get Linear dependency
from sklearn.linear_model import LinearRegression
df_ss = housing.loc[:,['households','total_bedrooms']].dropna() ## 
X,y = np.log(df_ss[df_ss.loc[:, df_ss.columns != 'total_bedrooms'].columns]), \
          np.log(df_ss['total_bedrooms'])

## Get score using original model
linreg = LinearRegression()
linreg.fit(X,y)
scores = cross_val_score(linreg, X,y, cv=10)
print('CV accuracy (original): %.3f +/- %.3f' % (np.mean(scores), np.std(scores))) # 95% correct, Not bad.
# highest_score = np.mean(scores)
# print intercept and coefficients
# print linreg.intercept_ , linreg.coef_

## select null rows:
rowIX = housing[housing['total_bedrooms'].isnull()]
predX = np.array(np.log(rowIX['households'])).reshape(-1,1)
housing.loc[rowIX.index.values, 'total_bedrooms'] = np.exp(linreg.predict(predX))

print ("missing values?", housing.isnull().sum().max()) #just checking that there's no missing data missing...
##2. Using PCA to combine features to 1 principal component that can account for max. variance in the features.
from sklearn.decomposition import PCA
housing_features_pca = PCA(1)
X_select  = housing.loc[:,['total_rooms','total_bedrooms', 'population','households']]
housing['rb/hp']=housing_features_pca.fit_transform(X_select) 
print (housing_features_pca.explained_variance_ratio_) ## 1 Principal Component >> 95% var of data, Good enough.
## New df with combined features: 
housing_redu = housing.drop(columns=['total_rooms', 'total_bedrooms','population', 'households'])

##3. Engineer new "more informative" features
def label_HousingPrice (row):
    if row['median_income'] <= 3 :
        return 'Cheap'
    elif row['median_income'] <=6 :
        return 'Nominal'
    elif (row['median_income'] <=17):
        return 'Expensive'
#   else:
#     return 'VeryExpensive'

housing_redu['PriceCatg'] = housing_redu.apply (lambda row: label_HousingPrice (row),axis=1)
var = 'Expensive'
# housing_ss = housing_redu[['housing_median_age','rb/hp','median_income',\
#                            'PriceCatg','median_house_value']]
housing_ss = housing_redu.loc[housing_redu['PriceCatg'] == var]

# sns.distplot(housing_ss.loc[housing_ss['median_house_value'] < 495e3]['median_house_value'])#, fit=stats.norm);
sns.distplot(housing_ss['median_house_value'])#, fit=stats.norm);
housing_ss['median_house_value'].max()
### Find this "very expensive" neighbourhood (visually)
plt.figure(figsize=(15,10))
plt.scatter(housing_redu['longitude'],housing_redu['latitude'],c=housing_redu['median_house_value'],s=5,cmap='viridis')
expn = housing[housing_redu['PriceCatg'] == 'Expensive']
vexpn = expn[expn['median_house_value']> 495e3]
plt.scatter(vexpn['longitude'],vexpn['latitude'],c='r')
# plt.colorbar()
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.title('House price wrt geo-coordinates')
vexpn.describe()
expn.describe()
from sklearn.neighbors import KernelDensity

def getKDE(var):
  housing_ss = housing_redu[housing_redu['PriceCatg'] == var]
  X = housing_ss[['latitude','longitude']]
  X *= np.pi / 180.  # Convert lat/long to radians
  kde = KernelDensity(bandwidth=0.0001, metric='haversine',
                        kernel='gaussian', algorithm='ball_tree')
  kde.fit(X)
  return kde

column_KDE =[]
for i,var in enumerate(housing_redu['PriceCatg'].unique()):
  print (" generating KDE of PriceCatg: ", var)
  kde = getKDE(var)
  column_KDE.append(var+'_KDE')
  housing_redu[column_KDE[-1]] =  np.exp(kde.score_samples(housing_redu[['latitude','longitude']]* np.pi / 180.))
from sklearn.preprocessing import MinMaxScaler,StandardScaler
###OneHotEncoder: converting ocean_proximity to dummies

housing_onehot=pd.concat([pd.get_dummies(housing_redu['ocean_proximity'],drop_first=True),housing_redu],axis=1).drop('ocean_proximity',axis=1)
housing_onehot = housing_onehot.drop(columns=['PriceCatg',
                                              'latitude','longitude','median_income'])#

from sklearn.model_selection import train_test_split
X = housing_onehot.drop('median_house_value',axis=1)
X[column_KDE] = StandardScaler().fit_transform(X[column_KDE])
y = housing_onehot['median_house_value']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## Transform test and train data & rescale

scalerMM = StandardScaler().fit(X_train)
X_train = scalerMM.transform(X_train)
X_test = scalerMM.transform(X_test)


X.head() ## checking to make sure 
from sklearn.ensemble import RandomForestRegressor

regr = RandomForestRegressor(max_depth=20, random_state=0,n_estimators=100,n_jobs=-1)
regr.fit(X_train, y_train)
scores = cross_val_score(regr, X_train, y_train, cv=10)
print('CV accuracy (original): %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
highest_score = np.mean(scores)

print ("Test score:", regr.score(X_test,y_test))
regr_pred = regr.predict(X_test)
test_mse = np.mean(((regr_pred - y_test)**2))
test_rmse = np.sqrt(test_mse)
print ('final test rmse:', test_rmse) ## to beat (xgb): 41430 >> mine: 35271
plt.figure(figsize=(12,8))
plt.title('Feature Importance')
sns.barplot(data={'importance':regr.feature_importances_,'feature':housing_onehot.columns[housing_onehot.columns!='median_house_value']},y='feature',x='importance')
## Credit: https://www.kaggle.com/pmarcelino/data-analysis-and-feature-extraction-with-python
# Plot learning curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Validation score")

    plt.legend(loc="best")
    return plt

# Plot validation curve
def plot_validation_curve(estimator, title, X, y, param_name, param_range, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    train_scores, test_scores = validation_curve(estimator, X, y, param_name, param_range, cv)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    plt.plot(param_range, train_mean, color='r', marker='o', markersize=5, label='Training score')
    plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.15, color='r')
    plt.plot(param_range, test_mean, color='g', linestyle='--', marker='s', markersize=5, label='Validation score')
    plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha=0.15, color='g')
    plt.grid() 
    plt.xscale('log')
    plt.legend(loc='best') 
    plt.xlabel('Parameter') 
    plt.ylabel('Score') 
    plt.ylim(ylim)


# Plot learning curves
plot_learning_curve(regr, "Toy Model", X_train, 
                    y_train, ylim=(0.3, 1.01), cv=10, n_jobs=-1);
# Plot validation curve ## 
title = 'Validation Curve (Regression)'
param_name = 'n_estimators'
param_range = [500,1000,1500,2000] 
cv = 10
plot_validation_curve(estimator=regr, title=title, X=X_train, y=y_train, 
                      param_name=param_name, ylim=(0.5, 1.01), param_range=param_range);
### This could be considered Engineering a Feature to find Price Category based on geospatial location
def label_HousingPrice (row):
    if row['median_house_value'] <= 150e3 :
        return 'Cheap'
    elif row['median_house_value'] <=300e3 :
        return 'Nominal'
    elif row['median_house_value'] <=400e3 :
        return 'Expensive'
    else:
        return 'VeryExpensive'
housing_play = housing_redu.copy()
housing_play['PriceCatg'] = housing_redu.apply (lambda row: label_HousingPrice (row),axis=1)
def PLAYgetKDE(var):
    housing_ss = housing_play[housing_play['PriceCatg'] == var]
    X = housing_ss[['latitude','longitude']]
    X *= np.pi / 180.  # Convert lat/long to radians
    kde = KernelDensity(bandwidth=0.0001, metric='haversine',
                        kernel='gaussian', algorithm='ball_tree')
    kde.fit(X)
    return kde
PLAYcolumn_KDE =[]
for i,var in enumerate(housing_play['PriceCatg'].unique()):
    print (" generating KDE of PriceCatg: ", var)
    kde = PLAYgetKDE(var)
    PLAYcolumn_KDE.append(var+'_KDE')
    housing_play[PLAYcolumn_KDE[-1]] =  np.exp(kde.score_samples(housing_play[['latitude','longitude']]* np.pi / 180.))
    
### Model
housing_play=pd.concat([pd.get_dummies(housing_play['ocean_proximity'],drop_first=True),housing_play],axis=1).drop('ocean_proximity',axis=1)
housing_play = housing_play.drop(columns=['PriceCatg', 'latitude','longitude'])#
X = housing_play.drop('median_house_value',axis=1)
X[column_KDE] = StandardScaler().fit_transform(X[column_KDE])
y = housing_play['median_house_value']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
## Transform test and train data & rescale
scalerMM = StandardScaler().fit(X_train)
X_train = scalerMM.transform(X_train)
X_test = scalerMM.transform(X_test)

###Fit
regr = RandomForestRegressor(max_depth=20, random_state=0,n_estimators=100,n_jobs=-1)
regr.fit(X_train, y_train)
scores = cross_val_score(regr, X_train, y_train, cv=10)
print('CV accuracy (original): %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
highest_score = np.mean(scores)

print ("Test score:", regr.score(X_test,y_test))
regr_pred = regr.predict(X_test)
test_mse = np.mean(((regr_pred - y_test)**2))
test_rmse = np.sqrt(test_mse)
print ('final test rmse:', test_rmse) ## to beat (xgb): 41430 
plt.figure(figsize=(12,8))
plt.title('Feature Importance')
sns.barplot(data={'importance':regr.feature_importances_,
                  'feature':housing_play.columns[housing_play.columns!='median_house_value']},y='feature',x='importance')
# Plot learning curves
title = "Learning Curves (Regression)"
plot_learning_curve(regr, title, X_train, 
                    y_train, ylim=(0.8, 1.01), cv=10, n_jobs=-1);
