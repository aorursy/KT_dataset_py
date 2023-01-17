import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_validation import ShuffleSplit,train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from scipy import stats
from scipy.stats import norm, skew
import folium
from mpl_toolkits.basemap import Basemap
import os
print(os.listdir("../input"))
dc = pd.read_csv("../input/DC_Properties.csv")
dc.head()
dc.info()
print(dc.isnull().sum())
dc_na = (dc.isnull().sum() / len(dc)) * 100
dc_na = dc_na.drop(dc_na[dc_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing %' :dc_na})
missing_data.head(20)
f, ax = plt.subplots(figsize=(15, 12))
sns.barplot(y=dc_na.index, x=dc_na)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)
plt.show()
lat = dc['LATITUDE'].values
lon = dc['LONGITUDE'].values

# 1. Draw the map background
fig = plt.figure(figsize=(8, 8))
m = Basemap(projection='lcc', resolution='l', 
            lat_0=39, lon_0=-78,
            width=1E6, height=1.2E6)
m.shadedrelief()
m.drawcoastlines(color='gray')
m.drawcountries(color='gray')
m.drawstates(color='gray')

# 2. scatter lat and long values

m.scatter(lon, lat, latlon=True,
          cmap='Reds', alpha=0.5)
plt.show()
locations = dc[['LATITUDE', 'LONGITUDE']]
locationlist = locations.values.tolist()
print(len(locationlist))
print(locationlist[1])
map = folium.Map(location=[38.9146833, -77.04076447], zoom_start=12)
folium.Marker(locationlist[1]).add_to(map)
map
# map = folium.Map(location=[38.9146833, -77.04076447], zoom_start=12)
# for point in range(0, len(locationlist)):
#    folium.Marker(locationlist[point]).add_to(map)
# map    
dc_clean=dc
dc_clean.head()
dc_clean=dc_clean.drop(['NATIONALGRID','ASSESSMENT_SUBNBHD','CENSUS_BLOCK','X','Y','QUADRANT'], axis=1)
dc_clean.NUM_UNITS[dc_clean.NUM_UNITS.isnull()] =  dc_clean.NUM_UNITS.mode().iloc[0]
dc_clean.loc[dc_clean['YR_RMDL'].isnull(), 'YR_RMDL'] = dc_clean['EYB']
dc_clean.loc[dc_clean['AYB'].isnull(), 'AYB'] = dc_clean['EYB']
dc_clean.STORIES=dc_clean.STORIES.round()
dc_clean.STORIES[dc_clean.STORIES.isnull()] =  dc_clean.STORIES.mode().iloc[0]
print(dc_clean.GBA.describe())
dc_clean.GBA[dc_clean.GBA.isnull()] =  dc_clean.GBA.mean()
dc_clean.STYLE[dc_clean.STYLE.isnull()] =  dc_clean.STYLE.mode().iloc[0]
dc_clean.STRUCT[dc_clean.STRUCT.isnull()] =  dc_clean.STRUCT.mode().iloc[0]
dc_clean.GRADE[dc_clean.GRADE.isnull()] =  dc_clean.GRADE.mode().iloc[0]
dc_clean.CNDTN[dc_clean.CNDTN.isnull()] =  dc_clean.CNDTN.mode().iloc[0]
dc_clean.EXTWALL[dc_clean.EXTWALL.isnull()] =  dc_clean.EXTWALL.mode().iloc[0]
dc_clean.ROOF[dc_clean.ROOF.isnull()] =  dc_clean.ROOF.mode().iloc[0]
dc_clean.INTWALL[dc_clean.INTWALL.isnull()] =  dc_clean.INTWALL.mode().iloc[0]
dc_clean.KITCHENS[dc_clean.KITCHENS.isnull()] =  dc_clean.KITCHENS.mode().iloc[0]
print(dc_clean.LIVING_GBA.describe())
dc_clean.LIVING_GBA[dc_clean.LIVING_GBA.isnull()] =  dc_clean.LIVING_GBA.median()
dc_clean['CMPLX_NUM1'] = dc_clean['FULLADDRESS'].str[0:4]
dc_clean.loc[dc_clean['CMPLX_NUM'].isnull(), 'CMPLX_NUM'] = dc_clean['CMPLX_NUM1']
dc_clean.CMPLX_NUM[dc_clean.CMPLX_NUM.isnull()] =  dc_clean.CMPLX_NUM.mode().iloc[0]
dc_clean.CITY[dc_clean.CITY.isnull()] =  'WASHINGTON'
dc_clean.STATE[dc_clean.STATE.isnull()] =  'DC'
dc_clean.LATITUDE[dc_clean.LATITUDE.isnull()] =  38.9146833
dc_clean.LONGITUDE[dc_clean.LONGITUDE.isnull()] =  -77.04076447
dc_clean=dc_clean.drop(['FULLADDRESS','CMPLX_NUM1'], axis=1)
dc_clean.head()
dc_clean.info()
unknown_data=dc_clean[dc_clean.PRICE.isnull()]
unknown_data.head()
dc_clean = dc_clean[dc_clean.PRICE.notnull()]
dc_clean.head()
dc_clean.rename(columns={"Unnamed: 0": "id"}, inplace= True)
dc_clean.set_index("id", inplace = True)
dc_clean.head()
unknown_data.rename(columns={"Unnamed: 0": "id"}, inplace= True)
unknown_data.set_index("id", inplace = True)
unknown_data.head()
dc_clean = pd.get_dummies(dc_clean, prefix='HEAT_', columns=['HEAT'])
dc_clean = pd.get_dummies(dc_clean, prefix='AC_', columns=['AC'])
dc_clean = pd.get_dummies(dc_clean, prefix='QUALIFIED_', columns=['QUALIFIED'])
dc_clean = pd.get_dummies(dc_clean, prefix='STYLE_', columns=['STYLE'])
dc_clean = pd.get_dummies(dc_clean, prefix='STRUCT_', columns=['STRUCT'])
dc_clean = pd.get_dummies(dc_clean, prefix='GRADE_', columns=['GRADE'])
dc_clean = pd.get_dummies(dc_clean, prefix='CNDTN_', columns=['CNDTN'])
dc_clean = pd.get_dummies(dc_clean, prefix='EXTWALL_', columns=['EXTWALL'])
dc_clean = pd.get_dummies(dc_clean, prefix='ROOF_', columns=['ROOF'])
dc_clean = pd.get_dummies(dc_clean, prefix='INTWALL_', columns=['INTWALL'])
dc_clean = pd.get_dummies(dc_clean, prefix='SOURCE_', columns=['SOURCE'])
dc_clean = pd.get_dummies(dc_clean, prefix='WARD_', columns=['WARD'])
print(dc_clean.describe())
def stepwise_selection(X, y, 
                       initial_list=[], 
                       threshold_in=0.01, 
                       threshold_out = 0.05, 
                       verbose=True):
    """ Perform a forward-backward feature selection 
    based on p-value from statsmodels.api.OLS
    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features 
    Always set threshold_in < threshold_out to avoid infinite looping.
    """
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.argmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.argmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included
X = dc_clean[['BATHRM', 'HF_BATHRM', 'NUM_UNITS', 'ROOMS', 'BEDRM', 'AYB',
       'YR_RMDL', 'EYB', 'STORIES', 'SALE_NUM',
       'GBA', 'BLDG_NUM', 'KITCHENS', 'FIREPLACES', 'USECODE', 'LANDAREA',
         'LIVING_GBA',
       'ZIPCODE', 'LATITUDE', 'LONGITUDE',
       'CENSUS_TRACT', 'HEAT__Air Exchng', 'HEAT__Air-Oil',
       'HEAT__Elec Base Brd', 'HEAT__Electric Rad', 'HEAT__Evp Cool',
       'HEAT__Forced Air', 'HEAT__Gravity Furnac', 'HEAT__Hot Water Rad',
       'HEAT__Ht Pump', 'HEAT__Ind Unit', 'HEAT__No Data',
       'HEAT__Wall Furnace', 'HEAT__Warm Cool', 'HEAT__Water Base Brd',
       'AC__0', 'AC__N', 'AC__Y', 'QUALIFIED__Q', 'QUALIFIED__U',
       'STYLE__1 Story', 'STYLE__1.5 Story Fin', 'STYLE__1.5 Story Unfin',
       'STYLE__2 Story', 'STYLE__2.5 Story Fin', 'STYLE__2.5 Story Unfin',
       'STYLE__3 Story', 'STYLE__3.5 Story Fin', 'STYLE__3.5 Story Unfin',
       'STYLE__4 Story', 'STYLE__4.5 Story Fin', 'STYLE__4.5 Story Unfin',
       'STYLE__Bi-Level', 'STYLE__Default', 'STYLE__Outbuildings',
       'STYLE__Split Foyer', 'STYLE__Split Level', 'STYLE__Vacant',
       'STRUCT__Default', 'STRUCT__Multi', 'STRUCT__Row End',
       'STRUCT__Row Inside', 'STRUCT__Semi-Detached', 'STRUCT__Single',
       'STRUCT__Town End', 'STRUCT__Town Inside', 'GRADE__Above Average',
       'GRADE__Average', 'GRADE__Excellent', 'GRADE__Exceptional-A',
       'GRADE__Exceptional-B', 'GRADE__Exceptional-C',
       'GRADE__Exceptional-D', 'GRADE__Fair Quality',
       'GRADE__Good Quality', 'GRADE__Low Quality', 'GRADE__No Data',
       'GRADE__Superior', 'GRADE__Very Good', 'CNDTN__Average',
       'CNDTN__Default', 'CNDTN__Excellent', 'CNDTN__Fair', 'CNDTN__Good',
       'CNDTN__Poor', 'CNDTN__Very Good', 'EXTWALL__Adobe',
       'EXTWALL__Aluminum', 'EXTWALL__Brick Veneer',
       'EXTWALL__Brick/Siding', 'EXTWALL__Brick/Stone',
       'EXTWALL__Brick/Stucco', 'EXTWALL__Common Brick',
       'EXTWALL__Concrete', 'EXTWALL__Concrete Block', 'EXTWALL__Default',
       'EXTWALL__Face Brick', 'EXTWALL__Hardboard',
       'EXTWALL__Metal Siding', 'EXTWALL__Plywood', 'EXTWALL__SPlaster',
       'EXTWALL__Shingle', 'EXTWALL__Stone', 'EXTWALL__Stone Veneer',
       'EXTWALL__Stone/Siding', 'EXTWALL__Stone/Stucco',
       'EXTWALL__Stucco', 'EXTWALL__Stucco Block',
       'EXTWALL__Vinyl Siding', 'EXTWALL__Wood Siding', 'ROOF__Built Up',
       'ROOF__Clay Tile', 'ROOF__Comp Shingle', 'ROOF__Composition Ro',
       'ROOF__Concrete', 'ROOF__Concrete Tile', 'ROOF__Metal- Cpr',
       'ROOF__Metal- Pre', 'ROOF__Metal- Sms', 'ROOF__Neopren',
       'ROOF__Shake', 'ROOF__Shingle', 'ROOF__Slate', 'ROOF__Typical',
       'ROOF__Water Proof', 'ROOF__Wood- FS', 'INTWALL__Carpet',
       'INTWALL__Ceramic Tile', 'INTWALL__Default', 'INTWALL__Hardwood',
       'INTWALL__Hardwood/Carp', 'INTWALL__Lt Concrete',
       'INTWALL__Parquet', 'INTWALL__Resiliant', 'INTWALL__Terrazo',
       'INTWALL__Vinyl Comp', 'INTWALL__Vinyl Sheet',
       'INTWALL__Wood Floor', 'SOURCE__Condominium',
       'SOURCE__Residential', 'WARD__Ward 1', 'WARD__Ward 2',
       'WARD__Ward 3', 'WARD__Ward 4', 'WARD__Ward 5', 'WARD__Ward 6',
       'WARD__Ward 7', 'WARD__Ward 8']]
y = dc_clean['PRICE']
result = stepwise_selection(X, y)
print('resulting features:')
print(result)
X = dc_clean[['WARD__Ward 3', 'QUALIFIED__Q', 'HEAT__Ht Pump', 'YR_RMDL', 'QUALIFIED__U', 'EYB', 'LIVING_GBA',
              'STRUCT__Single', 'LATITUDE', 'FIREPLACES', 'GRADE__Very Good', 'GRADE__Good Quality', 'AYB', 
              'GRADE__Excellent', 'WARD__Ward 2', 'CENSUS_TRACT', 'LONGITUDE', 'SALE_NUM', 'WARD__Ward 4',
              'ZIPCODE', 'WARD__Ward 7', 'STYLE__3 Story', 'GRADE__Superior', 'STRUCT__Semi-Detached', 
              'GRADE__Exceptional-C','BATHRM', 'HF_BATHRM', 'NUM_UNITS', 'ROOMS', 'BEDRM']]
y = dc_clean['PRICE']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8 , random_state=100)
corrmat = X_train.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True,cmap = "viridis")
plt.show()
X_train_sm = X_train 
X_train_sm = sm.add_constant(X_train_sm)
lm_sm = sm.OLS(y_train,X_train_sm).fit()
lm_sm.params
lm_sm.summary()
X_test_sm = X_test
X_test_sm = sm.add_constant(X_test_sm)
y_pred_ols = lm_sm.predict(X_test_sm)
y_pred_ols
mse = mean_squared_error(y_test, y_pred_ols)
r_squared = r2_score(y_test, y_pred_ols)
print('Mean Squared Error:' ,mse)
print('r square:',r_squared)
ridge1 = Ridge(alpha = 4, normalize = True)
ridge1.fit(X_train, y_train)              # Fit a ridge regression on the training data
pred_ridge1 = ridge1.predict(X_test)           # Use this model to predict the test data
print(pd.Series(ridge1.coef_, index = X.columns))# Print coefficients
print("MSE for test",mean_squared_error(y_test, pred_ridge1)) 
print("R2 for test",r2_score(y_test,pred_ridge1))
ridge2 = Ridge(alpha = 10**10, normalize = True)
ridge2.fit(X_train, y_train)             # Fit a ridge regression on the training data
pred_ridge2 = ridge2.predict(X_test)           # Use this model to predict the test data
print(pd.Series(ridge2.coef_, index = X.columns)) # Print coefficients
print("MSE for test",mean_squared_error(y_test, pred_ridge2))
print("R2 for test",r2_score(y_test,pred_ridge2))
ridge3 = Ridge(alpha = 0, normalize = True) # alpha = 0 is a linear regression
ridge3.fit(X_train, y_train)             # Fit a ridge regression on the training data
pred_ridge3 = ridge3.predict(X_test)            # Use this model to predict the test data
print(pd.Series(ridge3.coef_, index = X.columns)) # Print coefficients
print("MSE for test",mean_squared_error(y_test, pred_ridge3)) 
print("R2 for test",r2_score(y_test,pred_ridge3))
alphas = 10**np.linspace(10,-2,100)*0.5
print(alphas)

ridgecv = RidgeCV(alphas = alphas, scoring = 'neg_mean_squared_error', normalize = True)
ridgecv.fit(X_train, y_train)
print("Ideal alpha",ridgecv.alpha_)

ridge4 = Ridge(alpha = ridgecv.alpha_, normalize = True)
ridge4.fit(X_train, y_train)
pred_ridge4 = ridge4.predict(X_test)            # Use this model to predict the test data
print(pd.Series(ridge4.coef_, index = X.columns))
print("MSE for test",mean_squared_error(y_test, pred_ridge4))
print("R2 for test",r2_score(y_test,pred_ridge4))
lasso = Lasso(max_iter = 10000, normalize = True)
coefs = []

for a in alphas:
    lasso.set_params(alpha=a)
    lasso.fit(X_train, y_train)
    coefs.append(lasso.coef_)
    
ax = plt.gca()
ax.plot(alphas*2, coefs)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')
lassocv = LassoCV(alphas = None, cv = 10, max_iter = 100000, normalize = True)
lassocv.fit(X_train, y_train)

lasso.set_params(alpha=lassocv.alpha_)
lasso.fit(X_train, y_train)
pred_lasso = lasso.predict(X_test)            # Use this model to predict the test data
print(pd.Series(lasso.coef_, index = X.columns))
print("MSE for test",mean_squared_error(y_test, pred_lasso))
print("R2 for test",r2_score(y_test,pred_lasso))
knn2 = KNeighborsRegressor(n_neighbors=2)
knn2.fit(X_train, y_train)
print("R2 for Train dataset",knn2.score(X_train, y_train))
pred_knn2 = knn2.predict(X_test)
print("MSE for test",mean_squared_error(y_test,pred_knn2))
print("R2 for test",r2_score(y_test,pred_knn2))
knn5 = KNeighborsRegressor(n_neighbors=5)
knn5.fit(X_train, y_train)
print("R2 for Train dataset",knn5.score(X_train, y_train))
pred_knn5 = knn5.predict(X_test)
print("MSE for test",mean_squared_error(y_test,pred_knn5))
print("R2 for test",r2_score(y_test,pred_knn5))

knn3 = KNeighborsRegressor(n_neighbors=3)
knn3.fit(X_train, y_train)
print("R2 for Train dataset",knn3.score(X_train, y_train))
pred_knn3 = knn3.predict(X_test)
print("MSE for test",mean_squared_error(y_test,pred_knn3))
print("R2 for test",r2_score(y_test,pred_knn3))
knn4 = KNeighborsRegressor(n_neighbors=4)
knn4.fit(X_train, y_train)
print("R2 for Train dataset",knn4.score(X_train, y_train))
pred_knn4 = knn4.predict(X_test)
print("MSE for test",mean_squared_error(y_test,pred_knn4))
print("R2 for test",r2_score(y_test,pred_knn4))
#svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
#svr_rbf.fit(X_train,y_train)
#pred = svr_rbf.predict(X_test)
#print("MSE",mean_squared_error(y_test,pred))
#print("R2",r2_score(y_test,pred))
#svr_rbf = SVR(kernel='linear', C=1e3)
#svr_rbf.fit(X_train,y_train)
#pred = svr_rbf.predict(X_test)
#print("MSE",mean_squared_error(y_test,pred))
#print("R2",r2_score(y_test,pred))
regr1 = RandomForestRegressor(random_state=0, n_jobs=-1)
model1 = regr1.fit(X_train, y_train)
model1.score(X_train, y_train)
pred_rf1 = model1.predict(X_test)
print("MSE for test",mean_squared_error(y_test,pred_rf1))
print("R2 for test",r2_score(y_test,pred_rf1))
regr2 = RandomForestRegressor(n_estimators=50,random_state=0, n_jobs=-1)
model2 = regr2.fit(X_train, y_train)
model2.score(X_train, y_train)
pred_rf2 = model2.predict(X_test)
print("MSE for test",mean_squared_error(y_test,pred_rf2))
print("R2 for test",r2_score(y_test,pred_rf2))
regr3 = RandomForestRegressor(n_estimators=100,random_state=0, n_jobs=-1)
model3 = regr3.fit(X_train, y_train)
model3.score(X_train, y_train)
pred_rf3 = model3.predict(X_test)
print("MSE for test",mean_squared_error(y_test,pred_rf3))
print("R2 for test",r2_score(y_test,pred_rf3))
parameters = {
    "n_estimators": [5, 10, 25, 50, 70, 100, 110, 150], # Test out various amounts of trees in the forest
    #"max_features": [0.25] # Test amount of features
}
regr_grid = RandomForestRegressor()
grid_rf = GridSearchCV(regr_grid,parameters)
grid_rf.fit(X_train, y_train)
grid_rf.best_params_

regr4 = RandomForestRegressor(n_estimators=25,random_state=0, n_jobs=-1)
model4 = regr4.fit(X_train, y_train)
model4.score(X_train, y_train)
pred_rf4 = model4.predict(X_test)
print("MSE for test",mean_squared_error(y_test,pred_rf4))
print("R2 for test",r2_score(y_test,pred_rf4))
feature_importances = pd.DataFrame(regr4.feature_importances_,
                                   index = X_train.columns,
                                    columns=['importance']).sort_values('importance',ascending=False)
feature_importances
f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
sns.barplot(x=feature_importances.index, y=feature_importances.importance,palette="deep")
plt.xlabel('Features', fontsize=15)
plt.ylabel('Feature importance', fontsize=15)
plt.title('Feature importance using Random Forest model', fontsize=15)
gbr1=GradientBoostingRegressor(n_estimators=100) 
gbr1.fit(X_train, y_train) 
pred_gbr1=gbr1.predict(X_test)
print("MSE for test",mean_squared_error(y_test,pred_gbr1))
print("R2 for test",r2_score(y_test,pred_gbr1))
print("Train score",gbr1.score(X_train, y_train))
print("Test score",gbr1.score(X_test, y_test))
feature_importances_gbr1 = pd.DataFrame(gbr1.feature_importances_,
                                   index = X_train.columns,
                                    columns=['importance']).sort_values('importance',ascending=False)
feature_importances_gbr1
f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
sns.barplot(x=feature_importances_gbr1.index, y=feature_importances_gbr1.importance,palette="pastel")
plt.xlabel('Features', fontsize=15)
plt.ylabel('Feature importance', fontsize=15)
plt.title('Feature importance using GBR model', fontsize=15)
gbr2 = GradientBoostingRegressor(alpha=0.9, init=None, learning_rate=0.1, loss='ls', max_depth=6, 
                          max_features=1.0, max_leaf_nodes=None, min_samples_leaf=3, 
                          min_samples_split=2, n_estimators=100, random_state=None, subsample=1.0, 
                          verbose=0, warm_start=False) 
gbr2.fit(X_train,y_train)
pred_gbr2=gbr2.predict(X_test)
print("MSE for test",mean_squared_error(y_test,pred_gbr2))
print("R2 for test",r2_score(y_test,pred_gbr2))
print("Train score",gbr2.score(X_train, y_train))
print("Test score",gbr2.score(X_test, y_test))
feature_importances_gbr2 = pd.DataFrame(gbr2.feature_importances_,
                                   index = X_train.columns,
                                    columns=['importance']).sort_values('importance',ascending=False)
feature_importances_gbr2
f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
sns.barplot(x=feature_importances_gbr2.index, y=feature_importances_gbr2.importance,palette="bright")
plt.xlabel('Features', fontsize=15)
plt.ylabel('Feature importance', fontsize=15)
plt.title('Feature importance using GBR model', fontsize=15)
def GradientBooster(param_grid, n_jobs): 
    estimator = GradientBoostingRegressor()
    cv = ShuffleSplit(X_train.shape[0], n_iter=10, test_size=0.2) 
    classifier = GridSearchCV(estimator=estimator, cv=cv, param_grid=param_grid, n_jobs=n_jobs)
    classifier.fit(X_train, y_train)
    print ("Best Estimator learned through GridSearch")
    print (classifier.best_estimator_) 
    return cv, classifier.best_estimator_ 
param_grid={'n_estimators':[100], 'learning_rate': [0.1, 0.05, 0.02, 0.01], 'max_depth':[6,4],
            'min_samples_leaf':[3,5,9,17], 'max_features':[1.0,0.3] } 
n_jobs=4 #Let's fit GBRT to the digits training dataset by calling the function we just created. 
cv,best_est=GradientBooster(param_grid, n_jobs) 
gbr3 = GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
             learning_rate=0.1, loss='ls', max_depth=6, max_features=0.3,
             max_leaf_nodes=None, min_impurity_decrease=0.0,
             min_impurity_split=None, min_samples_leaf=3,
             min_samples_split=2, min_weight_fraction_leaf=0.0,
             n_estimators=100, n_iter_no_change=None, presort='auto',
             random_state=None, subsample=1.0, tol=0.0001,
             validation_fraction=0.1, verbose=0, warm_start=False)
gbr3.fit(X_train,y_train)
pred_gbr3=gbr3.predict(X_test)
print("MSE for test",mean_squared_error(y_test,pred_gbr3))
print("R2 for test",r2_score(y_test,pred_gbr3))
print("Train score",gbr3.score(X_train, y_train))
print("Test score",gbr3.score(X_test, y_test))
feature_importances_gbr3 = pd.DataFrame(gbr3.feature_importances_,
                                   index = X_train.columns,
                                    columns=['importance']).sort_values('importance',ascending=False)
feature_importances_gbr3
f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
sns.barplot(x=feature_importances_gbr3.index, y=feature_importances_gbr3.importance,palette="muted")
plt.xlabel('Features', fontsize=15)
plt.ylabel('Feature importance', fontsize=15)
plt.title('Feature importance using GBR model', fontsize=15)