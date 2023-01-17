import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import scipy.stats as st
df_usa = pd.read_csv("../input/kc_house_data.csv")
print(df_usa.shape)
print(df_usa.nunique())
print(df_usa.info())
df_usa.head()
plt.figure(figsize = (12, 6))

plt.subplot(121)
plt.title('Price Distribuition')
sns.distplot(df_usa['price'])

plt.subplot(122)
g1 = plt.scatter(range(df_usa.shape[0]), np.sort(df_usa.price.values))
g1= plt.title("Price Curve Distribuition", fontsize=15)
g1 = plt.xlabel("")
g1 = plt.ylabel("Amount(US)", fontsize=12)

plt.subplots_adjust(wspace = 0.3, hspace = 0.5,
                    top = 0.9)
plt.show()
print("Price Min")
print(df_usa['price'].min())
print("Price Mean")
print(df_usa['price'].mean())
print("Price Median")
print(df_usa['price'].median())
print("Price Max")
print(df_usa['price'].max())
print("Price Std")
print(df_usa['price'].std())
plt.figure(figsize = (8, 5))
sns.jointplot(df_usa.sqft_living, df_usa.price, 
              alpha = 0.5)
plt.xlabel('Sqft Living')
plt.ylabel('Sale Price')
plt.show()
condition = df_usa['condition'].value_counts()

print("Condition counting: ")
print(condition)

fig, ax = plt.subplots(ncols=2, figsize=(14,5))
sns.countplot(x='condition', data=df_usa, ax=ax[0])
sns.boxplot(x='condition', y= 'price',
            data=df_usa, ax=ax[1])
plt.show()
plt.figure(figsize = (12,8))
g = sns.FacetGrid(data=df_usa, hue='condition',size= 5, aspect=2)
g.map(plt.scatter, "sqft_living", "price")
plt.show()
df_usa["bathrooms"] = df_usa['bathrooms'].round(0).astype(int)

print("Freuency bathroom description:")
print(df_usa["bathrooms"].value_counts())

plt.figure(figsize = (12,8))
plt.subplots_adjust(hspace = 0.4, top = 0.8)

ax1 = plt.subplot(221)
ax1 = sns.countplot(x="bathrooms", data=df_usa,
                    ax=ax1)
ax1.set_xticklabels(ax1.get_xticklabels(),rotation=90)
ax1.set_title("Bathrooms counting", fontsize=15)
ax1.set_xlabel("Bathrooms number")
ax1.set_xlabel("count")

ax2 = plt.subplot(222)
ax2 = sns.boxplot(x="bathrooms", y='price',
                  data=df_usa, ax=ax2)
ax2.set_xticklabels(ax2.get_xticklabels(),rotation=90)
ax2.set_title("Bathrooms distribution price", fontsize=15)
ax2.set_xlabel("Bathrooms number")
ax2.set_ylabel("log Price(US)")

ax0 = plt.subplot(212)
ax0 = sns.stripplot(x="bathrooms", y="price",
                    data=df_usa, alpha=0.5,
                    jitter=True, hue="condition")
ax0.set_title("Better view distribuition through price", fontsize=15)
ax0.set_xlabel("Bathroom number")
ax0.set_ylabel("log Price(US)")
ax0.set_xticklabels(ax0.get_xticklabels(),rotation=90)

plt.show()
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

plt.figure(figsize = (12,6))
ax1 = plt.subplot2grid((2,2), (0,0), colspan = 2)
ax1.set_color_cycle(sns.color_palette('hls', 10))
for val in range(1,6,1):
    indeX = df_usa.condition == val
    ax1.scatter(df_usa.sqft_living.loc[indeX], df_usa.price.loc[indeX], label = val, alpha=0.5)
ax1.legend(bbox_to_anchor = [1.1, 1])
ax1.set_xlabel('sqfit living area')
ax1.set_ylabel('Price house')
ax1.set_title('Sqft Living - Price w.r.t Conditions')

ax2 = plt.subplot2grid((2,2), (1,0))
sns.boxplot(x = 'condition', y = 'price', data = df_usa, ax = ax2)
ax2.set_title('Box Plot Condition & Price', fontsize = 12)

ax3 = plt.subplot2grid((2,2), (1,1))
cubicQual = df_usa.groupby(['condition'])['price'].mean().round(0)
testTrain = df_usa.loc[:, ['condition', 'price']].copy()
testTrain['sqCond'] = np.power(testTrain['condition'],2)
mdl = linear_model.LinearRegression()
mdl.fit(testTrain[['condition', 'sqCond']], testTrain['price'])
y_pred = mdl.predict(testTrain[['condition', 'sqCond']])
print("Mean squared error: %.2f" % mean_squared_error(y_pred, testTrain.price))
# Plot outputs
ax3.scatter(testTrain['condition'], testTrain['price'],  color='black')
ax3.plot(testTrain['condition'], y_pred, color='blue', linewidth=3)
ax3.set_title('LinReg, price ~ condtion + sqft_cond', fontsize = 12)
ax3.set_xlabel('Condition Rate')
plt.subplots_adjust(hspace = 0.5, top = 0.9)
plt.suptitle('Condition Effect to Sale Price', fontsize = 14)
plt.show()
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

plt.figure(figsize = (12,6))
ax1 = plt.subplot2grid((2,2), (0,0), colspan = 2)

for val in range(0,5,1):
    indeX = df_usa.view == val
    ax1.scatter(df_usa.sqft_living.loc[indeX], df_usa.price.loc[indeX], label = val, alpha=0.4)
ax1.legend(bbox_to_anchor = [1.1, 1])
ax1.set_xlabel('sqfit living area')
ax1.set_ylabel('Price house')
ax1.set_title('Sqft Living - Price w.r.t View')

ax2 = plt.subplot2grid((2,2), (1,0))
sns.boxplot(x = 'view', y = 'price', data = df_usa, ax = ax2)
ax2.set_title('Box Plot View & Price', fontsize = 12)

ax3 = plt.subplot2grid((2,2), (1,1))
cubicV = df_usa.groupby(['view'])['price'].mean().round(0)
testTrain = df_usa.loc[:, ['view', 'price']].copy()
testTrain['sqview'] = np.power(testTrain['view'],2)
mdl = linear_model.LinearRegression()
mdl.fit(testTrain[['view', 'sqview']], testTrain['price'])
y_pred = mdl.predict(testTrain[['view', 'sqview']])
print("Mean squared error: %.2f" % mean_squared_error(y_pred, testTrain.price))
# Plot outputs
ax3.scatter(testTrain['view'], testTrain['price'],  color='black')
ax3.plot(testTrain['view'], y_pred, color='blue', linewidth=3)
ax3.set_title('LinReg, price ~ condtion + sqft_cond', fontsize = 12)
ax3.set_xlabel('View rate')
plt.subplots_adjust(hspace = 0.5, top = 0.9)
plt.suptitle('"VIEW" Effect To SalePrice', fontsize = 14)
plt.show()
#How can I color the scatter plot by bedrooms? 
bedrooms = df_usa.bedrooms.value_counts()


plt.figure(figsize = (12,8))
plt.subplots_adjust(hspace = 0.4, top = 0.8)


ax1 = plt.subplot(221)
ax1 = sns.countplot(x="bedrooms", data=df_usa,
                    ax=ax1)
ax1.set_title("bedrooms counting", fontsize=15)
ax1.set_xlabel("Bathrooms number")
ax1.set_ylabel("count")

ax2 = plt.subplot(222)
ax2 = sns.regplot(x="bedrooms", y='price', 
                  data=df_usa, ax=ax2, x_jitter=True)
ax2.set_title("Bedrooms distribution price", fontsize=15)
ax2.set_xlabel("Bedrooms number")
ax2.set_ylabel("log Price(US)")

ax0 = plt.subplot(212)
ax0 = sns.lvplot(x="bedrooms", y="price",
                    data=df_usa)
ax0.set_title("Better understaning price", fontsize=15)
ax0.set_xlabel("Bedrooms")
ax0.set_ylabel("log Price(US)")

plt.show()
print("Floors counting description")
print(df_usa['floors'].value_counts())


plt.figure(figsize = (12,8))
plt.subplots_adjust(hspace = 0.4, top = 0.8)

ax1 = plt.subplot(221)
ax1 = sns.lvplot(x="floors", y='price', 
                    data=df_usa, ax=ax1, )
ax1.set_title("Floors counting", fontsize=15)
ax1.set_xlabel("Floors number")
ax1.set_ylabel("Count")

ax2 = plt.subplot(222)
ax2 = sns.countplot(x="floors",
                  data=df_usa, ax=ax2)
ax2.set_title("Floor distribution by price", fontsize=15)
ax2.set_xlabel("Floor number")
ax2.set_ylabel("log Price(US)")

ax0 = plt.subplot(212)
ax0 = sns.regplot(x="floors", y="price", #I need to change floors by sqft_living and hue bye floors
                    data=df_usa, x_jitter=True)
ax0.set_title("Better understaning price by floor", fontsize=15)
ax0.set_xlabel("Floor")
ax0.set_ylabel("log Price(US)")

plt.show()
plt.figure(figsize = (12,8))
g=sns.lmplot(x="sqft_living", y="price", aspect=1.8,
                    data=df_usa, hue="floors", fit_reg=False)
g.set_titles("Floors by sqft_living and price", fontsize=15)
g.set_xlabels("Sqft Living")
g.set_ylabels("Price(US)")
plt.show()
print("Grade counting description")
print(df_usa['grade'].value_counts())


plt.figure(figsize = (12,8))
plt.subplots_adjust(hspace = 0.4, top = 0.8)

ax1 = plt.subplot(221)
ax1 = sns.lvplot(x="grade", y='price', 
                    data=df_usa, ax=ax1, )
ax1.set_xticklabels(ax1.get_xticklabels(),rotation=90)
ax1.set_title("grade counting", fontsize=15)
ax1.set_xlabel("Grade number")
ax1.set_ylabel("Count")

ax2 = plt.subplot(222)
ax2 = sns.countplot(x="grade",
                  data=df_usa, ax=ax2)
ax2.set_xticklabels(ax2.get_xticklabels(),rotation=90)
ax2.set_title("Grade distribution price", fontsize=15)
ax2.set_xlabel("Grade number")
ax2.set_ylabel("log Price(US)")

ax0 = plt.subplot(212)
ax0 = sns.regplot(x="grade", y="price",
                    data=df_usa, x_jitter=True)
ax0.set_title("Better understaning price by grade", fontsize=15)
ax0.set_xlabel("Grade")
ax0.set_ylabel("log Price(US)")
ax0.set_xticklabels(ax0.get_xticklabels(),rotation=90)

plt.show()
#Clearly view of bathrooms and bedrooms correlation

bath = ['bathrooms', 'bedrooms']
cm = sns.light_palette("green", as_cmap=True)
pd.crosstab(df_usa[bath[0]], df_usa[bath[1]]).style.background_gradient(cmap = cm)

bath_cond = ['bathrooms', 'condition']
cm = sns.light_palette("green", as_cmap=True)
pd.crosstab(df_usa[bath_cond[0]], df_usa[bath_cond[1]]).style.background_gradient(cmap = cm)
bed_cond = ['bedrooms', 'condition']
cm = sns.light_palette("green", as_cmap=True)
pd.crosstab(df_usa[bed_cond[0]], df_usa[bed_cond[1]]).style.background_gradient(cmap = cm)
cond_water = ['condition', 'waterfront']
cm = sns.light_palette("green", as_cmap=True)
pd.crosstab(df_usa[cond_water[0]], df_usa[cond_water[1]]).style.background_gradient(cmap = cm)
grade_cond = ['grade', 'condition']
cm = sns.light_palette("green", as_cmap=True)
pd.crosstab(df_usa[grade_cond[0]], df_usa[grade_cond[1]]).style.background_gradient(cmap = cm)
grade_bed = ['grade', 'bedrooms']
cm = sns.light_palette("green", as_cmap=True)
pd.crosstab(df_usa[grade_bed[0]], df_usa[grade_bed[1]]).style.background_gradient(cmap = cm)
grade_bath = ['grade', 'bathrooms']
cm = sns.light_palette("green", as_cmap=True)
pd.crosstab(df_usa[grade_bath[0]], df_usa[grade_bath[1]]).style.background_gradient(cmap = cm)
corr = df_usa[['bathrooms', 'bedrooms', 'sqft_living', 'sqft_lot', 'floors', 'grade', 'price']]

plt.figure(figsize=(10,8))
plt.title('Correlation of variables')
sns.heatmap(corr.astype(float).corr(),vmax=1.0,  annot=True)
plt.show()
sns.distplot(df_usa['yr_built'])
g = sns.factorplot(x="yr_built", y = "price", data=df_usa[df_usa['price'] < 1000000], 
                   size= 7, aspect = 2, kind="box" )
g.set_xticklabels(rotation=90)
plt.show()
df_usa = df_usa.merge(pd.get_dummies(df_usa.floors, drop_first=True, prefix='Floors'), left_index=True, right_index=True)
df_usa = df_usa.merge(pd.get_dummies(df_usa.waterfront, drop_first=True, prefix='watFront'), left_index=True, right_index=True)
df_usa = df_usa.merge(pd.get_dummies(df_usa.view, drop_first=True, prefix='View'), left_index=True, right_index=True)
df_usa = df_usa.merge(pd.get_dummies(df_usa.condition, drop_first=True, prefix='Cond'), left_index=True, right_index=True)
df_usa = df_usa.merge(pd.get_dummies(df_usa.grade, prefix='Grade'), left_index=True, right_index=True)
df_usa = df_usa.merge(pd.get_dummies(df_usa.bedrooms, drop_first=True, prefix='Bedrooms'), left_index=True, right_index=True)
del df_usa['floors'],
del df_usa['waterfront']
del df_usa['view']
del df_usa['condition']
del df_usa['grade']
del df_usa['bedrooms']


plt.figure(figsize=(15,12))
plt.title('Correlation of variables', fontsize=20)
sns.heatmap(df_usa.corr().astype(float).corr(),vmax=1.0)
plt.show()

from sklearn.model_selection import train_test_split, KFold, cross_val_score # to split the data
from sklearn.metrics import explained_variance_score, median_absolute_error, r2_score, mean_squared_error, accuracy_score, confusion_matrix, classification_report, fbeta_score #To evaluate our model

from sklearn.grid_search import GridSearchCV
########################################################
######## IMPORTING NECESSARY MODULES AND MODELS ########
########################################################

from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import KFold, cross_val_score, train_test_split # Model evaluation
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler # Preprocessing
from sklearn.linear_model import Lasso, Ridge, ElasticNet, RANSACRegressor, SGDRegressor, HuberRegressor, BayesianRidge # Linear models
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor  # Ensemble methods
from xgboost import XGBRegressor, plot_importance # XGBoost
from sklearn.svm import SVR, SVC, LinearSVC  # Support Vector Regression
from sklearn.tree import DecisionTreeRegressor # Decision Tree Regression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline # Streaming pipelines
from sklearn.decomposition import KernelPCA, PCA # Dimensionality reduction
from sklearn.feature_selection import SelectFromModel # Dimensionality reduction
from sklearn.model_selection import learning_curve, validation_curve, GridSearchCV # Model evaluation
from sklearn.base import clone # Clone estimator
from sklearn.metrics import mean_squared_error as MSE
df_usa.drop(['id', 'date'], axis=1, inplace=True)
X = df_usa.drop("price",axis=1).values
y = df_usa["price"].values
# Spliting X and y into train and test version
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=3)
thresh = 5 * 10**(-3)
model = XGBRegressor()
model.fit(X_train, y_train)
#select features using threshold
selection = SelectFromModel(model, threshold=thresh, prefit=True)
select_X_train = selection.transform(X_train)
# eval model
select_X_val = selection.transform(X_test)
# test 
select_X_test = selection.transform(X_test)
pipelines = []
seed = 2

pipelines.append(
                ("Scaled_Ridge", 
                 Pipeline([
                     ("Scaler", StandardScaler()), 
                     ("Ridge", Ridge(random_state=seed, tol=10 ))
                      ]))
                )
pipelines.append(
                ("Scaled_Lasso", 
                 Pipeline([
                     ("Scaler", StandardScaler()), 
                     ("Lasso", Lasso(random_state=seed, tol=1))
                      ]))
                )
pipelines.append(
                ("Scaled_Elastic", 
                 Pipeline([
                     ("Scaler", StandardScaler()), 
                     ("Lasso", ElasticNet(random_state=seed))
                      ]))
                )

pipelines.append(
                ("Scaled_SVR",
                 Pipeline([
                     ("Scaler", StandardScaler()),
                     ("SVR",  SVR(kernel='linear', C=1e2, degree=5))
                 ])
                )
                )

pipelines.append(
                ("Scaled_RF_reg",
                 Pipeline([
                     ("Scaler", StandardScaler()),
                     ("RF", RandomForestRegressor(random_state=seed))
                 ])
                )
                )

pipelines.append(
                ("Scaled_ET_reg",
                 Pipeline([
                     ("Scaler", StandardScaler()),
                     ("ET", ExtraTreesRegressor(random_state=seed))
                 ])
                )
                )
pipelines.append(
                ("Scaled_BR_reg",
                 Pipeline([
                     ("Scaler", StandardScaler()),
                     ("BR", BaggingRegressor(random_state=seed))
                 ]))) 

pipelines.append(
                ("Scaled_Hub-Reg",
                 Pipeline([
                     ("Scaler", StandardScaler()),
                     ("Hub-Reg", HuberRegressor())
                 ]))) 
pipelines.append(
                ("Scaled_BayRidge",
                 Pipeline([
                     ("Scaler", StandardScaler()),
                     ("BR", BayesianRidge())
                 ]))) 

pipelines.append(
                ("Scaled_XGB_reg",
                 Pipeline([
                     ("Scaler", StandardScaler()),
                     ("XGBR", XGBRegressor(seed=seed))
                 ]))) 

pipelines.append(
                ("Scaled_DT_reg",
                 Pipeline([
                     ("Scaler", StandardScaler()),
                     ("DT_reg", DecisionTreeRegressor())
                 ]))) 

pipelines.append(
                ("Scaled_KNN_reg",
                 Pipeline([
                     ("Scaler", StandardScaler()),
                     ("KNN_reg", KNeighborsRegressor())
                 ])))
#pipelines.append(
#                ("Scaled_ADA-Reg",
#                 Pipeline([
#                     ("Scaler", StandardScaler()),
#                     ("ADA-reg", AdaBoostRegressor())
#                 ]))) 

pipelines.append(
                ("Scaled_Gboost-Reg",
                 Pipeline([
                     ("Scaler", StandardScaler()),
                     ("GBoost-Reg", GradientBoostingRegressor())
                 ])))

pipelines.append(
                ("Scaled_RFR_PCA",
                 Pipeline([
                     ("Scaler", StandardScaler()),
                     ("PCA", PCA(n_components=3)),
                     ("XGB", RandomForestRegressor())
                 ])))

pipelines.append(
                ("Scaled_XGBR_PCA",
                 Pipeline([
                     ("Scaler", StandardScaler()),
                     ("PCA", PCA(n_components=3)),
                     ("XGB", XGBRegressor())
                 ])))

#'neg_mean_absolute_error', 'neg_mean_squared_error','r2'
scoring = 'r2'
n_folds = 7

results, names  = [], [] 

for name, model  in pipelines:
    kfold = KFold(n_splits=n_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, y_train, cv= kfold,
                                 scoring=scoring, n_jobs=-1)    
    names.append(name)
    results.append(cv_results)    
    msg = "%s: %f (+/- %f)" % (name, cv_results.mean(),  cv_results.std())
    print(msg)
    
# boxplot algorithm comparison
fig = plt.figure(figsize=(15,6))
fig.suptitle('Algorithm Comparison', fontsize=22)
ax = fig.add_subplot(111)
sns.boxplot(x=names, y=results)
ax.set_xticklabels(names)
ax.set_xlabel("Algorithmn Name", fontsize=20)
ax.set_ylabel("R Squared Score of Models", fontsize=18)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
plt.show()
xgb = XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=7)
xgb.fit(X_train, y_train)
y_hat = xgb.predict(X_test)
xgb.score(X_test,y_test)
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 200, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train, y_train)
#Knowning the best params
rf_random.best_params_
# Predicting with best params
y_hat_Search = rf_random.predict(X_test)
print("XGBoost Regressor R2-score: {}".format(round(r2_score(y_hat, y_test),4)))
print("RandomForest Regressor Prediction R2-score: {}".format(round(r2_score(y_hat_Search, y_test),4)))
print("\nMSE of XGBoost Regressor: {}".format(median_absolute_error(y_hat, y_test)))
print("MSE of RandomForest Regressor: {} ".format(median_absolute_error(y_hat_Search, y_test)))
