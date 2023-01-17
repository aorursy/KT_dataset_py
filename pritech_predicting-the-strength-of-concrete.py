# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import itertools

from sklearn import model_selection

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.utils import resample

from sklearn import metrics

from sklearn.neighbors import KNeighborsClassifier

from scipy.stats import zscore

from sklearn import preprocessing

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.preprocessing import PolynomialFeatures

from sklearn.cluster import KMeans

from sklearn.svm import SVR

from pprint import pprint

from matplotlib import pyplot

import time

from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.ensemble import RandomForestRegressor
df  = pd.read_csv('/kaggle/input/concrete-compressive-strength/concrete.csv')

df.head(10)
rows_count, columns_count = df.shape

print('Total Number of rows :', rows_count)

print('Total Number of columns :', columns_count)

df.info()
sns.heatmap(df.isna(), yticklabels=False, cbar=False, cmap='viridis')
df.apply(lambda x: sum(x.isnull()))
df_transpose = df.describe().T

df_transpose
concrete_df = df.copy()
plt.figure(figsize=(12,6))

sns.boxplot(data=concrete_df, orient="h", palette="Set2", dodge=False)
sns.pairplot(concrete_df,markers="h", diag_kind = 'kde')

plt.show()
fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (13, 5))

fig.set_size_inches(20,6)

sns.distplot(concrete_df['cement'],ax=ax1)

ax1.tick_params(labelsize=15)

ax1.set_xlabel('cement', fontsize=15)

ax1.set_title("Distribution Plot")





sns.boxplot(concrete_df['cement'],ax=ax2)

ax2.set_title("Box Plot")

ax2.set_xlabel('cement', fontsize=15)
fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (13, 5))

fig.set_size_inches(20,6)

sns.distplot(concrete_df['slag'],ax=ax1)

ax1.set_xlabel('Slag', fontsize=15)

ax1.set_title("Distribution Plot")



sns.boxplot(concrete_df['slag'],ax=ax2)

ax2.set_xlabel('Slag', fontsize=15)

ax2.set_title("Box Plot")

fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (13, 5))

fig.set_size_inches(20,6)

sns.distplot(concrete_df['ash'],ax=ax1)

ax1.set_xlabel('Ash', fontsize=15)

ax1.set_title("Distribution Plot")



sns.boxplot(concrete_df['ash'],ax=ax2)

ax2.set_xlabel('Ash', fontsize=15)

ax2.set_title("Box Plot")

fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (13, 5))

fig.set_size_inches(20,6)

sns.distplot(concrete_df['water'],ax=ax1)

ax1.set_xlabel('Water', fontsize=15)

ax1.set_title("Distribution Plot")



sns.boxplot(concrete_df['water'],ax=ax2)

ax2.set_xlabel('Water', fontsize=15)

ax2.set_title("Box Plot")



outlier_columns = []



Q1 =  concrete_df['water'].quantile(0.25) # 1º Quartile

Q3 =  concrete_df['water'].quantile(0.75) # 3º Quartile

IQR = Q3 - Q1                      # Interquartile range



LTV_water = Q1 - 1.5 * IQR   # lower bound 

UTV_water = Q3 + 1.5 * IQR   # upper bound



print('Interquartile range = ', IQR)

print('water <',LTV_water ,'and >',UTV_water, ' are outliers')

print('Numerber of outliers in water column below the lower whisker =', concrete_df[concrete_df['water'] < (Q1-(1.5*IQR))]['water'].count())

print('Numerber of outliers in water column above the upper whisker =', concrete_df[concrete_df['water'] > (Q3+(1.5*IQR))]['water'].count())



# storing column name and upper-lower bound value where outliers are presense 

outlier_columns.append('water')

upperLowerBound_Disct = {'water':UTV_water}
fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (13, 5))

fig.set_size_inches(20,6)

sns.distplot(concrete_df['superplastic'],ax=ax1)

ax1.set_xlabel('superplastic', fontsize=15)

ax1.set_title("Distribution Plot")



sns.boxplot(concrete_df['superplastic'],ax=ax2)

ax2.set_xlabel('Superplastic', fontsize=15)

ax2.set_title("Box Plot")

Q1 =  concrete_df['superplastic'].quantile(0.25) # 1º Quartile

Q3 =  concrete_df['superplastic'].quantile(0.75) # 3º Quartile

IQR = Q3 - Q1                      # Interquartile range



LTV_superplastic = Q1 - 1.5 * IQR   # lower bound 

UTV_superplastic = Q3 + 1.5 * IQR   # upper bound



print('Interquartile range = ', IQR)

print('superplastic <',LTV_superplastic ,'and >',UTV_superplastic, ' are outliers')

print('Numerber of outliers in superplastic column below the lower whisker =', concrete_df[concrete_df['superplastic'] < (Q1-(1.5*IQR))]['superplastic'].count())

print('Numerber of outliers in superplastic column above the upper whisker =', concrete_df[concrete_df['superplastic'] > (Q3+(1.5*IQR))]['superplastic'].count())



# storing column name and upper-lower bound value where outliers are presense

outlier_columns.append('superplastic')

upperLowerBound_Disct['superplastic'] = UTV_superplastic
fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (13, 5))

fig.set_size_inches(20,6)

sns.distplot(concrete_df['coarseagg'],ax=ax1)

ax1.set_xlabel('Coarseagg', fontsize=15)

ax1.set_title("Distribution Plot")



sns.boxplot(concrete_df['coarseagg'],ax=ax2)

ax2.set_xlabel('Coarseagg', fontsize=15)

ax2.set_title("Box Plot")
fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (13, 5))

fig.set_size_inches(20,6)

sns.distplot(concrete_df['fineagg'],ax=ax1)

ax1.set_xlabel('Fineagg', fontsize=15)

ax1.set_title("Distribution Plot")



sns.boxplot(concrete_df['fineagg'],ax=ax2)

ax2.set_xlabel('Fineagg', fontsize=15)

ax2.set_title("Box Plot")

fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (13, 5))

fig.set_size_inches(20,6)

sns.distplot(concrete_df['age'],ax=ax1)

ax1.set_xlabel('Age', fontsize=15)

ax1.set_title("Distribution Plot")



sns.boxplot(concrete_df['age'],ax=ax2)

ax2.set_xlabel('Age', fontsize=15)

ax2.set_title("Box Plot")
Q1 =  concrete_df['age'].quantile(0.25) # 1º Quartile

Q3 =  concrete_df['age'].quantile(0.75) # 3º Quartile

IQR = Q3 - Q1                      # Interquartile range



LTV_age = Q1 - 1.5 * IQR   # lower bound 

UTV_age = Q3 + 1.5 * IQR   # upper bound



print('Interquartile range = ', IQR)

print('age <',LTV_age ,'and >',UTV_age, ' are outliers')

print('Numerber of outliers in age column below the lower whisker =', concrete_df[concrete_df['age'] < (Q1-(1.5*IQR))]['age'].count())

print('Numerber of outliers in age column above the upper whisker =', concrete_df[concrete_df['age'] > (Q3+(1.5*IQR))]['age'].count())



# storing column name and upper-lower bound value where outliers are presense

outlier_columns.append('age')

upperLowerBound_Disct['age'] = UTV_age
fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (13, 5))

fig.set_size_inches(20,6)

sns.distplot(concrete_df['strength'],ax=ax1)

ax1.tick_params(labelsize=15)

ax1.set_xlabel('strength', fontsize=15)

ax1.set_title("Distribution Plot")





sns.boxplot(concrete_df['strength'],ax=ax2)

ax2.set_title("Box Plot")

ax2.set_xlabel('strength', fontsize=15)
print('These are the columns which have outliers : \n\n',outlier_columns)

print('\n\n',upperLowerBound_Disct)
concrete_df_new = concrete_df.copy()
for col_name in concrete_df_new.columns[:-1]:

    q1 = concrete_df_new[col_name].quantile(0.25)

    q3 = concrete_df_new[col_name].quantile(0.75)

    iqr = q3 - q1

    low = q1-1.5*iqr

    high = q3+1.5*iqr

    

    concrete_df_new.loc[(concrete_df_new[col_name] < low) | (concrete_df_new[col_name] > high), col_name] = concrete_df_new[col_name].median()
plt.figure(figsize=(15,8))

sns.boxplot(data=concrete_df_new, orient="h", palette="Set2", dodge=False)
concrete_df_new.shape
concrete_df_new.corr()
mask = np.zeros_like(concrete_df_new.corr(), dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

plt.figure(figsize=(15,7))

plt.title('Correlation of Attributes', y=1.05, size=19)

sns.heatmap(concrete_df_new.corr(),vmin=-1, cmap='plasma',annot=True,  mask=mask, fmt='.2f')
cluster_range = range( 2, 6 )   # expect 3 to four clusters from the pair panel visual inspection hence restricting from 2 to 6

cluster_errors = []

for num_clusters in cluster_range:

  clusters = KMeans( num_clusters, n_init = 5)

  clusters.fit(concrete_df_new)

  labels = clusters.labels_

  centroids = clusters.cluster_centers_

  cluster_errors.append( clusters.inertia_ )

clusters_df = pd.DataFrame( { "num_clusters":cluster_range, "cluster_errors": cluster_errors } )

clusters_df[0:15]
# Elbow plot



plt.figure(figsize=(12,6))

plt.plot( clusters_df.num_clusters, clusters_df.cluster_errors, marker = "o" )
cluster = KMeans( n_clusters = 3, random_state = 2354 )

cluster.fit(concrete_df_new)



prediction=cluster.predict(concrete_df_new)

concrete_df_new["GROUP"] = prediction     # Creating a new column "GROUP" which will hold the cluster id of each record



concrete_df_new_copy = concrete_df_new.copy(deep = True)  # Creating a mirror copy for later re-use instead of building repeatedly
centroids = cluster.cluster_centers_

centroids
# All variables are on same scale, hence we can omit scaling.

# But to standardize the process we will do it here

XScaled = concrete_df_new.apply(zscore)

XScaled.head()
plt.figure(figsize=(12,6))

sns.boxplot(data=XScaled, orient="h", palette="Set2", dodge=False)
y_set = XScaled[['strength']]

X_set = XScaled.drop(labels= "strength" , axis = 1)
y_set = XScaled[['strength']]

X_set = XScaled.drop(labels= "strength" , axis = 1)



# data spliting using 80:20 train test data ratio and randon seeding 7

X_model_train, X_test, y_model_train, y_test = train_test_split(X_set, y_set, test_size=0.20, random_state=7)
print('---------------------- Data----------------------------- \n')

print('x train data {}'.format(X_model_train.shape))

print('y train data {}'.format(y_model_train.shape))

print('x test data  {}'.format(X_test.shape))

print('y test data  {}'.format(y_test.shape))

# data spliting using 70:30 train test data ratio and randon seeding 7

X_train, X_validate, y_train, y_validate = train_test_split(X_model_train, y_model_train, test_size=0.30, random_state=7)
print('---------------------- Data----------------------------- \n')

print('x train data {}'.format(X_train.shape))

print('y train data {}'.format(y_train.shape))

print('x test data  {}'.format(X_validate.shape))

print('y test data  {}'.format(y_validate.shape))

# Defining the kFold function for the cross validation

n_split = 10

randon_state = 7

kfold = KFold(n_split, random_state = randon_state)

linear_model = []

linear_model_score = []

linear_model_RMSE = []

linear_model_R_2 = []

Model = []

RMSE = []

R_sq = []
regression_model = LinearRegression()

regression_model.fit(X_train, y_train)

linear_model.append('Linear Regression')

# coefficients for each of the independent attributes

for idx, col_name in enumerate(X_train.columns):

    print("The coefficient for {} is {}".format(col_name, regression_model.coef_[0][idx]))



intercept = regression_model.intercept_[0]



print("The intercept for our model is {}".format(intercept))



lr_score = regression_model.score(X_validate, y_validate)

linear_model_score.append(lr_score)

print("Linear Regression Model Score:",lr_score)



lr_rmse = np.sqrt((-1) * cross_val_score(regression_model, X_train, y_train.values.ravel(), cv=kfold, scoring='neg_mean_squared_error').mean())

print("Linear Regression Model RMSE :", lr_rmse)



linear_model_RMSE.append(lr_rmse)



lr_r2 = cross_val_score(regression_model, X_train, y_train.values.ravel(), cv=kfold, scoring='r2').mean()

print("Linear Regression Model R-Square Value :",lr_r2)





linear_model_R_2.append(lr_r2)
poly = PolynomialFeatures(degree=2, interaction_only=True)

X_train_ = poly.fit_transform(X_train)

X_test_ = poly.fit_transform(X_validate)

print("Shape", X_train_.shape)

linear_model.append('Polynomial Features - 2D')



poly_clf = LinearRegression()



poly_clf.fit(X_train_, y_train)



pf_score = poly_clf.score(X_test_, y_validate)

print("2D Polynomial Model Score:",pf_score)

linear_model_score.append(pf_score)



pf_rmse = np.sqrt((-1) * cross_val_score(poly_clf, X_train_, y_train.values.ravel(), cv=kfold, scoring='neg_mean_squared_error').mean())

print("2D Polynomial Model RMSE :", pf_rmse)

linear_model_RMSE.append(pf_rmse)



pf_r2 = cross_val_score(poly_clf, X_train_, y_train.values.ravel(), cv=kfold, scoring='r2').mean()

linear_model_R_2.append(pf_r2)

print("2D Polynomial Model R-Square Value :", pf_r2)
poly = PolynomialFeatures(degree=3, interaction_only=True)

X_train__ = poly.fit_transform(X_train)

X_test__ = poly.fit_transform(X_validate)

print("Shape", X_train__.shape)

linear_model.append('Polynomial Features - 3D')



poly_clf_3d = LinearRegression()



poly_clf_3d.fit(X_train__, y_train)



pf3_score = poly_clf_3d.score(X_test__, y_validate)

print("3D Polynomial Model Score:",pf3_score)

linear_model_score.append(pf3_score)



pf3_rmse = np.sqrt((-1) * cross_val_score(poly_clf_3d, X_train__, y_train.values.ravel(), cv=kfold, scoring='neg_mean_squared_error').mean())

print("3D Polynomial Model RMSE :",pf3_rmse)

linear_model_RMSE.append(pf3_rmse)



pf3_r2 = cross_val_score(poly_clf_3d, X_train__, y_train.values.ravel(), cv=kfold, scoring='r2').mean()

print("3D Polynomial Model R-Square Value :",pf3_r2)

linear_model_R_2.append(pf3_r2)
ridge = Ridge(alpha=.3)

ridge.fit(X_train,y_train)

linear_model.append('Ridge - with general data')

print ("Coefficients of the Ridge model",ridge.coef_)

rid_score = ridge.score(X_validate, y_validate)

linear_model_score.append(rid_score)

print("Ridge Model Score:", rid_score)

rig_rmse = np.sqrt((-1) * cross_val_score(ridge, X_train, y_train.values.ravel(), cv=kfold, scoring='neg_mean_squared_error').mean())

print("Ridge Model RMSE :", rig_rmse)

linear_model_RMSE.append(rig_rmse)

rid_r2 = cross_val_score(ridge, X_train, y_train.values.ravel(), cv=kfold, scoring='r2').mean()

print("Ridge Model R-Square Value :",rid_r2)

linear_model_R_2.append(rid_r2)





ridge_pf2 = Ridge(alpha=.3)

ridge_pf2.fit(X_train_,y_train)

linear_model.append('Ridge - with 2d Polynomial features')

print("Coefficients of the Ridge Model - with 2d Polynomial features")

print (ridge_pf2.coef_)

rid_score = ridge_pf2.score(X_test_, y_validate)

linear_model_score.append(rid_score)

print("Ridge Model (2d Polynomial features) Score:",rid_score)

rig_rmse = np.sqrt((-1) * cross_val_score(ridge_pf2, X_train_, y_train.values.ravel(), cv=kfold, scoring='neg_mean_squared_error').mean())

print("Ridge Model (2d Polynomial features) RMSE :", rig_rmse)

linear_model_RMSE.append(rig_rmse)

rid_r2 = cross_val_score(ridge_pf2, X_train_, y_train.values.ravel(), cv=kfold, scoring='r2').mean()

print("Ridge Model (2d Polynomial features) R-Square Value :",rid_r2)

linear_model_R_2.append(rid_r2)
lasso = Lasso(alpha=.3)

lasso.fit(X_train,y_train)

linear_model.append('Lasso - with general data')

print("Coefficients of the Lasso model")

print (lasso.coef_)

lasso_score = lasso.score(X_validate, y_validate)

linear_model_score.append(lasso_score)

print("Lasso Model Score:", lasso_score)

lasso_rmse = np.sqrt((-1) * cross_val_score(lasso, X_train, y_train.values.ravel(), cv=kfold, scoring='neg_mean_squared_error').mean())

print("Lasso Model RMSE :", lasso_rmse)

linear_model_RMSE.append(lasso_rmse)

lasso_r2 = cross_val_score(lasso, X_train, y_train.values.ravel(), cv=kfold, scoring='r2').mean()

print("Lasso Model R-Square Value :",lasso_r2)

linear_model_R_2.append(lasso_r2)





lasso_pf2 = Lasso(alpha=.3)

lasso_pf2.fit(X_train_,y_train)

linear_model.append('Lasso - with 2d Polynomial features')

print("Coefficients of the Lasso Model - with 2d Polynomial features")

print (lasso_pf2.coef_)

lasso_pf2_score = lasso_pf2.score(X_test_, y_validate)

linear_model_score.append(lasso_pf2_score)

print("Lasso Model (2d Polynomial features) Score:",lasso_pf2_score)

lasso_pf2_rmse = np.sqrt((-1) * cross_val_score(lasso_pf2, X_train_, y_train.values.ravel(), cv=kfold, scoring='neg_mean_squared_error').mean())

print("Lasso Model (2d Polynomial features) RMSE :", lasso_pf2_rmse)

linear_model_RMSE.append(lasso_pf2_rmse)

lasso_pf2_r2 = cross_val_score(lasso_pf2, X_train_, y_train.values.ravel(), cv=kfold, scoring='r2').mean()

print("Lasso Model (2d Polynomial features) R-Square Value :", lasso_pf2_r2)

linear_model_R_2.append(lasso_pf2_r2)
compare_lr_model_df = pd.DataFrame({'Model': linear_model,

                           'Score': linear_model_score,

                           'RMSE': linear_model_RMSE,

                           'R Squared': linear_model_R_2})

print("BELOW ARE THE TRAINING SCORES: ")

compare_lr_model_df
compare_lr_model_df[(compare_lr_model_df['RMSE'] == compare_lr_model_df['RMSE'].min()) & (compare_lr_model_df['R Squared'] == compare_lr_model_df['R Squared'].max())]
rfTree = RandomForestRegressor(n_estimators=100)

rfTree.fit(X_train, y_train.values.ravel())

print('Random Forest Regressor')

rfTree_train_score = rfTree.score(X_train, y_train)

print("Random Forest Regressor Model Training Set Score:",rfTree_train_score)





rfTree_score = rfTree.score(X_validate, y_validate)

print("Random Forest Regressor Model Validation Set Score:", rfTree_score)



rfTree_rmse = np.sqrt((-1) * cross_val_score(rfTree, X_train, y_train.values.ravel(), cv=kfold, scoring='neg_mean_squared_error').mean())

print("Random Forest Regressor Model RMSE :", rfTree_rmse)





rfTree_r2 = cross_val_score(rfTree, X_train, y_train.values.ravel(), cv=kfold, scoring='r2').mean()

print("Random Forest Regressor Model R-Square Value :", rfTree_r2)



rfTree_model_df = pd.DataFrame({'Trainng Score': [rfTree_train_score],

                           'Validation Score': [rfTree_score],

                           'RMSE': [rfTree_rmse],

                           'R Squared': [rfTree_r2]})

rfTree_model_df
print("Random Forest Regressor Model Test Data Set Score:")

rfTree_test_score = rfTree.score(X_test, y_test)

print(rfTree_test_score)
rf = RandomForestRegressor(random_state = 7)

print('Parameters currently in use:\n')

pprint(rf.get_params())
# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 10 , stop = 100, num = 3)]   # returns evenly spaced 10 numbers

# Number of features to consider at every split

max_features = ['auto', 'log2']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(5, 10, num = 2)]  # returns evenly spaced numbers can be changed to any

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



pprint(random_grid)
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,

                              n_iter = 5, scoring='neg_mean_absolute_error', 

                              cv = kfold, verbose=2, random_state=7, n_jobs=-1,

                              return_train_score=True)

# Fit the random search model

rf_random.fit(X_train, y_train.values.ravel());
# best ensemble model (with optimal combination of hyperparameters)

rfTree = rf_random.best_estimator_

rfTree.fit(X_train, y_train.values.ravel())

print('Random Forest Regressor')

rfTree_train_score = rfTree.score(X_train, y_train)

print("Random Forest Regressor Model Training Set Score:",rfTree_train_score)



rfTree_score = rfTree.score(X_validate, y_validate)

print("Random Forest Regressor Model Validation Set Score:",rfTree_score)



rfTree_rmse = np.sqrt((-1) * cross_val_score(rfTree, X_train, y_train.values.ravel(), cv=kfold, scoring='neg_mean_squared_error').mean())

print("Random Forest Regressor Model RMSE :", rfTree_rmse)



rfTree_r2 = cross_val_score(rfTree, X_train, y_train.values.ravel(), cv=kfold, scoring='r2').mean()

print("Random Forest Regressor Model R-Square Value :", rfTree_r2)



rfTree_random_model_df = pd.DataFrame({'Trainng Score': [rfTree_train_score],

                           'Validation Score': [rfTree_score],

                           'RMSE': [rfTree_rmse],

                           'R Squared': [rfTree_r2]})

rfTree_random_model_df
rfTree_test_score = rfTree.score(X_test, y_test)

print("Random Forest Regressor Model Test Data Set Score:", rfTree_test_score)
param_grid = {

    'bootstrap': [True],

    'max_depth': [10],

    'max_features': ['log2'],

    'min_samples_leaf': [1, 2, 3],

    'min_samples_split': [5,10],

    'n_estimators': np.arange(50, 71)

}

rfg = RandomForestRegressor(random_state = 7)



grid_search = GridSearchCV(estimator = rfg, param_grid = param_grid, 

                          cv = kfold, n_jobs = 1, verbose = 0, return_train_score=True)



grid_search.fit(X_train, y_train.values.ravel());

grid_search.best_params_
# best ensemble model (with optimal combination of hyperparameters)

rfTree = grid_search.best_estimator_

rfTree.fit(X_train, y_train.values.ravel())

print('Random Forest Regressor')

rfTree_train_score = rfTree.score(X_train, y_train)

print("Random Forest Regressor Model Training Set Score:", rfTree_train_score)



rfTree_score = rfTree.score(X_validate, y_validate)

print("Random Forest Regressor Model Validation Set Score:",rfTree_score)



rfTree_rmse = np.sqrt((-1) * cross_val_score(rfTree, X_train, y_train.values.ravel(), cv=kfold, scoring='neg_mean_squared_error').mean())

print("Random Forest Regressor Model RMSE :", rfTree_rmse)



rfTree_r2 = cross_val_score(rfTree, X_train, y_train.values.ravel(), cv=kfold, scoring='r2').mean()

print("Random Forest Regressor Model R-Square Value :", rfTree_r2)



rfTree_random_model_df = pd.DataFrame({'Trainng Score': [rfTree_train_score],

                           'Validation Score': [rfTree_score],

                           'RMSE': [rfTree_rmse],

                           'R Squared': [rfTree_r2]})

rfTree_random_model_df
def input_scores(name, model, x, y):

    Model.append(name)

    RMSE.append(np.sqrt((-1) * cross_val_score(model, x, y, cv=kfold, 

                                               scoring='neg_mean_squared_error').mean()))

    R_sq.append(cross_val_score(model, x, y, cv=kfold, scoring='r2').mean())

#Comment: Above function uses to append the cross validation scores of the algorithms.
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, 

                              AdaBoostRegressor)



names = ['Linear Regression', 'Ridge Regression',

         'K Neighbors Regressor', 'Decision Tree Regressor', 

         'Random Forest Regressor', 'Gradient Boosting Regressor',

         'Adaboost Regressor', 'Support Vector Regressor']

models = [LinearRegression(), Ridge(),

          KNeighborsRegressor(), DecisionTreeRegressor(),

          RandomForestRegressor(n_estimators=100), GradientBoostingRegressor(), 

          AdaBoostRegressor(), SVR(gamma= "auto")]



#Running all algorithms

for name, model in zip(names, models):

    input_scores(name, model, X_train, y_train.values.ravel())
compare_model_df = pd.DataFrame({'Model': Model,

                           'RMSE': RMSE,

                           'R Squared': R_sq})

print("BELOW ARE THE TRAINING SCORES: ")

compare_model_df
# configure bootstrap

values = XScaled.values







n_iterations = 1000              # Number of bootstrap samples to create

n_size = int(len(XScaled) * 1)    # size of a bootstrap sample



# run bootstrap

stats = list()   # empty list that will hold the scores for each bootstrap iteration



for i in range(n_iterations):



    # prepare train and test sets

    train = resample(values, n_samples=n_size)  # Sampling with replacement

    test = np.array([x for x in values if x.tolist() not in train.tolist()])  # picking rest of the data not considered in sample



    # fit model

    rfTree = RandomForestRegressor(n_estimators=50)  

    rfTree.fit(train[:,:-1], train[:,-1])   # fit against independent variables and corresponding target values



    rfTree.fit(train[:,:-1], train[:,-1])   # fit against independent variables and corresponding target values

    y_test = test[:,-1]    # Take the target column for all rows in test set



    # evaluate model

    predictions = rfTree.predict(test[:, :-1])   # predict based on independent variables in the test data

    score = rfTree.score(test[:, :-1] , y_test)



    stats.append(score)

# plot scores

pyplot.hist(stats)

pyplot.show()

# confidence intervals

alpha = 0.95                             # for 95% confidence 

p = ((1.0-alpha)/2.0) * 100              # tail regions on right and left .25 on each side indicated by P value (border)

lower = max(0.0, np.percentile(stats, p))  

p = (alpha+((1.0-alpha)/2.0)) * 100

upper = min(1.0, np.percentile(stats, p))

print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))
rfTree_random_model_df