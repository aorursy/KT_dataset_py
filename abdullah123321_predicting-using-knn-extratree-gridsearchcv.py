
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import ExtraTreesRegressor,GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import validation_curve
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score

DOWNLOAD_ROOT = "../input/kc_house_data.csv"

def create_dataframe(data_path):
    df = pd.read_csv(data_path)
    return df
housing = create_dataframe(DOWNLOAD_ROOT)
housing.head()
housing.info()

print(housing.isnull().any())

with sns.plotting_context("notebook",font_scale=2.5):
    g = sns.pairplot(housing[['sqft_lot','sqft_above','price','sqft_living','bedrooms']], 
                 hue='bedrooms', palette='tab20',size=6)
g.set(xticklabels=[])
%matplotlib inline
import matplotlib.pyplot as plt
housing.hist(bins=10, figsize=(20,15))
plt.show()
import seaborn as sns
corr = housing.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
housing.plot(kind="scatter", x="long", y="lat",alpha=0.1)

#before scaling the data I'm going to deop price col since it is a label then droping date and id since does't make any
housing_scaled = housing.drop('price', axis=1)  # drop labels for training set
housing_scaled = housing_scaled.drop('date', axis=1)  # drop date  for training set
housing_scaled = housing_scaled.drop('id', axis=1)  # drop id  for training set

#extract the labels
housing_labels = housing['price'].copy()
#scale the data with the label
scaler = preprocessing.StandardScaler().fit(housing_scaled)
new_df = scaler.transform(housing_scaled)
housing_scaled = pd.DataFrame(data=new_df, index=list(range(len(new_df))), columns=housing_scaled.columns)
housing_scaled.head()


np.random.seed(0) 
#split the data with raito 0.2
X_train, X_test, y_train, y_test = train_test_split(housing_scaled.values, housing_labels.values, test_size=0.2)

print (X_train.shape)
print (y_train.shape)
print (X_test.shape)
print (y_test.shape)
from sklearn.linear_model import Ridge
ridge_reg=Ridge(alpha=1,solver="cholesky")
ridge_reg.fit(X_train,y_train)


ridge_reg.score(X_test, y_test)


ridge_reg.score(X_train, y_train)


y_pred=ridge_reg.predict(X_test)
r2_score(y_test, y_pred)


from sklearn.neighbors import KNeighborsRegressor
knn_rg = KNeighborsRegressor(n_neighbors=3)
knn_rg.fit(X_train, y_train)
knn_rg.score(X_test, y_test)


knn_rg.score(X_train, y_train)
y_pred=knn_rg.predict(X_test)
r2_score(y_test, y_pred)


from sklearn.linear_model import SGDRegressor
#trying SGD with penalty l2
sgd_reg=SGDRegressor(max_iter=500,penalty='l2',eta0=0.1)
sgd_reg.fit(X_train,y_train.ravel())


sgd_reg.score(X_test, y_test)


sgd_reg.score(X_train, y_train)


y_pred=sgd_reg.predict(X_test)
r2_score(y_test, y_pred)


from sklearn.ensemble import RandomForestRegressor
forest_reg_clf = RandomForestRegressor(random_state=0)
forest_reg_clf.fit(X_train,y_train)


forest_reg_clf.score(X_test, y_test)


forest_reg_clf.score(X_train, y_train)


y_pred=forest_reg_clf.predict(X_test)
r2_score(y_test, y_pred)


param_grid=[
{'n_estimators':[3,10,30,40,50],	'max_features':[10,14,18]},
{'n_estimators':[3,10],'max_features':[14,18]}]
forest_reg=RandomForestRegressor(random_state=0,n_jobs=-1)
rnd_grid_search=GridSearchCV(forest_reg,param_grid,cv=10)
rnd_grid_search.fit(X_train,y_train)


rnd_grid_search.cv_results_['mean_test_score']


rnd_grid_search.cv_results_['mean_train_score']


rnd_grid_search.score(X_test, y_test)


rnd_grid_search.score(X_train, y_train)


y_pred=rnd_grid_search.predict(X_test)
r2_score(y_test, y_pred)


gbrt=GradientBoostingRegressor(max_depth=3,	n_estimators=120)
gbrt.fit(X_train,y_train)
errors=[mean_squared_error(y_test,y_pred)
for y_pred in gbrt.staged_predict(X_test)]
bst_n_estimators=np.argmin(errors)
param_grid=[
{'n_estimators':[bst_n_estimators],'max_features':[2,4,6,8],'max_depth':[1,2,3],'learning_rate':[0.1,0.2,0.5],'random_state':[0]}]

gbrt_best=GradientBoostingRegressor()
gbrt_grid_search=GridSearchCV(gbrt_best,param_grid,cv=5)

gbrt_grid_search.fit(X_train,y_train)


gbrt_grid_search.cv_results_['mean_test_score']


gbrt_grid_search.cv_results_['mean_train_score']


gbrt_grid_search.score(X_test, y_test)


gbrt_grid_search.score(X_train, y_train)


y_pred=gbrt_grid_search.predict(X_test)
r2_score(y_test, y_pred)


param_grid=[
{'n_estimators':[3,10,30,40,50],'max_features':[2,4,6,8]},
{'n_estimators':[3,10],'max_features':[2,3,4]}]
forest_reg_extra=ExtraTreesRegressor()
rnd_grid_search_extra=GridSearchCV(forest_reg_extra,param_grid,cv=10)
rnd_grid_search_extra.fit(X_train,y_train)


rnd_grid_search_extra.cv_results_['mean_test_score']


rnd_grid_search_extra.cv_results_['mean_train_score']


rnd_grid_search_extra.score(X_test, y_test)


rnd_grid_search_extra.score(X_train, y_train)


y_pred=rnd_grid_search_extra.predict(X_test)
r2_score(y_test, y_pred)


# Visualising the results
housing_predictions_ridge_reg=ridge_reg.predict(X_test)
plt.plot(y_test, color = 'red', label = 'house Price')
plt.plot(housing_predictions_ridge_reg, color = 'blue', label = 'Predicted house Price')
plt.title(' house Price Prediction')

plt.legend()
plt.show()


housing_predictions=ridge_reg.predict(X_test)


# housing_predictions=ridge_reg.predict(X_test)
housing_predictions_train=ridge_reg.predict(X_train)
plt.scatter(housing_predictions_train, housing_predictions_train - y_train, c = "blue", marker = "s", label = "Training data")
plt.scatter(housing_predictions, housing_predictions - y_test, c = "lightgreen", marker = "s", label = "Validation data")
plt.title("Linear regression")
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.legend(loc = "upper left")
plt.hlines(y = 0, xmin = 11.5, xmax = 15.5, color = "red")
plt.show()


# Visualising the results
housing_predictions_knn_rg=knn_rg.predict(X_test)
plt.plot(y_test, color = 'red', label = 'house Price')
plt.plot(housing_predictions_knn_rg, color = 'blue', label = 'Predicted house Price')
plt.title(' house Price Prediction')

plt.legend()
plt.show()


gbrt_grid_search.best_estimator_


# Visualising the results
housing_predictions_forest_reg_clf=forest_reg_clf.predict(X_test)
plt.plot(y_test, color = 'red', label = 'house Price')
plt.plot(housing_predictions_forest_reg_clf, color = 'blue', label = 'Predicted house Price')
plt.title(' house Price Prediction')

plt.legend()
plt.show()


#plotting a scatter plot between the real price  values and predict price  values

plt.scatter(housing_predictions_forest_reg_clf, y_test, c='red', alpha=0.5)
plt.show()


# Visualising the results
housing_predictions_rnd_grid_search=rnd_grid_search.predict(X_test)
plt.plot(y_test, color = 'red', label = 'house Price')
plt.plot(housing_predictions_rnd_grid_search, color = 'blue', label = 'Predicted house Price')
plt.title(' house Price Prediction')

plt.legend()
plt.show()


#plotting a scatter plot between the real price  values and predict price  values

plt.scatter(housing_predictions_rnd_grid_search, y_test, c='red', alpha=0.5)
plt.show()


rnd_grid_search.best_estimator_


