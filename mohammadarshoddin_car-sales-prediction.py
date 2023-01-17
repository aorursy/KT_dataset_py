import pandas as pd 



import numpy as np



import matplotlib.pyplot as plt



%matplotlib inline



import seaborn as sns
df_car = pd.read_csv('../input/cardata/car data.csv')



df_car.head()
print("Shape of dataset ::  ", df_car.shape)
print("Seller Type  ::  ", df_car['Seller_Type'].unique())



print("Fuel Type  ::  ", df_car['Fuel_Type'].unique())



print("Transmission  ::  ", df_car['Transmission'].unique())



print("Owner  ::  ", df_car['Owner'].unique())
df_car.isnull().sum()
df_car.describe()
df_cardata = df_car.drop(columns = 'Car_Name')



df_cardata.head()
df_cardata['Current_Year'] = 2020



df_cardata.head()
df_cardata['No_Year'] = df_cardata['Current_Year'] - df_cardata['Year']



df_cardata.head()
df_cardata = df_cardata.drop(columns = 'Year')



df_cardata.head()
df_cardata = pd.get_dummies(df_cardata, drop_first = True)



df_cardata.head()
df_cardata = df_cardata.drop(['Current_Year'], axis = 1)



df_cardata.head()
df_cardata.corr()
sns.pairplot(df_cardata)



plt.show()
corr_matrix = df_cardata.corr()



top_corr_features = corr_matrix.index



plt.figure(figsize = (20, 20))



sns.heatmap(df_cardata[top_corr_features].corr(), annot = True, cmap = "RdYlGn")



plt.show()
x_data = df_cardata.drop(['Selling_Price'], axis = 1)



x_data.head()
y_data = df_cardata['Selling_Price']



y_data.head()
from sklearn.ensemble import ExtraTreesRegressor



etr_model = ExtraTreesRegressor()



etr_model.fit(x_data, y_data)
print(etr_model.feature_importances_)
feature_importances = pd.Series(etr_model.feature_importances_, index = x_data.columns)



feature_importances.nlargest(5).plot(kind = 'barh')



plt.show()
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.2, random_state = 1)



print("shape of x_train  :: ", x_train.shape)



print("\nshape of x_test  :: ", x_test.shape)



print("\nshape of y_train  :: ", y_train.shape)



print("\nshape of y_test  :: ", y_test.shape)
from sklearn.ensemble import RandomForestRegressor



rfr_model = RandomForestRegressor()
n_estimators = [int(x) for x in np.linspace(100, 1200, 12)]



print("n_estimators :: ", n_estimators)



max_features = ['auto', 'sqrt']



print("\nmax_features  ::  ", max_features)



max_depth = [int(x) for x in np.linspace(5, 30, 6)]



print("\nmax_depth  ::  ", max_depth)



min_samples_split = [2, 5, 10, 15, 100]



print("\nmin_samples_split  ::  ", min_samples_split)



min_samples_leaf = [1, 2, 5, 10]



print("\nmin_samples_leaf  ::  ", min_samples_leaf)
random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf}

random_grid
from sklearn.model_selection import RandomizedSearchCV



rscv_model = RandomizedSearchCV(estimator = rfr_model, param_distributions = random_grid,

                             scoring = 'neg_mean_squared_error', n_iter = 10, cv = 5,

                                verbose = 2, random_state = 42, n_jobs = 1)



rscv_model.fit(x_train, y_train)
rscv_model.best_params_
rscv_model.best_score_
predictions = rscv_model.predict(x_test)
sns.distplot(y_test - predictions)



plt.show()
plt.scatter(y_test, predictions)



plt.show()
from sklearn import metrics



print("Mean Absolute Error , MAE:: ", metrics.mean_absolute_error(y_test, predictions))



print("\nMean Squared Error , MSE :: ", metrics.mean_squared_error(y_test, predictions))



print("\nRMSE :: ", np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print(rscv_model)
import pickle 



file = open('random_forest_regression_model.pkl', 'wb')



pickle.dump(rscv_model, file)