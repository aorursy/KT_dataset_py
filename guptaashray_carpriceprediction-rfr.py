import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error,mean_squared_error

# Loading the dataset
df = pd.read_csv("/kaggle/input/vehicle-dataset-from-cardekho/CAR DETAILS FROM CAR DEKHO.csv")
df.shape
df.dtypes
final_df = df[['year', 'selling_price', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner']]
final_df['owner'].value_counts()
final_df.drop(final_df[final_df['owner']=='Test Drive Car'].index,axis=0,inplace=True)
final_df['No_of_previous_owner'] = final_df['owner'].map({'First Owner':1,'Second Owner':2,'Third Owner':3,"Fourth & Above":4})
final_df.head()
final_df.drop(final_df[final_df['seller_type']=='Trustmark Dealer'].index,axis=0,inplace=True)
final_df['Current Year'] = 2020
final_df['No_of_Years'] = final_df['Current Year'] - final_df['year']
final_df.drop(['year','Current Year'],axis=1,inplace=True)
# percentage of missing values in each column
round(100*(final_df.isnull().sum()/len(final_df)),2).sort_values(ascending = False)
# percentage of missing values in each row
round(100*(final_df.isnull().sum(axis=1)/len(final_df)),2).sort_values(ascending = False)
final_df_dup=final_df.copy()
# Checking for duplicates and dropping the entire duplicate row if any
final_df_dup.drop_duplicates(subset=None, inplace=True)
final_df.shape
final_df_dup.shape
final_df=final_df_dup
final_df.head()
final_df = pd.get_dummies(final_df,drop_first=True)
final_df.head()
# Seprating the dependent variable and target variable
y = final_df['selling_price']
X = final_df.drop('selling_price',axis=1)
# Splitting training and testing data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=5)
# Building a machine learning model using Random Forest Regressor
regressor = RandomForestRegressor()
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)
# search across 100 different combinations
rf_random = RandomizedSearchCV(estimator = regressor, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = 1)
final_df.isnull().sum()
# Training the data
X_train = X_train.fillna(X_train.mean())
y_train = y_train.fillna(y_train.mean())
X_test = X_test.fillna(X_test.mean())
y_test = y_test.fillna(y_test.mean())
rf_random.fit(X_train,y_train)
rf_random.best_params_
y_pred = rf_random.predict(X_test)
errors = abs(y_pred - y_test)
mape = 100 * np.mean(errors / y_test)
accuracy = 100 - mape
print('Model Performance')
print('MAE:',mean_absolute_error(y_test,y_pred))
print('MSE:', mean_squared_error(y_test,y_pred))
print('RMSE:', np.sqrt(mean_squared_error(y_test,y_pred)))
print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
print('Accuracy = {:0.2f}%.'.format(accuracy))
    
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('MAE:',mean_absolute_error(y_test,y_pred))
    print('MSE:', mean_squared_error(y_test,y_pred))
    print('RMSE:', np.sqrt(mean_squared_error(y_test,y_pred)))
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy
base_model = RandomForestRegressor(n_estimators = 10, random_state = 42)
base_model.fit(X_train, y_train)
base_accuracy = evaluate(base_model, X_test, y_test)
best_random = rf_random.best_estimator_
random_accuracy = evaluate(best_random, X_test, y_test)
print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))
from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}
# Create a based model
rf = RandomForestRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)
# Fit the grid search to the data
grid_search.fit(X_train, y_train)
grid_search.best_params_
best_grid = grid_search.best_estimator_
grid_accuracy = evaluate(best_grid, X_test, y_test)
print('Improvement of {:0.2f}%.'.format( 100 * (grid_accuracy - base_accuracy) / base_accuracy))
# Create a new dataframe of only numeric variables:

car_n=final_df[[ 'selling_price', 'km_driven', 'No_of_Years']]

sns.pairplot(car_n, diag_kind='kde')
plt.show()
# Let's check the correlation coefficients to see which variables are highly correlated. Note:
# here we are considering only those variables (dataframe: car) that were chosen for analysis

plt.figure(figsize = (25,20))
sns.heatmap(final_df.corr(), annot = True, cmap="RdBu")
plt.show()
