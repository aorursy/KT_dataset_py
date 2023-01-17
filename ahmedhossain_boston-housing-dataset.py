import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
%matplotlib inline
#Importing Dataset

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df = pd.read_csv('/kaggle/input/boston-house-prices/housing.csv', header=None, delimiter=r"\s+", names= column_names)
df.head()
df.info()    # Checking Data types and missing values
# Another way (Most used) of checking missing data in dataset
df.isnull().sum()
df.drop("CHAS", axis = 1).describe().transpose()
# Correlation Matrix
plt.figure(figsize= (12, 8))
sns.heatmap(df.drop("CHAS", axis = 1).corr(), annot = True)
# Correlation with predictors 
plt.figure(figsize= (10, 6))
correlation = df.drop("CHAS", axis = 1).corr().iloc[0:12,-1]
correlation.plot(kind = "bar")
#Univariate Analysis of MEDV
plt.figure(figsize= (8, 6))
sns.distplot(df["MEDV"])
plt.figure(figsize= (8, 6))
sns.scatterplot(x = df["LSTAT"], y= df["MEDV"])
plt.figure(figsize= (8, 6))
sns.scatterplot(x = df["RM"], y= df["MEDV"])
plt.figure(figsize= (8, 6))
sns.regplot(x = df["TAX"], y= df["RAD"])
plt.figure(figsize= (8, 6))
sns.regplot(x = df["INDUS"], y= df["NOX"])
df["CHAS"].value_counts().plot(kind = "bar")
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
X = df.drop("MEDV", axis = 1)
y = df["MEDV"]
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size = 0.3, random_state = 42)
y_train.hist()
y_test.hist()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train, y_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
rmse = mean_squared_error(y_test, y_pred_lr)**(1/2)
rmse
lr.score(X_test_scaled, y_test)
plt.figure(figsize= (10, 6))
sns.regplot(y_test, y_pred_lr)
plt.xlim([0, 60])
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state = 42)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)
rmse = mean_squared_error(y_test, y_pred_rf)**0.5
rmse
r2_score(y_test, y_pred_rf)
plt.figure(figsize= (10, 6))
sns.regplot(y_test, y_pred_rf)
plt.xlim([0, 60])
from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

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

print(random_grid)
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

# Fit the random search model
rf_random.fit(X_train_scaled, y_train)
# Extract best hyperparameters from 'rf_random'

best_hyperparams = rf_random.best_params_
print('Best hyerparameters:\n', best_hyperparams)
# Extract best model from 'rf_random'
best_model = rf_random.best_estimator_

# Predict the test set labels
y_pred_rf = best_model.predict(X_test_scaled)

# Evaluate the test set RMSE
rmse_test = mean_squared_error(y_test, y_pred_rf)**(1/2)

# Print the test set RMSE
print('Test set RMSE of gb: {:.2f}'.format(rmse_test))
r2_score(y_test, y_pred_rf)
