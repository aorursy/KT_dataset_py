# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
data_path  = '../input/vehicle-dataset-from-cardekho/car data.csv'

df = pd.read_csv(data_path)
df.head()
df.shape
df.info()
df.describe()
df.duplicated().sum()
df.isnull().sum()
df.drop_duplicates(subset=None, keep='first', inplace=True)
df.shape
print(df['Fuel_Type'].unique())

print(df['Seller_Type'].unique())

print(df['Transmission'].unique())

print(df['Owner'].unique())
import seaborn as sns

sns.heatmap(df.isnull(),yticklabels = False,cbar = False,cmap = 'viridis')
df.columns
final_data = df[['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven',

       'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]
final_data.head()
final_data['current_year'] = 2020
final_data.head()
final_data['Age_Car'] = final_data['current_year'] - final_data['Year']
final_data.head()
final_data.drop(['Year', 'current_year'], axis = 1, inplace = True)
final_data.head()
Dummies = pd.get_dummies(final_data[['Fuel_Type', 'Seller_Type', 'Transmission']], drop_first = True)

final_data = final_data.drop(['Fuel_Type', 'Seller_Type', 'Transmission'], axis = 1)

final_data = pd.concat([final_data, Dummies], axis = 1)
final_data.head()
final_data.corr()
from matplotlib import pyplot as plt

%matplotlib inline
corrmat = final_data.corr()

plt.figure(figsize=(10,10))

#plot heat map

sns.heatmap(corrmat, annot=True, cmap="RdYlGn")
sns.pairplot(final_data)
final_data.columns
plt.figure(figsize =(8,5))

sns.boxplot(x = final_data['Present_Price']);
import numpy as np
data = final_data['Present_Price']
Present_Price_list = final_data['Present_Price'].tolist() 
def detect_outlier(data):

    outlier = []

    threshold = 3

    mean = np.mean(data)

    std = np.std(data)

    for i in data:

        z_score = (i - mean)/std

        if np.abs(z_score)>threshold:

            outlier.append(i)

    return outlier
outlier_pt = detect_outlier(Present_Price_list)
outlier_pt
Kms_Driven_list = final_data['Kms_Driven'].tolist() 
Kms_Driven_list_outlier = detect_outlier(Kms_Driven_list)
Kms_Driven_list_outlier
final_data.drop(final_data[final_data['Present_Price'] > 35.96].index, inplace = True) 
final_data.shape
final_data.drop(final_data[final_data['Kms_Driven'] > 197175].index, inplace = True) 
final_data.shape
final_data.head()
X = final_data.drop(['Selling_Price'], axis = 1)

y = final_data['Selling_Price']
X.head()
y.head()
### Feature Importance



from sklearn.ensemble import ExtraTreesRegressor

model = ExtraTreesRegressor()

model.fit(X,y)
print(model.feature_importances_)
#plot graph of feature importances for better visualization

feat_importances = pd.Series(model.feature_importances_, index=X.columns)

feat_importances.nlargest(5).plot(kind='barh')

plt.show()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()

rf.fit(X_train, y_train)
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]

print(n_estimators)
from sklearn.model_selection import RandomizedSearchCV
#Randomized Search CV



# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(5, 50, num = 10)]

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10, 15, 100]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 5, 10]
# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf}



print(random_grid)
# Use the random grid to search for best hyperparameters

# First create the base model to tune

model_rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 

# search across 100 different combinations

rf_random = RandomizedSearchCV(estimator = model_rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)
rf_random.fit(X_train,y_train)
rf_random.best_params_
rf_random.best_score_
predictions=rf_random.predict(X_test)


sns.distplot(y_test-predictions)
plt.scatter(y_test,predictions)
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))

print('MSE:', metrics.mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
metrics.r2_score(y_test, predictions)