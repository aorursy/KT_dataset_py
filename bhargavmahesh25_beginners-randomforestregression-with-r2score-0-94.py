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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
data = pd.read_csv("../input/vehicle-dataset-from-cardekho/car data.csv")
data.head()
data.shape
data.info()
data['Seller_Type'].unique()
data['Fuel_Type'].unique()
data['Transmission'].unique()
data['Owner'].unique()
data.isnull().sum()
data.describe()
data.columns
new_data = data[['Car_Name', 'Year', 'Selling_Price', 'Present_Price', 'Kms_Driven',
       'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]
new_data.shape
new_data['Current_Year'] = 2020
new_data.head()
new_data['Age']=new_data['Current_Year']-new_data['Year']
new_data.head()
new_data.drop(['Year'], axis=1, inplace=True)
new_data.drop(['Current_Year'], axis=1, inplace=True)
new_data.drop(['Car_Name'], axis=1, inplace=True)
new_data.head()
new_data = pd.get_dummies(new_data, drop_first=True)
new_data.head()
correlation = new_data.corr()
correlation
sns.pairplot(new_data)
correlation = new_data.corr()
top_corr = correlation.index
plt.figure(figsize=(20,20))
g = sns.heatmap(correlation, annot=True, cmap='RdYlGn')
correlation.index
new_data.columns
x = new_data.iloc[:,1:]
y = new_data.iloc[:,0]
x.head()
y.head()
# Feature Importance
from sklearn.ensemble import ExtraTreesRegressor
model=ExtraTreesRegressor()
model.fit(x,y)
model.feature_importances_
# Plot the Grpah ofFeature Importances for better visualizaton
feat_importances = pd.Series(model.feature_importances_, index=x.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=50)
x_train.shape
x_test.shape
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor()
## Hyperparameters
n_estimators = [int(x) for x in np.linspace(start=100, stop=1200, num=12)]
print(n_estimators)
# Number of features to consider every split
max_features = ['auto','sqrt']
# Maximum number of levels in the tree
max_depth = [int(x) for x in np.linspace(start = 5, stop = 30, num=6)]
print(max_depth)
# Minimum number of samples required to split the node
min_samples_split = [2,5,10,15,100]
# Minimum number of Samples at each leaf node
min_samples_leaf = [1,2,5,10]
from sklearn.model_selection import RandomizedSearchCV
# Create a Random Grid
random_grid = {'n_estimators': n_estimators,'max_features':max_features, 'max_depth':max_depth, 'min_samples_split':min_samples_split, 'min_samples_leaf':min_samples_leaf}
print(random_grid)
rf_random = RandomizedSearchCV(estimator = rf_model, param_distributions=random_grid, scoring = 'neg_mean_squared_error', n_iter=10, cv=5, verbose=2, random_state=45, n_jobs=1)
rf_random.fit(x_train, y_train)
prediction = rf_random.predict(x_test)
prediction
sns.distplot(y_test - prediction)
plt.scatter(y_test, prediction)
from sklearn import metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
mean_absolute_error = mean_absolute_error(y_test, prediction)
mean_absolute_error
mean_squared_error = mean_squared_error(y_test, prediction)
mean_squared_error
R2_score = r2_score(y_test, prediction)
R2_score
