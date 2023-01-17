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
# Load car data (from cardekho)

car_data = pd.read_csv('../input/vehicle-dataset-from-cardekho/CAR DETAILS FROM CAR DEKHO.csv')

car_data.head()
car_data.info()
car_data.shape
from datetime import datetime



car_data['no_years'] = datetime.now().year - car_data['year']
car_data.columns
sns.heatmap(car_data.corr(), annot=True);
car_data['fuel'].value_counts()
car_data['fuel'].value_counts().plot(kind='barh');
car_data['seller_type'].value_counts()
car_data['seller_type'].value_counts().plot(kind='barh');
car_data['transmission'].value_counts()
car_data['transmission'].value_counts().plot(kind='barh');
car_data['owner'].value_counts()
car_data.shape
one_hot_encoded = pd.get_dummies(car_data[['fuel', 'seller_type', 'transmission']])

one_hot_encoded
def prev_owner_counter(owner):

    """

    Convert the english

    """

    if owner == "First Owner":

        return 0

    elif owner == "Second Owner":

        return 1

    elif owner == "Third Owner":

        return 2

    elif owner == "Fourth & Above Owner":

        return 4

    elif owner == "Test Drive Car":

        return -1
car_data['prev_owners'] = car_data['owner'].apply(lambda owner: prev_owner_counter(owner))
car_data.head()
car_data = car_data.join(one_hot_encoded)
car_data.tail()
car_data.shape
car_data.columns
final_df = car_data[['selling_price', 'km_driven', 'no_years', 'prev_owners', 'fuel_CNG',

                     'fuel_Diesel', 'fuel_Electric', 'fuel_LPG', 'fuel_Petrol',

                     'seller_type_Dealer', 'seller_type_Individual',

                     'seller_type_Trustmark Dealer', 'transmission_Automatic', 'transmission_Manual'

                    ]]

final_df.head()
plt.figure(figsize=(20, 20))

sns.heatmap(data=final_df.corr(), annot=True);
corr = final_df[final_df.columns[1:]].corrwith(final_df['selling_price'])

corr
X = final_df[final_df.columns[1:]]

y = final_df[['selling_price']]



X.head(), y.head()
sns.pairplot(data=final_df)
from sklearn.model_selection import train_test_split



np.random.seed = 101

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
from sklearn.ensemble import ExtraTreesRegressor



model = ExtraTreesRegressor()

model.fit(X, y)
model.feature_importances_
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import RandomizedSearchCV
#Randomized Search CV



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
# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf}



print(random_grid)
rf_random = RandomizedSearchCV(estimator = RandomForestRegressor(), param_distributions = random_grid,scoring='neg_mean_squared_error',

                               n_iter = 10, cv = 5, verbose=2,  n_jobs = 1)

rf_random.fit(X_train, y_train)
rf_random.best_params_
rf_random.best_score_
y_predicted = rf_random.predict(X_test)

y_predicted
fig, ax = plt.subplots()

ax.scatter(y_predicted, y_test, edgecolors=(0, 0, 1))

ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=3)

ax.set_xlabel('Predicted')

ax.set_ylabel('Actual')

plt.show()
from sklearn import metrics

mae = mean_absolute_error(y_test, y_predicted)

mse = mean_squared_error(y_test, y_predicted)

r2 = r2_score(y_test, y_predicted)



print("The model performance for testing set")

print("--------------------------------------")

print('MAE:', metrics.mean_absolute_error(y_test, y_predicted))

print('MSE:', metrics.mean_squared_error(y_test, y_predicted))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_predicted)))