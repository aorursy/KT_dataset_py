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
data = pd.read_csv('/kaggle/input/california-housing-prices/housing.csv')
data.head()
data.info()
data.describe()
data['ocean_proximity'].value_counts()
data['total_rooms'].value_counts()
import matplotlib.image as mpimg

import matplotlib.pyplot as plt



california_img = mpimg.imread('../input/california-housing-feature-engineering/california.png')

data.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4, s=data['population']/100, label='population', figsize=(10,7), c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True)

plt.imshow(california_img, extent=[-124.55, -113.80, 32.45, 42.05], alpha=0.5)

plt.show()
_ = data.hist(bins=50, figsize=(20,15))
data[data['median_house_value'] >= 500001].count()
data[data['housing_median_age'] >= 51].count()
age_capped = data[data['housing_median_age'] >= 51].index
data.drop(age_capped, axis = 0, inplace = True)
value_capped = data[data['median_house_value'] >= 500001].index

data.drop(value_capped, axis = 0, inplace = True)
data[data['median_house_value'] >= 500001].count()
data[data['housing_median_age'] >= 51].count()
clean_data = data

clean_data['total_bedrooms'].fillna(clean_data['total_bedrooms'].median(), inplace = True)

clean_data.info()
def f_ocean(row):

    if row['ocean_proximity'] == '<1H OCEAN':

        return(1)

    else:

        return(0)

    

def f_inland(row):

    if row['ocean_proximity'] == 'INLAND':

        return(1)

    else:

        return(0)



def f_nbay(row):

    if row['ocean_proximity'] == 'NEAR BAY':

        return(1)

    else:

        return(0)



def f_nocean(row):

    if row['ocean_proximity'] == 'NEAR OCEAN':

        return(1)

    else:

        return(0)

    

clean_data['loc_ocean'] = clean_data.apply(f_ocean, axis = 1)

clean_data['loc_inland'] = clean_data.apply(f_inland, axis = 1)

clean_data['loc_near_bay'] = clean_data.apply(f_nbay, axis = 1)

clean_data['loc_near_ocean'] = clean_data.apply(f_nocean, axis = 1)
clean_data.drop('ocean_proximity', axis='columns', inplace=True)
clean_data.head()
clean_data['total_rooms'] = np.divide(clean_data['total_rooms'], clean_data['households'])

clean_data['total_bedrooms'] = np.divide(clean_data['total_bedrooms'], clean_data['households'])



clean_data.head()
from sklearn.preprocessing import PolynomialFeatures

from pandas import DataFrame



poly = PolynomialFeatures(2)

poly_data = poly.fit_transform(clean_data)

print(poly_data[0])
from sklearn.model_selection import train_test_split



train, test = train_test_split(clean_data, test_size=0.25, random_state=42)
train.info(), test.info()
test.columns
features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms','total_bedrooms', 'households', 'median_income', 'loc_ocean', 'loc_inland', 'loc_near_bay', 'loc_near_ocean']

target = 'median_house_value'



X_train = train[features]

y_train = train[target]



X_test = test[features]

y_test = test[target]
from sklearn.preprocessing import StandardScaler



scaled = StandardScaler()

X_train = scaled.fit_transform(X_train)

X_test = scaled.fit_transform(X_test)
from pandas import DataFrame



X_train = DataFrame(X_train, columns = features)

X_test = DataFrame(X_test, columns = features)

X_train.info()

X_train.head()
from sklearn.linear_model import LinearRegression



model = LinearRegression()
_ = model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(predictions)
print(y_test)
from sklearn.metrics import mean_squared_error



mean_squared_error(y_test, predictions)
from sklearn.metrics import r2_score



r2_score(y_test, predictions)
model.score(X_test, y_test)