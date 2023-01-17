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
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

california_img = mpimg.imread('../input/california-housing-feature-engineering/california.png')
data.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4, s=data['population']/100, label='population', figsize=(10,7), c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True)
plt.imshow(california_img, extent=[-124.55, -113.80, 32.45, 42.05], alpha=0.5)
plt.show()
_ = data.hist(bins=50, figsize=(20,15))
data[data['median_house_value'] >= 500001].count()
clean_data = data.dropna()
clean_data.drop('ocean_proximity', axis='columns', inplace=True)
clean_data.head()
from sklearn.model_selection import train_test_split

train, test = train_test_split(clean_data, test_size=0.25, random_state=42)
train.info(), test.info()
test.columns
features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms','total_bedrooms', 'households', 'median_income']
target = 'median_house_value'

X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]
from sklearn.linear_model import LinearRegression

model = LinearRegression()
_ = model.fit(X_train, y_train)
predictions = model.predict(X_test)
from sklearn.metrics import mean_squared_error

mean_squared_error(y_test, predictions)
from sklearn.metrics import r2_score

r2_score(y_test, predictions)
model.score(X_test, y_test)