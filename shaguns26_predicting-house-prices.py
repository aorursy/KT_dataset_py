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
house_data = pd.read_csv("/kaggle/input/housesalesprediction/kc_house_data.csv")

house_data.head(10)
house_data.describe()
house_data.info()
house_data.isnull().sum()
sns.set_style('whitegrid')

plt.figure(figsize=(10,4))

sns.distplot(house_data["price"], bins = 50)

plt.xlabel("Price of House")

plt.ylabel("Number of people who bought the house")

plt.title("Distribution of target variable-Price")
sns.set_style('whitegrid')

sns.countplot(house_data['bedrooms'], palette = "pastel").set_title("Number of Bedrooms")
house_data.hist(bins=30, figsize=(11,8), xlabelsize=6, ylabelsize=6, color = "skyblue", lw=0)

plt.tight_layout()

plt.show()

house_data.plot(kind="scatter", x="long", y="lat", alpha=0.1, title = "Geographic Location")
plt.figure(figsize=(4,4))

plt.scatter(x="bedrooms",y="price",data = house_data)

plt.xlabel("Bedrooms")

plt.ylabel("Price")

plt.title("Bedroom vs Price")

plt.tight_layout()

plt.show()
plt.figure(figsize=(4,4))

plt.scatter(x="bathrooms",y="price",data = house_data)

plt.xlabel("Bathrooms")

plt.ylabel("Price")

plt.title("Bathroom vs Price")

plt.tight_layout()

plt.show()
plt.figure(figsize=(4,4))

plt.scatter(x="sqft_living",y="price",data = house_data)

plt.xlabel("Living Area")

plt.ylabel("Price")

plt.title("Living Area vs Price")

plt.tight_layout()

plt.show()
plt.figure(figsize=(4,4))

plt.scatter(x="lat",y="price",data = house_data)

plt.xlabel("Latitude")

plt.ylabel("Price")

plt.title("Location vs Price")

plt.tight_layout()

plt.show()
plt.figure(figsize=(4,4))

plt.scatter(x="long",y="price",data = house_data)

plt.xlabel("Longitude")

plt.ylabel("Price")

plt.title("Location vs Price")

plt.tight_layout()

plt.show()
plt.figure(figsize=(4,4))



ax = house_data.plot(kind="scatter", x="long",y="price", color="b")

house_data.plot(kind="scatter", x="lat",y="price", color="r", ax=ax)





ax.set_xlabel("Location")

ax.set_ylabel("Price")

ax.set_title("Location vs Price")

plt.tight_layout()

plt.show()



plt.figure(figsize=(4,4))

plt.scatter(x="floors",y="price",data = house_data)

plt.xlabel("Floors")

plt.ylabel("Price")

plt.title("Floors vs Price")

plt.tight_layout()

plt.show()
plt.figure(figsize=(4,4))

plt.scatter(x="waterfront",y="price",data = house_data)

plt.xlabel("Waterfront")

plt.ylabel("Price")

plt.title("Waterfront vs Price")

plt.tight_layout()

plt.show()
plt.figure(figsize=(4,4))

plt.scatter(x="view",y="price",data = house_data)

plt.xlabel("View")

plt.ylabel("Price")

plt.title("View vs Price")

plt.tight_layout()

plt.show()
plt.figure(figsize=(4,4))

plt.scatter(x="yr_built",y="price",data = house_data)

plt.xlabel("Year Built")

plt.ylabel("Price")

plt.title("Year Built vs Price")

plt.tight_layout()

plt.show()
conv_dates = [1 if values == 2014 else 0 for values in house_data["date"]]

house_data['date'] = conv_dates

house_data = house_data.drop(['id'],axis=1)
#Split data into Features and Labels

X = house_data.drop("price", axis=1)

y = house_data["price"]
from sklearn.ensemble import RandomForestRegressor



regr = RandomForestRegressor(n_estimators=100)



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3)



regr.fit(X_train,y_train)



rfr = regr.score(X_test,y_test)



rfr
from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor
linear_regression = LinearRegression()

ridge = Ridge()

random_forest_regressor = RandomForestRegressor(n_estimators=100)

gradient_boosting_regressor = GradientBoostingRegressor(n_estimators = 400, max_depth = 5, min_samples_split = 2, learning_rate = 0.1, loss = 'ls')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3)
linear_regression.fit(X_train,y_train)

lr = linear_regression.score(X_test,y_test)

lr
ridge.fit(X_train,y_train)

r = ridge.score(X_test,y_test)

r
random_forest_regressor.fit(X_train,y_train)

rfr = random_forest_regressor.score(X_test,y_test)

rfr
gradient_boosting_regressor.fit(X_train,y_train)

gbr = gradient_boosting_regressor.score(X_test,y_test)

gbr
regression_results = {"LinearRegression" : lr,"Ridge":r,"RandomForestRegressor" : rfr,"GradientBoostingRegressor":gbr}

regression_results
from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score
y_preds = gradient_boosting_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_preds)

mse
mae =  mean_absolute_error(y_test, y_preds)

mae
r2 = r2_score(y_test, y_preds)

r2