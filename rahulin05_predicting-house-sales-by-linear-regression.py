import pandas as pd

housing_data = pd.read_csv("../input/kc_house_data.csv")

housing_data.head()
import datetime

current_year = datetime.datetime.now().year

housing_data["age_of_house"] = current_year - pd.to_datetime(housing_data["date"]).dt.year

housing_data.head()
housing_data.info()
housing_data.columns
feature_cols = [ u'age_of_house',  u'bedrooms', u'bathrooms', u'sqft_living',

       u'sqft_lot', u'floors', u'waterfront', u'view', u'condition', u'grade',

       u'sqft_above', u'sqft_basement', u'yr_built', u'yr_renovated']

x = housing_data[feature_cols]

y = housing_data["price"]
import seaborn as sns

%matplotlib inline



sns.pairplot(housing_data,x_vars=feature_cols,y_vars="price",size=7,aspect=0.7,kind = 'reg')
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x, y, random_state=3)
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(x_train, y_train)
accuracy = regressor.score(x_test, y_test)

"Accuracy: {}%".format(int(round(accuracy * 100)))