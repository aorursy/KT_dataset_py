import pandas as pd
sales = pd.read_csv('../input/house_sales.csv')
sales.head()
import matplotlib.pyplot as plt

plt.plot(sales['sqft_living'], sales['price'], 'bo')
from sklearn.model_selection import train_test_split

train, test = train_test_split(sales, test_size=0.2, random_state=0)
train.head()
from sklearn import linear_model

sqft_regr = linear_model.LinearRegression()

sqft_model = sqft_regr.fit(train.sqft_living.values.reshape(-1, 1), train.price.values.reshape(-1, 1))
sqft_model.coef_
from sklearn.metrics import mean_squared_error, r2_score

mean_squared_error(test['price'], sqft_model.predict(test['sqft_living'].values.reshape(-1, 1)))
r2_score(test['price'], sqft_model.predict(test['sqft_living'].values.reshape(-1, 1)))
import matplotlib.pyplot as plt

plt.scatter(train['sqft_living'], train['price'],  color='black')

plt.plot(train['sqft_living'], sqft_model.predict(train['sqft_living'].values.reshape(-1, 1)), color='blue', linewidth=3)

plt.xticks(())

plt.yticks(())

plt.show()
my_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode']
my_features_regr = linear_model.LinearRegression()

my_features_model = my_features_regr.fit(train[my_features], train.price.values.reshape(-1, 1))
my_features_model.coef_
mean_squared_error(test['price'], my_features_model.predict(test[my_features]))
r2_score(test['price'], my_features_model.predict(test[my_features]))
house1 = sales[sales.id.isin(['5309101200'])]
house1
print (house1['price'])
sqft_model.predict(house1.sqft_living.values.reshape(1, -1))
my_features_model.predict(house1[my_features])
house2 = sales[sales.id.isin(['1925069082'])]
house2
house2['price']
sqft_model.predict(house2.sqft_living.values.reshape(1, -1))
my_features_model.predict(house2[my_features])