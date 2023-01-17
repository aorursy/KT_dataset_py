import pandas as pd

from matplotlib import pyplot as plt

import seaborn as sns
data = pd.read_csv('../input/kc_house_data.csv')
data.shape
data.describe()
data['date'] = pd.to_datetime(data['date'])
data.head()
data['bedrooms'].value_counts()
data['bedrooms'].value_counts().plot(kind='bar')

plt.xlabel('Number of Bedrooms')

plt.ylabel('Count')
# Houses having 3 and 4 bedrooms are sold most
plt.scatter(data.sqft_living,data.price)

plt.title("Price vs Square Feet")

plt.xlabel("Square Feet")

plt.ylabel("Price")
plt.scatter(data.zipcode, data.price)

plt.xlabel("Price")

plt.ylabel("Zip")
data.groupby(['bedrooms', 'bathrooms']).size()
plt.scatter(data.bedrooms,data.bathrooms)

plt.title("Bedrooms vs Bathrooms")

plt.xlabel("Bedrooms")

plt.ylabel("Bathrooms")
#Hence we concluded that bathrooms is diretly proportional to bedrooms and bedrooms to size of the house
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
#As we want to predict Price of the house so we will set labels as the Price column
labels = data['price']
conv_dates = [1 if values == 2014 else 0 for values in data.date]

data['date'] = conv_dates

train1 = data.drop(['id','price'],axis=1)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(train1, labels,test_size = 0.2,random_state = 2)
reg.fit(x_train,y_train)
reg.score(x_test,y_test)