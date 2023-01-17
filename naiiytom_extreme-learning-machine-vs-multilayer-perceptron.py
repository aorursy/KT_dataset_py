# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.neural_network import MLPRegressor

from sklearn.exceptions import ConvergenceWarning

from sklearn.utils.testing import ignore_warnings



plt.style.use("ggplot")
class MLP(MLPRegressor):

    @ignore_warnings(category=ConvergenceWarning)

    def __init__(self, sizes=(100,), act='relu', max_i=200):

        super().__init__(hidden_layer_sizes=sizes, activation=act, max_iter=max_i)
class ELM(object):

    def __init__(self, hidden_units=200):

        self._hidden_units = hidden_units

        

    def train(self, X, Y):

        X = np.column_stack([X, np.ones([X.shape[0], 1])])

        self.random_weights = np.random.randn(X.shape[1], self._hidden_units)

        G = np.tanh(X.dot(self.random_weights))

        self.w_elm = np.linalg.pinv(G).dot(Y)

        

    def predict(self, X):

        X = np.column_stack([X, np.ones([X.shape[0], 1])])

        G = np.tanh(X.dot(self.random_weights))

        return G.dot(self.w_elm)
path = '../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv'

df = pd.read_csv(path, index_col='id')

df.head()
#drop useless feature

df = df.drop(axis=1, labels=['name', 'host_id', 'host_name', 'neighbourhood', 'latitude', 'longitude', 'last_review', 'reviews_per_month'])

df.head()
#dropping NaN

print(f'DF Shape before dropping NaN: {df.shape}')

df = df.dropna(axis=0)

print(f'DF\'s Shape after dropping NaN {df.shape}')
neighbor = df.groupby('neighbourhood_group').size()

neighbor
neighbor.plot(kind='bar', figsize=(16,9), title='Neighborhood Area')
avg_price = df.groupby('neighbourhood_group').agg('mean').price

avg_price
avg_price.plot(kind='bar', figsize=(16,9), title='Average Price by Neighbourhood')
le = LabelEncoder()

df['neighbourhood_group'] = le.fit_transform(df['neighbourhood_group'])

df['room_type'] = le.fit_transform(df['room_type'])
df.head()
price_div = df.price.max()

min_nights_div = df.minimum_nights.max()

n_review_div = df.number_of_reviews.max()

calcu_div = df.calculated_host_listings_count.max()

availa_div = df.availability_365.max()
df.price = df.price / price_div

df.minimum_nights = df.minimum_nights / min_nights_div

df.number_of_reviews = df.number_of_reviews / n_review_div

df.calculated_host_listings_count = df.calculated_host_listings_count / calcu_div

df.availability_365 = df.availability_365 / availa_div
X = df.drop(axis=1, labels=['price'])

Y = df.price

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=32)
class Helper(object):

    def get_elm(self):

        return self.elm

    

    def get_mlp(self):

        return self.mlp

    

    @ignore_warnings(category=ConvergenceWarning)

    def train_both(self, X, Y, n=100):

        self.iterations = list(range(1, n+1))

        self.elm_mse = []

        self.mlp_mse = []

        for i in self.iterations:

            print(f'{i}.. ', end=' ')

            self.elm = ELM(i)

            self.elm.train(X, Y)

            self.elm_mse.append(np.mean((Y - self.elm.predict(X))**2))

            self.mlp = MLP(sizes=(50,), act='tanh', max_i=i)

            self.mlp.fit(X, Y)

            self.mlp_mse.append(np.mean((Y - self.mlp.predict(X))**2))

        return (self.elm_mse, self.mlp_mse)
misc = Helper()

elm_mse, mlp_mse = misc.train_both(X_train, y_train, 100)
plt.figure(figsize=(16, 9))

plt.title("Extreme Learning Machine vs Multilayer Perceptron Performance Comparison")

plt.plot(elm_mse, '+-', label='Extreme Learning Machine')

plt.plot(mlp_mse, '^-', label='Multilayer Perceptron')

plt.legend()

#plt.yscale('log')

plt.show()
elm = misc.get_elm()

mlp = misc.get_mlp()
print(f'ELM MSE: {np.mean((y_test-elm.predict(X_test))**2)}')

print(f'MLP MSE: {np.mean((y_test-mlp.predict(X_test))**2)}')