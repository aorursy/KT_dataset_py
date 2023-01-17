import pandas as pd
recipies = pd.read_csv("../input/epirecipes/epi_r.csv")

bikes = pd.read_csv("../input/nyc-east-river-bicycle-crossings/nyc-east-river-bicycle-counts.csv", index_col=0)

weather = pd.read_csv("../input/szeged-weather/weatherHistory.csv")
# quickly clean our dataset

print('#rows in original recipies dataset: {}'.format(recipies.shape[0]))

recipies = recipies[recipies.calories < 10000]

print('#rows in recipies dataset without high calories: {}'.format(recipies.shape[0]))

recipies.dropna(inplace=True)

print('#rows in recipies dataset withouh high calories and NA rows: {}'.format(recipies.shape[0]))
# are the ratings all numeric?

recipies.rating.dtype
#we know it form float64 datatype,

#but let check if there are really some rating with decimal part

print('#rows with decimal part in rating: {}'.format(

    recipies[recipies.rating % 1 > 0].shape[0]))
import seaborn as sns
# plot calories by whether or not it's a dessert

sns.scatterplot(x='calories', y='dessert', data=recipies)
from sklearn.linear_model import LogisticRegression

import numpy as np

import matplotlib.pyplot as plt



%matplotlib inline
model = LogisticRegression()

X =recipies.calories.values.reshape(recipies.calories.count(),1)



model.fit(X, recipies.dessert)

proba = model.predict_proba(X)
plt.figure(figsize=(10,4))

sns.scatterplot(x='calories', y='dessert', data=recipies)

plt.plot(X, proba[:,1])
bikes.head(5)
df = bikes[['Precipitation', 'Total']].copy()

X = bikes.Precipitation

y = bikes.Total



def print_bad_values(df):

    print(df[df.Precipitation.map(lambda x : x.replace('.', '', 1).isdigit() == False)].Precipitation)

    

print_bad_values(df)
df.Precipitation = df.Precipitation.map(lambda x : '0.0' if x == 'T' else x)

df.Precipitation = df.Precipitation.map(lambda x : x.split()[0] if 'S' in x  else x)

print_bad_values(df)
df.Precipitation = df.Precipitation.map(lambda x : float(x))

print('min {}, max {}'.format(df.Precipitation.min(), df.Precipitation.max()))
df.info()
import sklearn

print('The scikit-learn version is {}.'.format(sklearn.__version__))
sns.scatterplot(x=df.Precipitation, y=df.Total)
from sklearn.linear_model import PoissonRegressor #for sklearn from version 0.23 #
#I would actualy use plotting of model prediction as I play around with Python not R

X = df.Precipitation.values.reshape(df.Precipitation.count(), 1)

y = df.Total



model = PoissonRegressor()

model.fit(X, y)

df['Preds'] = model.predict(X)
df.Preds = df.Preds.map(lambda x : int(x))

df.head()
plt.figure(figsize=(10, 6))

sns.scatterplot(x='Precipitation', y='Total', data=df)# fit_reg=False)

sns.lineplot(x='Precipitation', y='Preds', data=df)