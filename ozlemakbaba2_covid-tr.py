

import numpy as np 

import pandas as pd 



data = pd.read_csv("../input/covid19-in-turkey/covid_19_data_tr.csv")

data.head()
data.dtypes.value_counts()
data.head().T
data.drop(['Province/State', 'Country/Region', 'Last_Update'], axis=1, inplace=True)
data.columns
data.head()

data.dtypes.value_counts()
from sklearn.preprocessing import LabelBinarizer



lb = LabelBinarizer()



for col in ['Confirmed', 'Deaths', 'Recovered']:

    data[col] = lb.fit_transform(data[col])


x_cols = [x for x in data.columns if x != 'Confirmed']



# Verileri iki dataframe'e bolme

X_data = data[x_cols]

y_data = data['Confirmed']

from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors=3)



knn = knn.fit(X_data, y_data)



y_pred = knn.predict(X_data)



def accuracy(real, predict):

    return sum(y_data == y_pred) / float(real.shape[0])



print(accuracy(y_data, y_pred))
from math import sqrt





knn = KNeighborsClassifier(n_neighbors=3, p=1, metric='euclidean')



knn = knn.fit(X_data, y_data)



y_pred = knn.predict(X_data)



kare=sqrt(accuracy (y_data, y_pred))

           

print(kare)

           


knn = KNeighborsClassifier(n_neighbors=2, p=2, metric='minkowski')

knn = knn.fit(X_data, y_data)

y_pred = knn.predict(X_data)

kare_1 =sqrt(accuracy (y_data, y_pred))

print(kare_1)
y = (data['Confirmed']).astype(int)

fields = list(data.columns[:-1])

correlations = data[fields].corrwith(y)

correlations.sort_values(inplace=True)

correlations
import warnings

warnings.filterwarnings("ignore")



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



sns.set_context('talk')

sns.set_palette('dark')



sns.kdeplot(correlations, cumulative=True)