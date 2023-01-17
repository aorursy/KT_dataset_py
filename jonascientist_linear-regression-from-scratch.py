import os

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.metrics import mean_absolute_error
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
def return_cleanded_data(data):

  data_clean = data.copy(deep = True)

  data_clean.drop(columns = ["Unnamed: 0", "New_Price"], inplace = True)

  data_clean.dropna("index", inplace = True)

  data_clean = data_clean.reset_index(drop = True)



  names = list(data_clean.Name)

  for i in range(len(names)):

      names[i] = names[i].split(' ', 1)[0]

  data_clean.Name = names

  

  mileage = list(data_clean.Mileage)

  engine = list(data_clean.Engine)

  power = list(data_clean.Power)

  for i in range(len(names)):

      mileage[i] = mileage[i].split(' ', 1)[0]

      engine[i] = engine[i].split(' ', 1)[0]

      power[i] = power[i].split(' ', 1)[0]

  data_clean.Mileage = mileage

  data_clean.Engine = engine

  data_clean.Power = power

  

  data_clean["Price"] = data_clean["Price"].astype(float)

  data_clean["Kilometers_Driven"] = data_clean["Kilometers_Driven"].astype(float)

  data_clean["Mileage"] = data_clean["Mileage"].astype(float)

  data_clean["Engine"] = data_clean["Engine"].astype(float)

  idx = []

  lt = list(data_clean["Power"])

  for i in range(len(lt)):   

      if( lt[i] == "null"):

          idx.append(i)

  data_clean = data_clean.drop(idx)

  data_clean = data_clean.reset_index(drop = True)

  data_clean["Power"] = data_clean["Power"].astype(float)

  data_clean['Year'] = pd.Categorical(data_clean['Year'])

  data_clean['Seats'] = pd.Categorical(data_clean['Seats'])

  data_clean = pd.get_dummies(data_clean, prefix_sep='_', drop_first=True)

  return(data_clean)
data = pd.read_csv("/kaggle/input/used-cars-price-prediction/train-data.csv")

data_clean = return_cleanded_data(data)
y = data_clean[["Price"]].to_numpy()

data_clean = data_clean.drop(columns = ["Price"])



x = data_clean.values

columns = data_clean.columns

min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(x)

data_clean = pd.DataFrame(x_scaled)

data_clean.columns = columns



data_clean.head()
X = data_clean.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.85, test_size=0.15, random_state=1)
def compute_cost(X, weights, y):

  total_cost = 0

  for i in range(X.shape[0]):

    prediction = np.dot(X[i],weights)

    error = y[i][0] - prediction

    total_cost = total_cost + np.power(error,2)

  cost = total_cost/(2*X.shape[0])

  return(cost)
def update_weights(X, weights, learning_rate, weights_past, cost_past, y):

  assert weights.shape[0] ==   X.shape[1], "Unequal shapes"

  suma = np.zeros(X.shape[1])



  for i in range(X.shape[0]):

    for j in range(len(suma)):

      suma[j] = suma[j] + (np.dot(X[i],weights)-y[i][0]) *X[i][j]



  for i in range(weights.shape[0]):

    weights[i] = weights[i] - learning_rate/X.shape[1]*suma[i]



  return(weights)
weights = [1.]*X_train.shape[1]

weights = np.array(weights)

learning_rate = 0.01

weights_past = []

cost_past = []
for j in range(50):

  print(compute_cost(X_train, weights, y_train))

  weights_past.append(weights)

  cost_past.append(compute_cost(X_train, weights, y_train))

  weights = update_weights(X_train, weights, learning_rate, weights_past, cost_past, y_train)
plt.plot(cost_past)

plt.xlabel("Iteration")

plt.ylabel("Loss")

plt.show()
list(zip(weights,list(data_clean.columns)))
prediction = np.dot(X_test, weights)

mean_absolute_error(prediction, y_test)