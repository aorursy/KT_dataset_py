import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import warnings

warnings.simplefilter('ignore')



from sklearn import datasets

diabetes = datasets.load_diabetes()

df = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)

df.head()
X = df['bmi'].values

Y = diabetes.target



plt.scatter(X, Y);

plt.xlabel('Body mass index (BMI)');

plt.ylabel('Disease progression');
import random

random.seed(0)

idx = random.sample(range(len(df)), 5)

x1, y1 = X[idx], Y[idx]

plt.scatter(x1, y1);
def plot_line(w, b):

    x_values = np.linspace(X.min(), X.max(), 100)

    y_values = w*x_values + b

    plt.plot(x_values, y_values, 'r-')
w = 1300

b = 130

plt.scatter(x1, y1);

plot_line(w, b);
random.seed(12)

idx = random.sample(range(len(df)), 5)

x2, y2 = X[idx], Y[idx]

plt.scatter(x1, y1);

plt.scatter(x2, y2);

plot_line(w, b);
w = 1400

b = 140

plt.scatter(x1, y1);

plt.scatter(x2, y2);

plot_line(w, b);
x = np.concatenate([x1, x2])

y = np.concatenate([y1, y2])

y_pred = w*x + b

error = y - y_pred

pd.DataFrame({'x': x, 'y': y, 'y_pred': y_pred, 

              'error': error})
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
x = x.reshape(-1, 1)

lin_reg.fit(x, y)
w = lin_reg.coef_[0]

b = lin_reg.intercept_

w, b
plt.scatter(x, y);

plot_line(w, b);
X = X.reshape(-1, 1)

lin_reg.fit(X, Y)

w = lin_reg.coef_[0]

b = lin_reg.intercept_

plt.scatter(X, Y);

plot_line(w, b);
lin_reg.score(X, Y)