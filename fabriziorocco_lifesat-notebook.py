# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import sys

assert sys.version_info >= (3, 6)

# Scikit-Learn ≥0.21.3 is required

import sklearn

assert sklearn.__version__ >= "0.21.3"

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import sklearn.neighbors

import sklearn.linear_model
# Load the data

lifesat = pd.read_csv('/kaggle/input/lifesat.csv')

lifesat = lifesat.set_index('Country')

lifesat
# Visualize the data

lifesat.plot(kind='scatter', x="GDP per capita", y='Life satisfaction')

plt.tight_layout()

plt.savefig('scatter-plot.png', dpi=600)

plt.show()
# Visualize the data

lifesat.plot(kind='scatter', x="GDP per capita", y='Life satisfaction', figsize=(5,3))

plt.axis([0, 60000, 0, 10])

position_text = {

    "Hungary": (5000, 1),

    "Korea": (15000, 1.7),

    "Italy":(22000,1),

    "France": (29000, 2.4),

    "Australia": (40000, 3.0),

    "United States": (52000, 3.8),

}
items = position_text.items()

for country, pos_text in items:

    pos_data_x, pos_data_y = lifesat.loc[country]

    country = "U.S." if country == "United States" else country

    plt.annotate(country, xy=(pos_data_x, pos_data_y), xytext=pos_text,

            arrowprops=dict(facecolor='black', width=0.5, shrink=0.1, headwidth=5))

    plt.plot(pos_data_x, pos_data_y, "ro")

plt.xlabel("GDP per capita (USD)")

plt.tight_layout()

plt.savefig('scatter-plot-highlight.png', dpi=600)

plt.show()
# Select a linear model

model = sklearn.linear_model.LinearRegression()
X = lifesat[["GDP per capita"]]

y = lifesat["Life satisfaction"]

# Train the model

model.fit(X, y)
# outputs Intercept: 4.853052800266436

print(f"Intercept: {model.intercept_}")

# outputs Coefficients: [4.91154459e-05]

print(f"Coefficients: {model.coef_}")
#visualize again the data (along with the model)

lifesat.plot(kind='scatter', x="GDP per capita", y='Life satisfaction', figsize=(5,3))

plt.xlabel("GDP per capita (USD)")

plt.axis([0, 60000, 0, 10])

Xaxis=np.linspace(0, 60000, 1000)

plt.plot(Xaxis, model.intercept_ + model.coef_[0]*Xaxis, "b")

plt.text(5000, 3.1, r"$\theta_0 = 4.85$", fontsize=14, color="red")

plt.text(5000, 2.2, r"$\theta_1 = 4.91 \times 10^{-5}$", fontsize=14, color="red")

plt.tight_layout()

plt.savefig('scatter-plot-regression-line.png', dpi=600)

plt.show()
# Make a prediction for Cyprus

X_new = [[22587]]  # Cyprus' GDP per capita

print("Prediction (Linear Regression) for Cyprus:")

print(model.predict(X_new)) # it outputs [5.96242338]
# Select a k-Nearest Neighbors Regression Model

k=3

knnModel = sklearn.neighbors.KNeighborsRegressor(

    n_neighbors=k)
# Train the model

knnModel.fit(X, y)
# Make a new prediction for Cyprus using KNN

X_new = [[22587]]  # Cyprus' GDP per capita

print("Prediction (KNN) for Cyprus:")

# if k=1, it outputs [5.7]

# if k=3, it outputs [5.76666667]

print(knnModel.predict(X_new))