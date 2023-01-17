import numpy as np # linear algebra

import matplotlib.pyplot as plt

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

import seaborn as sns

from matplotlib.ticker import AutoMinorLocator





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
%matplotlib inline

# Read your file's data solar-system.csv

raw_data = pd.read_csv('/kaggle/input/solarsystem/solar-system.csv', encoding = 'unicode_escape')

raw_data[:50]
# Curve display

plt.plot(raw_data['PuissanceWc'], raw_data['CostTND'], 'ro', markersize=4)

plt.show()
# Clean the data

data = raw_data[raw_data["CostTND"] < 0.8]

# Display data (where CostTND < 0.8)

data
# Curve display

plt.plot(data['PuissanceWc'], data['CostTND'], 'ro', markersize=4)

plt.show()
# We decompose the dataset and we transform it into matrices to be able to perform our calculation

X = np.matrix([np.ones(data.shape[0]), data['PuissanceWc'].values]).T

y = np.matrix(data['CostTND']).T



# We perform the exact calculation of the theta parameter

theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)



print(theta)
# Our final model that sticks well to the data will therefore be in our case (approximately)

# CostTND = 4.92641113e-05 × PuissanceWc + 2.06393337e-01

plt.xlabel('PuissanceWc')

plt.ylabel('CostTND')



plt.plot(data['PuissanceWc'], data['CostTND'], 'ro', markersize=4)



# On affiche la droite entre 0 et 8000

plt.plot([0,8000], [theta.item(0),theta.item(0) + 8000 * theta.item(1)], linestyle='--', c='#000000')



plt.show()