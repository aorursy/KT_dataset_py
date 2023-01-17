# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib as ml

import matplotlib.pyplot as plt

%matplotlib inline

ml.style.use('fivethirtyeight')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/iris/Iris.csv')

data.head(60)
data.drop(columns=['Id'],inplace=True)

data
plt.figure(figsize=(20,10))

sns.boxplot(data=data,palette='Set2')

plt.title('Checking for the feature with least outliers')

plt.show()
sns.heatmap(data.corr(),annot=True,linewidth=0.1,linecolor='white')

plt.title("Checking for the most correlated features")

plt.show()
X = data['PetalLengthCm'].values

Y = data['PetalWidthCm'].values

X.shape,Y.shape
sns.scatterplot(x=X,y=Y,hue=data.Species)

plt.xlabel('PetalLengthCm')

plt.ylabel('PetalWidthCm')

plt.title("Raw plotting of the data")

plt.show()
# Mean X and Y

mn_x = np.mean(X)

mn_y = np.mean(Y)

total = len(X)    # Total number of values
# Using the formula to calculate b0 and b1

n = 0

d = 0

for i in range(total):

    n = n + ((X[i] - mn_x) * (Y[i] - mn_y))

    d = d + ((X[i] - mn_x) ** 2)

b1 = n / d                               # B1 = (summation([x[i] - x_mean]*(y[i] - y_mean)))/((summation(x[i] - x_mean))^2)

b0 = mn_y - (b1 * mn_x)                  # B0 = y_mean - (B1*x_mean)



# Print coefficients

print(b1, b0)
# Y = B0 + B1*X

# PetalWidthCm = 325.573421049 + (0.263429339489 * PetalLengthCm)         



# This is our linear model



# Now plotting it to obtain best fit curve



max_x = np.max(X)

min_x = np.min(X)
# Generating unseen data

# Calculating line values x and y

x = np.linspace(min_x, max_x, 1000)

y = b0 + b1 * x
# Plotting



plt.figure(figsize=(10,5))

# Ploting Line based on linear model made from scratch : The unseen data

plt.plot(x, y, color='blue', label='Regression Line')



# Ploting Scatter Points of our actual dataset to see how well the model performs : The predefined data

plt.scatter(X, Y, c='red', label='Scatter Plot')

plt.xlabel('Petal Length in cm')

plt.ylabel('Petal Width in cm')

plt.title("Implentation of Linear Regression from scratch visualized")

plt.legend()

plt.show()