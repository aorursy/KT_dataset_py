# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # plotting

from sklearn.svm import LinearSVR



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("../input/focusing-on-mobile-app-or-website/Ecommerce Customers")

df.head()
df.columns
X = df[['Length of Membership']]

Y = df[['Yearly Amount Spent']]

Y.head()
missing_columns_X = [col for col in X.columns if X[col].isnull().any()]

missing_columns_X
missing_columns_Y = [col for col in Y.columns if Y[col].isnull().any()]

missing_columns_Y
plt.scatter(X, Y, color='blue')

plt.xlabel("Length of Membership")

plt.ylabel("Yearly Amount Spent")

plt.show()
svm_reg = LinearSVR(epsilon=4)

svm_reg.fit(X, Y['Yearly Amount Spent'])
#6 Visualising the Support Vector Regression results

plt.scatter(X, Y, color = 'blue')

plt.plot(X, svm_reg.predict(X), color = 'green')

plt.plot(X, svm_reg.predict(X), color = 'green')



plt.title('Truth or Bluff (Support Vector Regression Model)')

plt.xlabel("Length of Membership")

plt.ylabel("Yearly Amount Spent")

plt.show()
