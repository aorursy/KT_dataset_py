# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Import packages

import sklearn

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv("/kaggle/input/insurance/insurance.csv")
df.head()

#Replace string type with integer type


df.sex[df.sex == 'male'] = 0

df.sex[df.sex == 'female'] = 1

df.smoker[df.smoker == 'no'] = 0

df.smoker[df.smoker == 'yes'] = 1

df.region[df.region == 'northwest'] = 0

df.region[df.region == 'southwest'] = 1

df.region[df.region == 'northeast'] = 2

df.region[df.region == 'southeast'] = 3
df.head() #Make sure string types are replaced with integer type
df.keys()
print(df["charges"])
df.info()
# based on above display statements, no missing values in dataset

df.shape
#DATA VISUALIZATION
df.plot()
sns.set()
df.charges.hist() # Histogram generation for "Charges" column
sns.distplot(df.charges) 

sns.kdeplot(df.charges) # Bell Curve generation
df.hist(figsize=(12,12))

plt.plot()
df.describe()

#NORMALIZATION
alist = [12, 33,34,23,45,21,22,36,33,32,13,43,34,24,25,26,36,35,50]

aaray = np.array(alist)

aaray
plt.hist(aaray)
bray = aaray/aaray.max() # it is called min-max scaler technique.

bray
plt.hist(bray)
Y= df[["charges"]] 

X =df.drop(columns=["charges"])

X
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit_transform(X)
# Train test split

from sklearn.model_selection import train_test_split
X_scaled = scaler.fit_transform(X)

X_scaled
X_train,X_Test,Y_train,Y_Test = train_test_split(X_scaled,Y,test_size=0.1)
X_train.shape,X_Test.shape,Y_train.shape,Y_Test.shape
# machine learning

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,Y_train)
lr.coef_ # will give the slope
lr.intercept_ # will give the constant
# Testing phase

#lr.predict(X_Test)

Y_pred = lr.predict(X_Test)

Y_pred
error = pd.DataFrame(Y_pred, columns=["Predicted"])

error["Actual"] = Y_Test.reset_index(drop=True)

error
# Plot Actual Vs Predicted curve

plt.figure(figsize=(14, 8))

plt.plot(error.Predicted, label="Predicted")

plt.plot(error.Actual, label="Actual")

plt.legend()
#Calculate the error

error["error"] = error.Predicted - error.Actual

error
#Calculate Penalty

error["Penalty"] = error.error**2
error.Penalty.mean()
#Calculate Absolute mean error

abs(error.error).mean()