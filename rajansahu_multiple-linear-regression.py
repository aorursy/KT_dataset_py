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
df =pd.read_csv("/kaggle/input/50-companies-datasets/50_companies.csv")
df.head()
df.info()
df.describe()
df.corr()[['Profit']]

from matplotlib import pyplot as plt
import numpy as np

x_set=["'Major&Minor Work'","'R&D Spend'","'Marketing Spend'"]

y = df['Profit']
for x in x_set:
    x1=df[x]
    plt.scatter(x1, y, color='r')
    plt.xlabel(x)
    plt.ylabel('Salary')
    plt.title('Relation ')
    plt.show()
#split the data into depenent and independent variable
X= df.iloc[:,:-1].values
Y=df.iloc[:,4].values
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
print(X)
#onehotencoder = OneHotEncoder(categories= 'auto' )
#onehotencoder = OneHotEncoder(categorical_features = [3])
#X = onehotencoder.fit_transform(X).toarray()

ct = ColumnTransformer([("Location", OneHotEncoder(), [3])], remainder = 'passthrough')
X = ct.fit_transform(X)

print(X)
#splitting the dataset as training and testing dataset
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2, random_state=0)


# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
#Actual Value 
print(Y_test)
#Prediced Value
print(y_pred)