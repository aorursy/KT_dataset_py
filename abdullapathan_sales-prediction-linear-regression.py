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
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
df = pd.read_csv("/kaggle/input/tvmarketing-dataset/tvmarketing.csv")

print(f"Data Shape: {df.shape}")
# Checking dataset columns

print(df.columns)



#Selecting Corresponding Features

X = df['TV'].values

y = df['Sales'].values



X = X.reshape(-1,1)

y = y.reshape(-1,1)
# X_feature - TV

# y_output - Sales

x_train, x_test, y_train, y_test = train_test_split(X,y,train_size=0.8,test_size=0.2,random_state=100)

print(f"X_train shape {x_train.shape}")

print(f"y_train shape {y_train.shape}")

print(f"X_test shap {x_test.shape}")

print(f"y_test shape {y_test.shape}")
import matplotlib.pyplot as plt

%matplotlib inline

plt.scatter(x_train,y_train,color='red')

plt.xlabel('TV')

plt.ylabel('Sale')

plt.title('Training data')

plt.show()
lm = LinearRegression()

lm.fit(x_train,y_train)

y_predict = lm.predict(x_test)

print(f"Train accuracy {round(lm.score(x_train,y_train)*100,2)} %")

print(f"Test accuracy {round(lm.score(x_test,y_test)*100,2)} %")
plt.scatter(x_train,y_train,color='red')

plt.plot(x_test,y_predict)

plt.xlabel("Years of Experience")

plt.ylabel("Salary in $")

plt.title("Trained model plot")

plt.plot
yoe = np.array([15,1.5,7.3,9.65])

yoe = yoe.reshape(-1,1)

yoe_sales = lm.predict(yoe)

for df in yoe_sales:

    print(f"$ {df}")