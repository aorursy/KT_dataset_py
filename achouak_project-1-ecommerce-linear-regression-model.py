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
# Import libraries 

import pandas as pd 

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn import metrics
#Read customer csv file 

customer_file="../input/ecommerce-customers/Ecommerce Customers.csv"

customer = pd.read_csv(customer_file)

#show the head of customer DataFrame

customer.head()
#show statistcs information

customer.describe()
#visualize data

sns.pairplot(customer)
sns.lmplot(x="Yearly Amount Spent", y="Length of Membership", data=customer)
features= ["Avg. Session Length", "Time on App" ,"Time on Website", "Length of Membership"]

X=customer[features] #Features

y=customer["Yearly Amount Spent"] #Outcome



# split the data (20% for testing 80%for training)

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=1)

#create an instance of Linear regession model

lmodel= LinearRegression()

#fit the model with the training data

lmodel.fit(X_train, y_train)

#testing the model by making predicitions  

prediction = lmodel.predict(X_test)
#visualize real data and predictif ones 

plt.scatter(y_test, prediction)

plt.xlabel("Real")

plt.ylabel("prediction")

plt.title("Yearly Amount Spent")
MAE= metrics.mean_absolute_error(y_test,prediction) #Mean absolute error 

MSE= metrics.mean_squared_error(y_test,prediction) #Mean squared error 

RMSE = np.sqrt(MSE) # Root Mean squared error 
print("Mean absolute error: ",MAE )
coef= lmodel.coef_

coeff_table = pd.DataFrame(coef, features, columns=['Coefficient'])

coeff_table