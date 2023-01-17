# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing datasets
data = pd.read_csv("../input/kc_house_data.csv")
space=data['sqft_living']
price=data['price']

#change X into 2D array   
X=np.array(space).reshape(-1,1)
Y=np.array(price)

#split data into train sets and test sets
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

#import LinearRegression
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)

y_pred=regressor.predict(X_test)

#Visualizing training sets 
plt.scatter(X_train,Y_train,color="red",label="Space area")
plt.plot(X_train,regressor.predict(X_train),color="blue",label="Price Value")
plt.xlabel("Space")
plt.ylabel("Price")
plt.legend()
plt.show()
   
#Visualizing test sets
plt.scatter(X_test,Y_test,color='red',label="Space area")
plt.plot(X_train,regressor.predict(X_train),color="blue",label="Price Value")
plt.xlabel("Space")
plt.ylabel("Price")
plt.legend()
plt.show()

# Any results you write to the current directory are saved as output.
