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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

datasheet=pd.read_csv('/kaggle/input/calcofi/bottle.csv')

X=datasheet[['Salnty']]

Y=datasheet[['T_degC']]





from sklearn.impute import SimpleImputer

imputer=SimpleImputer()



X=imputer.fit_transform(X)



Y=imputer.fit_transform(Y)

print(X)

print(Y)

print(X.shape)

print(Y.shape)











from sklearn.model_selection import train_test_split

X_Train,X_Test,Y_Train,Y_Test=train_test_split(X,Y,train_size=0.2,random_state=1)



from sklearn.linear_model import LinearRegression



regressor=LinearRegression()

regressor.fit(X_Train,Y_Train)

Y_Pred=regressor.predict(X_Train)





#Visualizing Data



plt.scatter(X_Train,Y_Train,color='red')

plt.plot(X_Train,regressor.predict(X_Train),color='blue')

plt.title('Salanity Vs Tempature Training data')

plt.xlabel('Salanity')

plt.ylabel('Tempaerature')

plt.show()



#Visualizing test data



plt.scatter(X_Test,Y_Test,color='yellow')

plt.plot(X_Test,regressor.predict(X_Test),color='green')

plt.title('Salanity Vs Temperature Test Data')

plt.xlabel('Salanity')

plt.ylabel('Temperature')

plt.show()