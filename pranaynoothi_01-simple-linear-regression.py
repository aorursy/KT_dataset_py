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
import numpy as np # linear algebra

import matplotlib.pyplot as plt # Plotting

import pandas as pd # data processing, CSV file I/O
df = pd.read_csv('../input/random-salary-data-of-employes-age-wise/Salary_Data.csv')
df.head()
df.shape
df.isnull().sum()
X = df.iloc[:,:-1].values 

Y = df.iloc[:,1].values
X
Y
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 0)
X_train
Y_train
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train,Y_train)
#bias(b)

model.intercept_
#slope/weight (w)

model.coef_
Y_pred = model.predict(X_test)
Y_pred
from sklearn.metrics import r2_score
score = r2_score(Y_test,Y_pred)
score
plt.scatter(X_train,Y_train,color = 'red')

plt.plot(X_train,model.predict(X_train),color = 'blue')

plt.title('Salary vs Year of experience(training set)')

plt.xlabel('Years of exp')

plt.ylabel('Salary')

plt.show()
plt.scatter(X_test,Y_test,color = 'red')

plt.plot(X_train,model.predict(X_train),color = 'blue')# no need to give the X_test,y_pred values

plt.title('Salary vs Year of experience(testing set)')

plt.xlabel('Years of exp')

plt.ylabel('Salary')

plt.show()