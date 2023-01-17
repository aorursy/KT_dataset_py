# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv("/kaggle/input/years-of-experience-and-salary-dataset/Salary_Data.csv")
data.head()
data.shape
data.info()
X= data['YearsExperience'].values

Y= data['Salary'].values
print(X)
print(Y)
plt.scatter(X,Y)
plt.title("Line plot between YearsExperience and Salary")
plt.xlabel("YearsExperience")
plt.ylabel("Salary")
plt.show()
class LinearRegression:
    
    def __init__ (self):
        self.slope = 0
        self.intercept = 0
    
    def fit(self, X, Y):
        mean_x = np.mean(X)
        mean_y = np.mean(Y)
        
        n = 0
        d = 0
        for (i,j) in zip(X, Y):
            n = n + (i-mean_x)*(j-mean_y)
            d = d + (i-mean_x)*(i-mean_x)
        
        self.slope = n/d
        self.intercept = mean_y - (self.slope*mean_x)
        
        print("Slope:", self.slope)
        print("Intercept:", self.intercept)
    
    def predict(self, value):
        ans = (self.slope*value) + self.intercept
        return ans
model = LinearRegression()
model.fit(X, Y)
plt.scatter(X, Y)
plt.plot([min(X), max(X)], [model.predict(min(X)), model.predict(max(X))], color='red')
plt.title("Line plot between YearsExperience and Salary")
plt.xlabel("YearsExperience")
plt.ylabel("Salary")
plt.show()
model.predict(4)
model.predict(7)
model.predict(10)