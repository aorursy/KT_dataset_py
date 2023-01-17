#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as mystats
from pandas.tools.plotting import scatter_matrix
from statsmodels.formula.api import ols as myols
#Importing the dataset
data = pd.read_csv("../input/Salary_Data.csv")
data
x = data.iloc[:,:1].values
x
y= data.iloc[:,-1].values
y
#Correlation analysis
scatter_matrix(data)
plt.show()
np.corrcoef(data.YearsExperience,data.Salary)
plt.scatter(x,y)
plt.show()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25)
from sklearn.linear_model import LinearRegression
mymodel = LinearRegression()
mymodel.fit(x_train,y_train)

y_pred = mymodel.predict(x_test)
plt.scatter(x_train,y_train,color = 'b')
plt.plot(x_train,mymodel.predict(x_train),color = 'r')
plt.title('Salary v/s YearsExperience')
plt.xlabel('Salary')
plt.ylabel('YearsExperience')
plt.show()
plt.scatter(x_test,y_test,color = 'b')
plt.plot(x_train,mymodel.predict(x_train),color = 'r')
plt.title('Salary v/s YearsExperience')
plt.xlabel('Salary')
plt.ylabel('YearsExperience')
plt.show()
