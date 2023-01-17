#import packages

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sb
data=pd.read_csv('../input/salary_data.csv')

data.head()

data.info()
X=data.iloc[:,:-1].values

Y=data.iloc[:,1].values
sb.distplot(data['YearsExperience'])
sb.scatterplot(data['YearsExperience'],data['Salary'])
#splitting the dataset into the Training set and test set



from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
#fitting simple linear regression to the training set

from sklearn.linear_model import LinearRegression

regressor=LinearRegression()

regressor.fit(x_train,y_train)
regressor.score(x_test,y_test)
#predicting the test set results

y_predict=regressor.predict(x_test)

y_predict
regressor.predict([[1.5]])
#visualizing the training set results

plt.scatter(x_train,y_train,color='red')

plt.plot(x_train,regressor.predict(x_train),color='blue')

plt.title('Salary vs Experience(Training set)')

plt.xlabel('Years of Experience')

plt.ylabel('Salary')

plt.show()
plt.scatter(x_test,y_test,color='red')

plt.plot(x_train,regressor.predict(x_train),color='blue')

plt.title('Salary vs Experience(Training set)')

plt.xlabel('Years of Experience')

plt.ylabel('Salary')

plt.show()