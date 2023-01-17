#importing libraries



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
#importing the dataset



data = pd.read_csv('../input/Salary_Data.csv')

data.head(3)
#we will built a simple linear regression model with 

# independent variable = YearsExperience

# dependent variable = Salary



#sqft_lot

YearsExperience = data.loc[:,"YearsExperience"].values.reshape(-1,1)

#price

Salary = data.loc[:,"Salary"].values.reshape(-1,1)
#creating training data and test data using sklearn



from sklearn.model_selection import train_test_split



YearsExperience_train,YearsExperience_test,Salary_train,Salary_test = train_test_split(YearsExperience,Salary, test_size = 1/3, random_state = 0)
#training the model



from sklearn.linear_model import LinearRegression



regressor = LinearRegression()

regressor.fit(YearsExperience_train,Salary_train)
Salary_predict = regressor.predict(YearsExperience_test)
#lets check the predictions



plt.scatter(YearsExperience_train,Salary_train,color="red")

plt.plot(YearsExperience_test,Salary_predict,color="blue")

plt.title('Salary vs Experience (Test set)')

plt.xlabel('Years of Experience')

plt.ylabel('Salary')

plt.show()
