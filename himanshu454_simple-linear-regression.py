import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
data = pd.read_csv("../input/salary-data-simple-linear-regression/Salary_Data.csv")
data.head(10)
data.info()
data.describe()

plt.figure(figsize=(16,9))
plt.scatter(data["YearsExperience"] , data["Salary"]  ,color = 'red')
plt.xlabel('Years')
plt.ylabel('Salary')
plt.title('Salary Prediction')
plt.show()
x = data.iloc[:,0:-1].values
y = data.iloc[:,-1].values
train_x , test_x , train_y , test_y = train_test_split(x,y,test_size = 0.7 , random_state = 100)
model = LinearRegression()
model.fit(train_x , train_y)
pred_y = model.predict(test_x)
# Plotting the actual and predicted values

c = [i for i in range (1,len(test_y)+1,1)]
plt.plot(c,test_y,color='r',linestyle='-')
plt.plot(c,pred_y,color='b',linestyle='-')
plt.xlabel('Salary')
plt.ylabel('index')
plt.title('Prediction')
plt.show()
# plotting the error
c = [i for i in range(1,len(test_y)+1,1)]
plt.plot(c,test_y-pred_y,color='green',linestyle='-')
plt.xlabel('index')
plt.ylabel('Error')
plt.title('Error Value')
plt.show()
mse = mean_squared_error(test_y,pred_y)
mse
rsq= r2_score(test_y,pred_y)
rsq
# Intecept and coeff of the line
print('Intercept of the model:',model.intercept_)
print('Coefficient of the line:',model.coef_)