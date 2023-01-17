import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
%matplotlib inline
from mlxtend.plotting import plot_linear_regression
import seaborn as sns

s_data = pd.read_csv('../input/student-dataset/student_scores%20-%20student_scores.csv')
print("Data imported successfully")

s_data.head(25)
x = np.array(s_data['Hours']).reshape((-1, 1))
y = np.array(s_data['Scores'])
print(x)
print(y)
model = LinearRegression()
model.fit(x, y)
model = LinearRegression().fit(x, y)
r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)

new_model = LinearRegression().fit(x, y.reshape((-1, 1)))
print('intercept:', new_model.intercept_)
print('slope:', new_model.coef_)
y_pred = model.predict(x)
print('predicted response:', y_pred, sep='\n')
y_pred = model.intercept_ + model.coef_ * x
print('predicted response:', y_pred, sep='\n')
x_new = np.arange(5).reshape((-1, 1))
print(x_new)
y_new = model.predict(x_new)
print(y_new)
plt.scatter(x , y, color = "red")
plt.plot(x, model.predict(x), color = "green")
plt.title("Scores vs Hours (Training set)")
plt.xlabel("Hours of study")
plt.ylabel("Scores")
plt.show()
## setting plot style 
plt.style.use('fivethirtyeight') 
  
## plotting residual errors in training data 
plt.scatter(model.predict(x), model.predict(x) - y, 
            color = "green", s = 10, label = 'Train data') 
  
## plotting residual errors in test data 
plt.scatter(model.predict(x), model.predict(x) - y, 
            color = "blue", s = 10, label = 'Test data') 
  
## plotting line for zero residual error 
plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2) 
  
## plotting legend 
plt.legend(loc = 'upper right') 
  
## plot title 
plt.title("Residual errors") 
  
## function to show plot 
plt.show() 
from sklearn.linear_model import LinearRegression
Regressor = LinearRegression()
Regressor.fit(x,y)
# Visualize the Data for Multiple Linear Regression
from mpl_toolkits.mplot3d import Axes3D


fig = plt.figure(figsize=(25,10))
### Set figure size
ax = fig.add_subplot(111, projection='3d')
ax.scatter(s_data['Hours'],s_data['Scores'],c='red', marker='o', alpha=0.5)
ax.plot_surface(x ,y, y_pred , color='b', alpha=0.3)
ax.set_xlabel('Hours')
ax.set_ylabel('Scores')
ax.set_zlabel('Adjusted')
plt.show()
sns.boxplot(x = 'Hours' , y = 'Scores' , data=s_data)
s_data.describe(include = 'all')
sns.pairplot(s_data)
corr=s_data.corr()
corr
s_data.hist(figsize=(10,10))
sns.distplot(s_data['Hours'],bins=1)
sns.distplot(s_data['Scores'],bins=1)
sns.heatmap(corr , annot=True)
s_data.plot.bar() 

# plot between 2 attributes 
plt.bar(s_data['Hours'], s_data['Scores']) 
plt.xlabel("Hours") 
plt.ylabel("Scores") 
plt.show() 
plt.pie(s_data['Hours'],autopct ='% 1.1f %%', shadow = True)  
plt.pie(s_data['Scores'],autopct ='% 1.1f %%', shadow = True)  
plt.show() 
sns.lmplot(x='Hours',y='Scores',data=s_data,legend=True,palette='red')
sns.countplot(x='Hours',data=s_data)
sns.countplot(x='Scores',data=s_data)