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
import seaborn as sns 
import matplotlib.pyplot as plt 
plt.style.use('ggplot')
%matplotlib inline
plt.rcParams['figure.figsize'] = (12, 8)
data = pd.read_csv('/kaggle/input/salary-data-simple-linear-regression/Salary_Data.csv')
data.head()


data.info()
ax = sns.scatterplot(x='YearsExperience', y='Salary', data=data)
ax.set_title("Salary according to YearsExperience");
def cost_function(x, y, theta ) : 
    m = len(y)
    y_pred = x.dot(theta)
    
    fn =  ( y_pred - y )**2
    
    cost = 1 / ( (2*m)* np.sum(fn))
    
    return cost

m = data.YearsExperience.values.size 
x = np.append(np.ones((m,1)), data.YearsExperience.values.reshape(m, 1), axis=1) 
y = data.Salary.values.reshape(m,1)
theta = np.zeros((2,1))

cost_function(x,y,theta)
def gradient_descent(x, y, theta, alpha, iters) :
    m = len(y)
    # tracking the history of all Costs in costs list
    costs = []
    for i in range(iters) :
        y_pred = x.dot(theta)
        fn = np.dot(x.T, (y_pred - y))
        theta -= alpha*1/m*np.sum(fn)
        costs.append(cost_function(x,y,theta))
    return theta,costs

theta, costs = gradient_descent(x, y, theta, alpha=0.01, iters=1000)

print("h(x) = {} + {}x1".format(str(round(theta[0, 0], 2)),
                                str(round(theta[1, 0], 2))))

from mpl_toolkits.mplot3d import Axes3D

theta_0 = np.linspace(-20,20,1000)
theta_1 = np.linspace(-1,4,1000)

cost_values = np.zeros((len(theta_0), len(theta_1)))

for i in range(len(theta_0)):
    for j in range(len(theta_1)):
        t = np.array([theta_0[i], theta_1[j]])
        cost_values[i, j] = cost_function(x, y, t)
        
fig = plt.figure(figsize = (12, 8))
ax = fig.gca(projection = '3d')

surf = ax.plot_surface(theta_0, theta_1, cost_values, cmap = "viridis", linewidth = 0.2)
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.xlabel("$\Theta_0$")
plt.ylabel("$\Theta_1$")
ax.set_zlabel("$J(\Theta)$")
ax.set_title("Cost Surface")
ax.view_init(30,330)

plt.show()
plt.plot(costs)
plt.xlabel("Iterations")
plt.ylabel("$J(\Theta)$")
plt.title("Values of Cost Function over iterations of Gradient Descent");
theta = np.squeeze(theta)
sns.scatterplot(x = "YearsExperience", y= "Salary", data = data)

x_value=[x for x in range(1, 12)]
y_value=[(x * theta[1] + theta[0]) for x in x_value]
sns.lineplot(x_value,y_value)

plt.xlabel("Years Experience")
plt.ylabel("Salary")
plt.title("Linear Regression Fit");
def predict(x, theta):
    y_pred = np.dot(theta.transpose(), x)
    return y_pred
x1 = 8.7
y_pred_1 = predict(np.array([1, x1]),theta) 

print("Salary [ for experience {x1} ] = ".format(x1=x1) + str(round(y_pred_1, 5)))