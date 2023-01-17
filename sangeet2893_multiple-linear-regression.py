# Importing required libraries

import pandas as pd

import numpy as np

%matplotlib inline

import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d
data = pd.read_csv("../input/advertising.csv/Advertising.csv")
data.drop("Unnamed: 0", axis =1,inplace=True)
data.head()
plt.scatter(data.TV, data.sales, color='blue', label='TV', alpha=0.5)

plt.scatter(data.radio, data.sales, color='red', label='radio', alpha=0.5)

plt.scatter(data.newspaper, data.sales, color='green', label='newspaper', alpha=0.5)



plt.legend(loc="lower right")

plt.title("Sales vs. Advertising")

plt.xlabel("Advertising [1000 $]")

plt.ylabel("Sales [Thousands of units]")

plt.grid()

plt.show()
from sklearn.linear_model import LinearRegression

sk_model = LinearRegression()

sk_model.fit(data.drop('sales', axis=1), data.sales)
print("Intercept: ", sk_model.intercept_)

print("Coefficients: ", sk_model.coef_)
# Importing statsmodels

import statsmodels.formula.api as sm



# Fitting the OLS on data

model = sm.ols('sales ~ TV + radio + newspaper', data).fit()

print(model.params)
# Simple Linear regression for sales vs newspaper

model_npaper = sm.ols('sales ~ newspaper', data).fit()

print(model_npaper.params)
data.corr()
# Plotting correlation heatmap

plt.ylim(-.5,3.5)

plt.imshow(data.corr(), cmap=plt.cm.GnBu, interpolation='nearest',data=True)

plt.colorbar()

tick_marks = [i for i in range(len(data.columns))]

plt.xticks(tick_marks, data.columns, rotation=45)

plt.yticks(tick_marks, data.columns, rotation=45)



# Putting annotations

for i in range(len(data.columns)):

    for j in range(len(data.columns)):

        text = '%.2f'%(data.corr().iloc[i,j])

        plt.text(i-0.2,j-0.1,text)
print(model.summary2())
# Defining the function to evaluate amodel

def evaluateModel (model):

    print("RSS = ", ((data.sales - model.predict())**2).sum())

    print("R2 = ", model.rsquared)
ad = pd.read_csv("../input/advertising.csv/Advertising.csv")
# For TV

model_TV = sm.ols('sales ~ TV', ad).fit()

print("model_TV")

evaluateModel(model_TV)

print("------------")



# For radio

model_radio = sm.ols('sales ~ radio', ad).fit()

print("model_radio")

evaluateModel(model_radio)

print("------------")



# For newspaper

model_newspaper = sm.ols('sales ~ newspaper', ad).fit()

print("model_newspaper")

evaluateModel(model_newspaper)

print("------------")
# For TV & radio

model_TV_radio = sm.ols('sales ~ TV + radio', ad).fit()

print("model_TV_radio")

evaluateModel(model_TV_radio)

print("------------")



# For TV & newspaper

model_TV_newspaper = sm.ols('sales ~ TV + newspaper', ad).fit()

print("model_TV_newspaper")

evaluateModel(model_TV_newspaper)

print("------------")
# For TV, radio & newspaper

model_all = sm.ols('sales ~ TV + radio + newspaper', ad).fit()

print("model_all")

evaluateModel(model_all)

print("------------")
"""

Created on Mon May 11 19:27:49 2020



@author: Sangeet Aggarwal (@datasciencewithsan)

"""



# Importing few libraries again for readibility



import pandas as pd

import numpy as np

from mpl_toolkits import mplot3d

%matplotlib inline

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D





fig = plt.figure(figsize=(12,10))

ax = fig.add_subplot(211, projection='3d')



fig.suptitle('Regression: Sales ~ TV & radio Advertising')





# Defining z function (or sales in terms of TV and radio)

def z_function(x,y):

    return (2.938889 + (0.045765*y) + (0.188530*x))



X, Y = np.meshgrid(range(0,50,2),range(0,300,10))

Z = z_function(X, Y)





## Creating Wireframe

ax.plot_wireframe(X, Y, Z, color='black')

ax.set_xlabel('x')

ax.set_ylabel('y')

ax.set_zlabel('z')



## Creating Surface plot

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='winter', edgecolor='black', alpha=0.8)



## Adding Scatter Plot

ax.scatter(ad.radio, ad.TV, ad.sales, c='red', s=25)



## Adding labels

ax.set_xlabel('Radio')

ax.set_ylabel('TV')

ax.set_zlabel('Sales')

ax.text(0,150,1, '@DataScienceWithSan')



## Rotating for better view

ax.view_init(10,30)