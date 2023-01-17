# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("../input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv")

anormality = data[data["class"] == "Abnormal"]

x = np.array(anormality.loc[:,'pelvic_incidence']).reshape(-1,1)
y = np.array(anormality.loc[:,'sacral_slope']).reshape(-1,1)

data.head()
data.info()
data.describe()
color_list = ['red' if i=='Abnormal' else 'green' for i in data.loc[:,'class']]
pd.plotting.scatter_matrix(data.loc[:, data.columns != 'class'],
                                       c=color_list,
                                       figsize= [15,15],
                                       diagonal='hist',
                                       alpha=0.5,
                                       s = 200,
                                       marker = '*',
                                       edgecolor= "black")
plt.show()
sns.countplot(x="class", data=data)
data.loc[:,'class'].value_counts()
plt.figure(figsize=[10,10])
plt.scatter(x=x,y=y)
plt.xlabel('pelvic_incidence')
plt.ylabel('sacral_slope')
plt.show()
from sklearn.linear_model import LinearRegression

linear_reg = LinearRegression()

linear_reg.fit(x,y)

predict_space = np.linspace(min(x), max(x)).reshape(-1,1)

prediction = linear_reg.predict(x)

print('R^2 score: ',linear_reg.score(x, y))
# Plot regression line and scatter
plt.plot(x, prediction, color='black', linewidth=3)
plt.scatter(x,y)
plt.xlabel('pelvic_incidence')
plt.ylabel('sacral_slope')
plt.show()
from sklearn.preprocessing import PolynomialFeatures
polynomial_regression = PolynomialFeatures(degree = 3) #degree increases reliability

x_poly = polynomial_regression.fit_transform(x) 

linear_regression2 = LinearRegression()
linear_regression2.fit(x_poly,y)


prediction_poly = linear_regression2.predict(x_poly)
plt.scatter(x,y)
plt.plot(x,prediction_poly,color="green",label = "poly")
plt.legend()
plt.show()

