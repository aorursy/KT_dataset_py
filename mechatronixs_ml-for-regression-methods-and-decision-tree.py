# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

# Any results you write to the current directory are saved as output.
data2_weka = pd.read_csv('../input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv')

print(plt.style.available) # look at available plot styles

plt.style.use('ggplot')
data2_weka.head()
data2_weka.info()
data2_weka.describe()
color_list = ['red' if i=='Abnormal' else 'green' for i in data2_weka.loc[:,'class']]

pd.plotting.scatter_matrix(data.loc[:, data2_weka.columns != 'class'],

                                       c=color_list,

                                       figsize= [15,15],

                                       diagonal='hist',

                                       alpha=0.5,

                                       s = 200,

                                       marker = '*',

                                       edgecolor= "black")

plt.show()
data1 = data2_weka[data2_weka['class'] =='Abnormal']

data1.head()
x = np.array(data1.loc[:,'pelvic_incidence']).reshape(-1,1)

y = np.array(data1.loc[:,'lumbar_lordosis_angle']).reshape(-1,1)

#Scatter

plt.figure(figsize=[10,10])

plt.scatter(x,y)

plt.xlabel('pelvic_incidence')

plt.ylabel('lumbar_lordosis')

plt.show()
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x,y)



y_head = lr.predict(x)

plt.plot(x,y_head, color="black")

plt.scatter(x,y)

plt.xlabel('pelvic_incidence')

plt.ylabel('lumbar_lordosis')

plt.show()

from sklearn.metrics import r2_score

print("r_square_score", r2_score(y,y_head))
from sklearn.preprocessing import PolynomialFeatures

pr = PolynomialFeatures(degree = 5)



x_polynomial = pr.fit_transform(x)



lr2 = LinearRegression()

lr2.fit(x_polynomial,y)



y_head2 = lr2.predict(x_polynomial).reshape(-1,1)

plt.plot(x,y_head2,color = "yellow",linewidth =3,label = "poly_reg")

plt.legend()

plt.scatter(x,y)

plt.xlabel('pelvic_incidence')

plt.ylabel('lumbar_lordosis')

plt.show()
from sklearn.metrics import r2_score

print("r_square_score", r2_score(y,y_head2))
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=50, random_state=42 )

rf.fit(x,y)



x_ = np.arange(min(x),max(x),0.05).reshape(-1,1)

y_head3 = rf.predict(x_)

plt.figure(figsize=[10,10])

plt.plot(x_,y_head3,color="green",label="randomforest")

plt.scatter(x,y,color="red")

plt.xlabel('pelvic_incidence')

plt.ylabel('lumbar_lordosis')

plt.show()
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()

tree_reg.fit(x,y)



y_head4 = tree_reg.predict(x)

plt.figure(figsize=[10,10])

plt.scatter(x,y,color="red")

plt.plot(x,y_head4,color="green",label="decisiontree")

plt.xlabel('pelvic_incidence')

plt.ylabel('lumbar_lordosis')

plt.show()

from sklearn.metrics import r2_score

print("r_square_score", r2_score(y,y_head4))
