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
import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d
data = [[2,0,81], [4,4,93], [6,2,91], [8,3,97]]

x1 = [i[0] for i in data]

x2 = [i[1] for i in data]

y = [i[2] for i in data]

ax = plt.axes(projection = '3d')

ax.set_xlabel('study_hours')

ax.set_ylabel('private_class')

ax.set_zlabel('score')



ax.dist=11

ax.scatter(x1,x2,y)

plt.show()
x1_data = np.array(x1)

x2_data = np.array(x2)

y_data = np.array(y)
a1 = 0

a2 = 0

b = 0
lr = 0.05
epochs = 2001
for i in range(epochs):

    y_pred = a1*x1_data +a2 *x2_data +b 

    error = y_data - y_pred

    a1_diff = -(1/len(x1_data)) * sum(x1_data * (error))

    a2_diff = -(1/len(x2_data)) * sum(x2_data * (error))

    b_new = -(1/len(x1_data)) * sum(y_data - y_pred)

    a1 = a1-lr*a1_diff

    a2 = a2-lr*a2_diff

    b = b-lr*b_new

    

    if i % 100 ==0:

        print("epoch=%.f, 기울기1=%.04f, 기울기2=%.04f, 절편=%.04f" %(i,a1,a2,b))
import statsmodels.api as statm

import statsmodels.formula.api as statfa



X = [i[0:2] for i in data]

y = [i[2] for i in data]



X_1 = statm.add_constant(X)

results= statm.OLS(y, X_1).fit()
hour_class=pd.DataFrame(X, columns = ['study_hours', 'private_class'])

hour_class['Score']=pd.Series(y)



model = statfa.ols(formula='Score ~ study_hours + private_class', data=hour_class)



results_formula = model.fit()



a, b = np.meshgrid(np.linspace(hour_class.study_hours.min(), hour_class.study_hours.max(), 100),

                   np.linspace(hour_class.private_class.min(), hour_class.private_class.max(), 100))



X_ax = pd.DataFrame({'study_hours': a.ravel(), 'private_class': b.ravel()})

fittedY = results_formula.predict(exog=X_ax)
fig = plt.figure()

graph = fig.add_subplot(111, projection = '3d')



graph.scatter(hour_class['study_hours'], hour_class['private_class'], hour_class['Score'],

              c='blue', marker = 'o', alpha=1)

graph.plot_surface(a,b, fittedY.values.reshape(a.shape),

                   rstride=1, cstride=1, color='none', alpha=0.4)



graph.set_xlabel('study hours')

graph.set_ylabel('private class')

graph.set_zlabel('Score')

graph.dist =11



plt.show()

