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



# Any results you write to the current directory are saved as output.
X1 = np.random.randint(500,2000,size=(50, 1)) 

X2 = np.random.randint(100,500,size=(50, 1))

X3 = X1*3 + np.random.random()

Y = X1*X2

len(X2)


data = pd.DataFrame(X1)

data.columns = ['X1']

data['X2'] = X2

data['X3'] = X3

data['Y'] = Y

data
focus_cols = ['Y']

data.corr().filter(focus_cols).drop(focus_cols)
data.plot(kind = 'scatter', x = 'X1', y = 'Y', title  = 'scatter plot illustrating the relationship between X1 & Y', figsize  = (12,10))
data.plot(kind = 'scatter', x = 'X2', y = 'Y', title  = 'scatter plot illustrating the relationship between X1 & Y', figsize  = (12,10))
indpt_data = data[['X1','X2']]

depnt_data = data['Y']
from sklearn.model_selection import train_test_split as tts

indpt_data_train, indpt_data_test, depnt_data_train, depnt_data_test = tts(indpt_data, depnt_data, test_size=0.30)
from sklearn import linear_model as lm

reg = lm.LinearRegression()

reg.fit(indpt_data_train,depnt_data_train)
reg.coef_
print("Regression Coefficients")

pd.DataFrame(reg.coef_,index=indpt_data_train.columns,columns=["Coefficient"])
from sklearn.linear_model import Ridge

from yellowbrick.datasets import load_concrete

from yellowbrick.regressor import ResidualsPlot

model = Ridge()

visualizer = ResidualsPlot(model)

visualizer = ResidualsPlot(model)

visualizer.fit(indpt_data_train, depnt_data_train)

visualizer.score(indpt_data_test, depnt_data_test)



visualizer.show()
reg.intercept_
reg.score(indpt_data_test,depnt_data_test)
predicted_val = reg.predict(indpt_data_test)



dff = pd.DataFrame({'Actual': depnt_data_test, 'Predicted': predicted_val})

dff
a=440

b=20

data = {'x':[a],'y':[b]}

newdataf=pd.DataFrame(data)

reg.predict(newdataf)
from sklearn.metrics import mean_absolute_error



from math import sqrt

mean_absolute_error(depnt_data_test, predicted_val)
from sklearn.metrics import mean_squared_error

mean_squared_error(depnt_data_test, predicted_val)
sqrt(mean_squared_error(depnt_data_test, predicted_val))