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
raindata = pd.read_csv('/kaggle/input/how-much-did-it-rain-ii/train.zip')
raindata_test = pd.read_csv('/kaggle/input/how-much-did-it-rain-ii/test.zip')
raindata.head()
raindata_test.head()

raindata_test = raindata_test.drop(['Id'],axis=1)
raindata_test.info()
raindata.info()
raindata = raindata.drop(['Id'],axis =1)
raindata.fillna(0, inplace = True)
raindata.isnull().values.any()
raindata.isnull().sum().sum()
from sklearn.linear_model import LinearRegression
x_train = raindata.drop(['Expected'],axis =1)

y_train = raindata[['Expected']]

#x_test = raindata_test
regression_model = LinearRegression()

regression_model.fit(x_train,y_train)
for idx, col_name in enumerate(x_train.columns):

    print("The coefficient for {} is {}".format(col_name, regression_model.coef_[0][idx]))
intercept = regression_model.intercept_[0]

print("The intercept for our model is {}".format(intercept))
regression_model.score(x_train, y_train)
regression_model.coef_.shape
#y_test = np.dot(raindata_test,np.transpose(regression_model.coef_))

#y_test1 = y_test + regression_model.intercept_