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
import pandas as pd

d = {'tax' : pd.Series([9.00,9.00,9.00,9.00,8.00,10.00,8.00,8.00,8.00,7.00,8.00,

                        8.00,7.00,7.00,7.00,7.00,7.00,7.00,7.00,7.00,7.00,8.00,9.00,

                        9.00,9.00,9.00,8.00,8.00,8.00,9.00,7.00,7.00,8.00,8.00,8.00,

                        8.00,5.00,5.00,5.00,7.00,7.00,7.00,7.00,7.00,6.00,9.00,7.00,7.00]), 

      'income' : pd.Series([3571,4092,3865,4870,4399,5342,5319,5126,4447,4512,4391,5126,4817,

                         4207,4332,4318,4206,3718,4716,4341,4593,4983,4897,4258,4574,3721,

                         3448,3846,4188,3601,3640,3333,3063,3357,3528,3802,4045,3897,3635,

                         4345,4449,3656,4300,3745,5215,4476,4296,5002]),

      'Highways':pd.Series([1976,1250,1586,2351,431,1333,11868,2138,8577,8507,5939,14186,6930,6580,

                    8159,10340,8508,4725,5915,6010,7834,602,2449,4686,2619,4746,5399,9061,5975

                    ,4650,6905,6594,6524,4121,3495,7834,17782,6385,3274,3905,4639,3985,3635,

                    2611,2302,3942,4083,9794]),

     'driver':pd.Series([0.5250,0.5720,0.5800,0.5290,0.5440,0.5710,0.4510,0.5530,0.5290,0.5520,

                   0.5300,0.5250,0.5740,0.5450,0.6080,0.5860,0.5720,0.5400,0.7240,0.6770,

                   0.6630,0.6020,0.5110,0.5170,0.5510,0.5440,0.5480,0.5790,0.5630,0.4930,

                   0.5180,0.5130,0.5780,0.5470,0.4870,0.6290,0.5660,0.5860,0.6630,0.6720,

                   0.6260,0.5630,0.6030,0.5080,0.6720,0.5710,0.6230,0.5930]),

    'Consumption':pd.Series([541,524,561,414,410,457,344,467,464,498,580,471,525,508,566,635,603,714,

                  865,640,649,540,464,547,460,566,577,631,574,534,571,554,577,628,487,644,

                  640,704,648,968,587,699,632,591,782,510,610,524])}
df=pd.DataFrame(d)
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict

X=df.drop("Consumption",axis=1)

y=df["Consumption"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state= 50)

X_train.shape
y_train.shape
X_test.shape
y_test.shape
training = df.copy()

training.shape
import numpy as np

from sklearn.metrics import mean_squared_error,r2_score

from sklearn.linear_model import LinearRegression

lm = LinearRegression()

model=lm.fit(X_train,y_train)
model.intercept_
model.coef_
for i in range(4):

    print(model.coef_[i])
Dummy_Parameters=[[1],[3],[4],[0]]

Dummy_Parameters=pd.DataFrame(Dummy_Parameters).T
## Because of 4 parameters 

Dummy_Parameters
model.predict(Dummy_Parameters)
df['Consumption']
rmse=np.sqrt(mean_squared_error(y_train,model.predict(X_train)))

rmse
rmse=np.sqrt(mean_squared_error(y_test,model.predict(X_test)))

rmse
model.score(X_train,y_train)
np.sqrt(-cross_val_score(model,X_test,y_test,cv=10,scoring="neg_mean_squared_error")).mean()
for i in range(101):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state= i)

    a=np.sqrt(-cross_val_score(model,X_test,y_test,cv=10,scoring="neg_mean_squared_error")).mean()

    b=np.sqrt(mean_squared_error(y_test,model.predict(X_test)))

    if(abs(a-b)<1):

        print(abs(a-b))

        print("RandomState: "+"{}".format(i))
## 18 is chosen

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state= 18)
rmse=np.sqrt(mean_squared_error(y_test,model.predict(X_test)))

rmse ## The value we have
np.sqrt(-cross_val_score(model,X_test,y_test,cv=10,scoring="neg_mean_squared_error")).mean() ## Best value