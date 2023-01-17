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

import seaborn as sbn

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

%matplotlib inline
dataframe = pd.read_csv("/kaggle/input/advertising-data/Advertising.csv")
dataframe.head(10)
dataframe = dataframe.iloc[:, 1:len(dataframe)]

dataframe.head(10)
dataframe.info()
X = dataframe.drop(["Sales"], axis= 1)

y = dataframe[["Sales"]]

X
y
X_train , X_test , y_train , y_test = train_test_split(X ,y , test_size = 0.2 , random_state = 15)

X_train
X_test
sbn.jointplot(x = "TV" , y = "Sales" , data = dataframe , kind = "reg") # x -> feature y -> label

plt.xlim(-10 , 310)

plt.ylim(bottom = 0)
simpleX = X_train[["TV"]]

simpleX
regression = LinearRegression()
model = regression.fit(simpleX,y_train)
constant = model.intercept_

coefficient = model.coef_

print("Constant : " , constant ,"\n","Coefficient : " , coefficient)
graph = sbn.regplot(dataframe["TV"] ,dataframe["Sales"]  , ci = None , scatter_kws = { "color" : "r" , "s" : 9} )

graph.set_title("Sales = Constant + Coefficient * TV graph")

graph.set_xlabel("TV spending")

graph.set_ylabel("Sales")

plt.xlim(-10,300)

plt.ylim(bottom = 0)

plt.show()
# for example let's find TV spending = 165 manually

salesmanually = constant + coefficient*165

# let's model find TV spending = 165

saleswithmodel = model.predict([[165]])

print("salesmanually : " , salesmanually , "\nsaleswithmodel : ", saleswithmodel )
simpleX_test = X_test["TV"]



simpleX_test = pd.DataFrame(simpleX_test )

realvalues = y_test.values

realvalues = pd.DataFrame(realvalues)

realvalues
realvalues

predictvalues =pd.DataFrame(model.predict(simpleX_test)[0:len(realvalues)])

errordf = pd.concat([realvalues , predictvalues] , axis =1 , ignore_index = True )

errordf.columns = ["real" , "predict"]

errordf["error"] = errordf["real"] - errordf["predict"]

errordf
# finding mean squarared error (mse)

errordf["se"] = errordf["error"] ** 2 # se means squared error

errordf
MSE = np.mean(errordf["se"])

MSE
MultiLinearRegression = LinearRegression()
Model = MultiLinearRegression.fit(X_train , y_train)
coefficient = Model.coef_

constant = Model.intercept_

print("coefficient is : " , coefficient , "Constant is : ", constant)
X_test.head(1)
manualtest = constant + coefficient[0][0] * 66.9 + coefficient[0][1] * 11.7 + coefficient[0][2] * 36.8

testwithmodel= Model.predict([[66.9 , 11.7 , 36.8]])

print("Manually : " , manualtest , "\nWith Model : ", testwithmodel)

#As we can see they are same 
multipleX_test = pd.DataFrame(X_test )

multiplerealvalues = y_test.values

multiplerealvalues = pd.DataFrame(multiplerealvalues)
multiplepredictedvalues = pd.DataFrame(Model.predict(multipleX_test)[0:len(realvalues)])

errordf2 = pd.concat([multiplerealvalues , multiplepredictedvalues ], axis = 1 , ignore_index = True)

errordf2.columns = ["mreal" , "mpredicted"]

errordf2["error"] = errordf2["mreal"] - errordf2["mpredicted"]

errordf2

from sklearn.metrics import mean_squared_error , mean_absolute_error
MSE = mean_squared_error(errordf2["mreal"] , errordf2["mpredicted"])

MSE
RMSE = np.sqrt(MSE)

RMSE
MAE = mean_absolute_error(errordf2["mreal"] , errordf2["mpredicted"])

MAE