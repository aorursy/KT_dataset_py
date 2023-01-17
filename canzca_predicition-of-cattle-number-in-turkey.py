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

df=pd.read_csv("../input/cows.csv")

df=df.drop(["Unnamed: 0"],axis=1)# delete index

df # year_> number of cattle,gnp-> grown natinonal pruduct ng-> national growth

df.describe() # basic statistic

pd.isnull(df)# we have to check if we see true we have to add correct value (fillna median etc)
from sklearn import linear_model

from sklearn.model_selection import train_test_split # sor separaiton



y=df["number"]# our goal estimate this 

X=df[["year","dollar","gnp","ng"]] # all columms added 

X_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=0) #we separeted our data trian and test
reg=linear_model.LinearRegression() #we made our model

df1=reg.fit(X_train,y_train)
#we want to draw our train data 

import matplotlib.pyplot as plt

y_predict=reg.predict(X_train)

df2=pd.DataFrame(reg.coef_,X.columns,columns=["cofficient"])

df2 # our estimate ^s equal is like y=m1*x+m2*x+m3*x+b m is our cofficient b is intercept



b=reg.intercept_

b

df3=pd.DataFrame({"estimate":reg.predict(X_train),"real":y_train})

df3.plot(kind="bar")
df4=pd.DataFrame({"our test result:":reg.predict(x_test),"real:":y_test})

df4

df4.plot(kind="bar")
#now we have to chek our eror and accuracy

from sklearn import metrics

import numpy as np

mean_abssulate_Eror=metrics.mean_absolute_error(y_test,reg.predict(x_test))

Mean_Squared_Error=metrics.mean_squared_error(y_test, reg.predict(x_test))

accurancy=metrics.r2_score(y_test,reg.predict(x_test))

df5=pd.DataFrame({"mean_abssulate_Eror":[mean_abssulate_Eror],"Mean_Squared_Error":[Mean_Squared_Error],"accurancy":[accurancy]})

df5
print( "2019 cattle number prediction:",(int(reg.predict([[2019,6,800,-2.5]]))))