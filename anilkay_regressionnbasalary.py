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
data=pd.read_csv("/kaggle/input/nba-salaries-prediction-in-20172018-season/2017-18_NBA_salary.csv")

data.head()
data.sort_values("Salary",ascending=False)[["Player","Salary","PER"]][0:10]
data[["PER","Salary"]].corr()
data[["PER","Salary","Age"]].corr()
data[["WS","Age"]].corr()
data[["DWS","Age"]].corr()
data[["USG%","Salary"]].corr()
data.sort_values("USG%",ascending=False)[["Player","Salary","USG%"]][0:10]
data2=data[data["USG%"]<=37]
data2[["USG%","Salary"]].corr()
data2.sort_values("MP",ascending=True)[["Player","Salary","MP"]][0:20]
data3=data2[data2["MP"]>=200]
data3[["USG%","Salary"]].corr()
data3[["PER","Salary"]].corr()
data3.sort_values("Salary",ascending=False)[["Player","Salary","PER"]][0:20]
data4=data3[data3["Player"]!="Carmelo Anthony"]
data4[["PER","Salary"]].corr()
data4.sort_values("OBPM",ascending=False)[["Player","Salary","OBPM"]][0:10]
data4[data4["Player"]=="Trey Burke"]
data5=data4[data4["G"]>=41]
data5.sort_values("OBPM",ascending=False)[["Player","Salary","OBPM"]][0:10]
data5[["PER","Salary"]].corr()
data5.columns
data5["MinPerGame"]=data5["MP"]/data5["G"]
data5[["MinPerGame","Salary"]].corr()
data5.sort_values("MinPerGame",ascending=False)[["Player","Salary","MinPerGame"]][0:10]
data5.sort_values("MinPerGame",ascending=True)[["Player","Salary","MinPerGame"]][0:10]
data5.head()

y=data5.iloc[:,1:2]

x=data5.iloc[:,3:]

del x["Tm"]
from sklearn.feature_selection import SelectKBest,f_regression

x_new = SelectKBest(f_regression, k=10).fit_transform(x, y)
x_new.shape
from sklearn.model_selection import train_test_split

x_train,x_test, y_train, y_test = train_test_split(x_new, y, test_size=0.33, random_state=42)
from sklearn.tree import DecisionTreeRegressor

dtreg=DecisionTreeRegressor()

dtreg.fit(x_train,y_train)

ypred=dtreg.predict(x_test)



import sklearn.metrics as metrik

metrik.mean_absolute_error(y_pred=ypred,y_true=y_test)
from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor(n_neighbors=7)

knn.fit(x_train,y_train)

ypred=knn.predict(x_test)

metrik.mean_absolute_error(y_pred=ypred,y_true=y_test)
from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor(n_neighbors=int(math.sqrt(x_train.shape[0])/2))

knn.fit(x_train,y_train)

ypred=knn.predict(x_test)

metrik.mean_absolute_error(y_pred=ypred,y_true=y_test)
from tpot import TPOTRegressor

tpot = TPOTRegressor(verbosity=2, random_state=19,max_time_mins=67)

tpot.fit(x_train, y_train)

ypred=tpot.predict(x_test)

metrik.mean_absolute_error(y_pred=ypred,y_true=y_test)
tpot.export("pipeline.py")