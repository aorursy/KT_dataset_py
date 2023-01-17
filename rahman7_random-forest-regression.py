# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# import Libraries:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
# import dataset:
df=pd.read_csv("../input/Position_Salaries.csv")
df
# visualization :
sns.pairplot(df)
sns.countplot(df['Salary'])
sns.countplot(df['Level'])
sns.scatterplot(df['Level'],df['Salary'],data=df)
# make the dataset in the form of dependent and independent :
X=df.iloc[:,1:2].values
y=df.iloc[:,2].values
y=y.reshape(len(y),1)

X

y
sns.heatmap(X)
sns.heatmap(y)
# fitting the dataset on Random_forest:
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=100,random_state=0)
regressor.fit(X,y)
# predictions:
y_pred=regressor.predict(X)
y_pred
# visualization of Random-forest:
plt.scatter(X,y,color='red')
plt.plot(X,regressor.predict(X),color='blue')
plt.title('Salary Vs Level')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()
# visualization  Random_forest(Higher maginification):
X_grid=np.arange(min(X),max(X),0.01)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.title('Salary Vs Level')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()
