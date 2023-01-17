import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
population_df = pd.read_csv('../input/API_SP.POP.TOTL_DS2_en_csv_v2.csv', skiprows = range(0,4))
population_df.head()
USA_df = population_df.iloc[249]
print(USA_df)
Atheist_US_= [0.153,0.160,0.168,0.174,0.186,0.196,0.212,0.228];
USA2007_2014_df = USA_df.iloc[51:59]
USA2007_2014_df
USA2007_2014_Population_Atheist = USA2007_2014_df[0:]*Atheist_US_
USA2007_2014_Population_Atheist

Atheist = pd.DataFrame()
Atheist_US_= [4.60884e+07,4.8655e+07,5.15376e+07,5.38266e+07,5.79694e+07,6.15437e+07,6.70354e+07,7.26325e+07];
Atheist = Atheist.append(Atheist_US_)
Atheist

#Data :-
#2007    4.60884e+07
#2008     4.8655e+07
#2009    5.15376e+07
#2010    5.38266e+07
#2011    5.79694e+07
#2012    6.15437e+07
#2013    6.70354e+07
#2014    7.26325e+07
Suicide = pd.DataFrame()
Suicide_US_= [11.3,11.6,11.8,12.1,12.3,12.6,12.6,13.0];
Suicide = Suicide.append(Suicide_US_)
Suicide
import seaborn as sns  
grid = sns.JointGrid(Atheist,Suicide, space=0, size=6, ratio=50)
grid.plot_joint(plt.scatter,color="g")

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from pandas import Series, DataFrame
from sklearn.neighbors import KNeighborsRegressor

X = Atheist
y = Suicide

X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, y, test_size=0.15, random_state=0)
X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=0.15/0.85, random_state=0)

model = KNeighborsRegressor(n_neighbors=3)
model.fit(X_train, Y_train)
print(model.score(X_train, Y_train))
plt.scatter(X_train, Y_train, marker='.')
plt.xlabel('Atheism')
plt.ylabel('SuicideRates')
y_pred = model.predict(X_val)
y_actual = Y_val
mean_squared_error(y_actual, y_pred)
y_pred_test = model.predict(X_test)
y_actual_test = Y_test
mean_squared_error(y_actual_test, y_pred_test)
# Test R^2
print(model.score(X_test, y_actual_test))
plt.scatter(y_pred_test, y_actual_test, marker='.')
plt.xlabel('Predicted y')
plt.ylabel('Actual y')
plt.show()