# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#import data

data = pd.read_csv("../input/column_2C_weka.csv")
print(data.info())
#normalized data

x_data = data.drop(["class"], axis = 1)
y_data = data["class"] #y_data = class (normal/abnormal)
x_data = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values

x_data.head()
#%% Show the ratio of normal/abnormal
import seaborn as sns

rate = y_data.value_counts()
plt.figure(figsize=[5,5])
plt.pie(rate.values, explode = [0, 0], labels = rate.index,  autopct = "%1.1f%%")
plt.show()
plt.figure(figsize=[15,5])

# Create dataframe and reshape
columns = list(x_data.columns) #column names

df = x_data.copy()
df["class"] = y_data #df = x_data + y_data
df = pd.melt(df, value_vars=columns, id_vars='class') #id = class olsun,  diğer columnları variable olarak dağıt

#Plot
pal = sns.cubehelix_palette(2, rot=.5, dark=.3)
sns.violinplot(x='variable', y='value',  hue='class', data=df,
               palette=pal,
               split=True, 
               inner="quart")
plt.show()
from mpl_toolkits.mplot3d import Axes3D #for 3d plot

#GET X,Y,Z AXIS VALUES
x_values= x_data["pelvic_incidence"].values.reshape(-1, 1) 
y_values= x_data["pelvic_tilt numeric"].values.reshape(-1, 1)
z_values = x_data["lumbar_lordosis_angle"].values.reshape(-1, 1) 

#PLOT SCATTER VALUES
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_values, y_values, z_values, c='r', marker='o', alpha=0.3)



#REGRESSION WITH LINEAR MODEL
from sklearn.linear_model import LinearRegression
xy_values = x_data.iloc[:, [0,1]].values #xy_values (pelvic_incidence,pelvic_tilt numeric) np.array

#LEARN Z VALUES ACCORDING TO (X,Y) VALUES
lr = LinearRegression()
lr.fit(xy_values, z_values)    #regression



# CREATE AN ARRAY FOR X AND Y VALUES
ax_values = np.arange(0, 1, 0.05)
#FIND THE Z_HEAD(PREDICT) VALUES CORRESPOND TO THESE X,Y VALUES
xy_for_predict= np.vstack((ax_values,ax_values)).T

#PREDICT z_head
z_head = lr.predict(xy_for_predict)

#PLOT REGRESSION LINE
ax.plot(ax_values, z_head, ax_values, c='b', label="Linear Regression(z_head) : lumbar_lordosis_angle")
ax.set_xlabel('pelvic_incidence (X)')
ax.set_ylabel('pelvic_tilt numeric (Y)')
ax.set_zlabel('lumbar_lordosis_angle (Z)')
plt.legend()
plt.show()


#LINEAR REGRESSION R SQUARE
from sklearn.metrics import r2_score
z_head = lr.predict(xy_values)
print("r_score: ", r2_score(z_values, z_head))
#%% DESICION TREE REGRESSION
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()

#REGRESSION WITH DECISION THREE 
tree_reg.fit(xy_values, z_values)

#R SQUARE
z_head = lr.predict(xy_values)
print("r_score: ", r2_score(z_values, z_head))
#PLOT SCATTER VALUES
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_values, y_values, z_values, c='r', marker='o', alpha=0.3)



#RANDOM FOREST ALGORITHM IMPLEMENT
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 100, random_state = 42) #(n_ estimators = number of tree)
rf.fit(xy_values, z_values.ravel())

#PREDICT
z_head = lr.predict(xy_for_predict)

#PLOT FOREST RANDOM REGRESSOR LINE
ax.plot(ax_values, z_head, ax_values, c='b', label="Random Forest Regressor(z_head) : lumbar_lordosis_angle")
ax.set_xlabel('pelvic_incidence (X)')
ax.set_ylabel('pelvic_tilt numeric (Y)')
ax.set_zlabel('lumbar_lordosis_angle (Z)')
plt.legend()
plt.show()



#R_SQUARE
z_head = rf.predict(xy_values)
print("r_score: ", r2_score(z_values, z_head))