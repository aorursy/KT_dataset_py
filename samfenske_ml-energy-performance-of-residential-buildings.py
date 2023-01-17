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
df=pd.read_csv('/kaggle/input/eergy-efficiency-dataset/ENB2012_data.csv')

df.columns = ['relative_compactness', 'surface_area', 'wall_area', 'roof_area', 'overall_height',

                'orientation', 'glazing_area', 'glazing_area_distribution', 'heating_load', 'cooling_load']

df=df.reset_index()

df
import matplotlib.pyplot as plt

import seaborn as sns



# Correlation between inputs and outputs

plt.figure(figsize=(5,5))

sns.pairplot(data=df, y_vars=['cooling_load','heating_load'],

             x_vars=['relative_compactness', 'surface_area', 'wall_area', 'roof_area', 'overall_height',

                     'orientation', 'glazing_area', 'glazing_area_distribution',])

plt.show()
for column in df.columns:

    print("\n" + column)

    print(df[column].value_counts())
for column in df.columns:

    print("\n" + column)

    print(len(df[column].value_counts()))
#from sklearn.preprocessing import Normalizer

#nr = Normalizer(copy=False)



X = df.drop(['heating_load','cooling_load'], axis=1)

#X = nr.fit_transform(X)

y = df[['heating_load','cooling_load']]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 2)
#Import decision tree regressor

from sklearn.tree import DecisionTreeRegressor

# Create decision tree model 

dt_model = DecisionTreeRegressor(random_state=2)

# Apply the model

dt_model.fit(X_train, y_train)

# Predicted value

y_pred1 = dt_model.predict(X_test)
from sklearn.metrics import r2_score

r2_score(y_test,y_pred1)
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

#Visualize the heating load output before optimization

ax1.plot(X_test['index'],y_test['heating_load'],'o',color='red',label = 'Actual Values')

ax1.plot(X_test['index'],y_pred1[:,0],'X',color='yellow',label = 'Predicted Values')

ax1.set_xlabel('index')

ax1.set_ylabel('Heating Load')

ax1.set_title('Heating  Load Before Optimization')

ax1.legend(loc = 'upper right')



#Visualize the cooling load output before optimization

ax2.plot(X_test['index'],y_test['cooling_load'].values,'o',color='green',label = 'Actual Values')

ax2.plot(X_test['index'],y_pred1[:,1],'X',color='blue',label = 'Predicted Values')

ax2.set_xlabel('index')

ax2.set_ylabel('Cooling Load')

ax2.set_title('Cooling Load Before Optimization')

ax2.legend(loc = 'upper right')



ax1.figure.set_size_inches(15, 8)





plt.show()
# Finding the best decision tree optimization parameters



f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

# Max Depth

dt_acc = []

dt_depth = range(1,11)

for i in dt_depth:

    dt = DecisionTreeRegressor(random_state=2, max_depth=i)

    dt.fit(X_train, y_train)

    dt_acc.append(dt.score(X_test, y_test))

ax1.plot(dt_depth,dt_acc)

ax1.set_title('Max Depth')



#Min Samples Split

dt_acc = []

dt_samples_split = range(10,21)

for i in dt_samples_split:

    dt = DecisionTreeRegressor(random_state=2, min_samples_split=i)

    dt.fit(X_train, y_train)

    dt_acc.append(dt.score(X_test, y_test))

ax2.plot(dt_samples_split,dt_acc)

ax2.set_title('Min Samples Split')



plt.show()
#Min Sample Leaf

plt.figure(figsize = (5,5))

dt_acc = []

dt_samples_leaf = range(1,10)

for i in dt_samples_leaf:

    dt = DecisionTreeRegressor(random_state=123, min_samples_leaf=i)

    dt.fit(X_train, y_train)

    dt_acc.append(dt.score(X_test, y_test))



plt.plot(dt_samples_leaf,dt_acc)

plt.title('Min Sample Leaf')



plt.show()
# Decision tree optimization parameters

from sklearn.model_selection import GridSearchCV

parameters = {'max_depth' : [7,8,9],

              'min_samples_split': [16,17,18],

              'min_samples_leaf' : [6,7,8]}





#Create new model using the GridSearch

dt_random = GridSearchCV(dt_model, parameters)
dt_random.fit(X_train, y_train)
dt_random.best_params_
