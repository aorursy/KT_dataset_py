# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
tsunami_data = pd.read_csv('../input/inland_sea_tsunami_damage_data.csv')
tsunami_data.head()
X = tsunami_data

y = X.damage

X.drop(['berthing_place'], axis=1, inplace=True)

X.drop(['gauge_point'], axis=1, inplace=True)

X.drop(['depth'], axis=1, inplace=True)

X = X.drop(['damage'], axis=1)

X = pd.get_dummies(X)
tsunami_data.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, test_size=0.1,random_state=0)
# Select categorical columns 

categorical_cols = [cname for cname in X_train.columns if X_train[cname].dtype == "object"]



# Select numeric columns

numeric_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ['int64', 'float64']]
X_train.head()
X_test.head()
from xgboost import XGBRegressor



# Define the model

my_model_1 = XGBRegressor(random_state=1) # Your code here



# Fit the model

my_model_1.fit(X_train, y_train) # Your code here
from sklearn.metrics import mean_absolute_error



# Get predictions

predictions_1 = my_model_1.predict(X_test) # Your code here

X_test.head()

print(predictions_1)



# Calculate MAE

mae_1 = mean_absolute_error(predictions_1, y_test) # Your code here



# Print MAE

print("Mean Absolute Error:" , mae_1)
X_test.head()
import shap

explainer = shap.TreeExplainer(my_model_1)

shap_values = explainer.shap_values(X)

shap.summary_plot(shap_values, X)
shap.dependence_plot(("maxTsuH"), shap_values, X)
import xgboost as xgb

xgb.plot_importance(my_model_1)
from xgboost import plot_tree

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(30, 30))

plot_tree(my_model_1, num_trees=2, ax=ax, rankdir='TB')

plt.show()
import seaborn as sb

heatmap1_data = pd.pivot_table(tsunami_data, values='damage', 

                     index=['tonnage'], 

                     columns='maxTsuVel')

sb.heatmap(heatmap1_data, cmap="YlGnBu")
f,ax = plt.subplots(figsize=(18, 18))

sb.heatmap(tsunami_data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
tsunami_data[tsunami_data["damage"]>0].count()
tsunami_heights = tsunami_data['maxTsuH'].unique().tolist()

tsunami_dataframe = pd.DataFrame(index=tsunami_heights)

for tsunami_height in tsunami_heights:

    tsunami_dataframe.loc[tsunami_height, 'damage 1'] = len(tsunami_data[(tsunami_data["damage"] > 0) & (tsunami_data["maxTsuH"] == tsunami_height)]) / len(tsunami_data[tsunami_data["maxTsuH"] == tsunami_height])

    tsunami_dataframe.loc[tsunami_height, 'damage 2'] = len(tsunami_data[(tsunami_data["damage"] > 1) & (tsunami_data["maxTsuH"] == tsunami_height)]) / len(tsunami_data[tsunami_data["maxTsuH"] == tsunami_height])

    tsunami_dataframe.loc[tsunami_height, 'damage 3'] = len(tsunami_data[(tsunami_data["damage"] > 2) & (tsunami_data["maxTsuH"] == tsunami_height)]) / len(tsunami_data[tsunami_data["maxTsuH"] == tsunami_height])                                   

    tsunami_dataframe.loc[tsunami_height, 'damage 4'] = len(tsunami_data[(tsunami_data["damage"] > 3) & (tsunami_data["maxTsuH"] == tsunami_height)]) / len(tsunami_data[tsunami_data["maxTsuH"] == tsunami_height])

    #tsunami_dataframe.loc[tsunami_height, 'damage4'] = len(tsunami_data[(tsunami_data["damage"] > 4) & (tsunami_data["maxTsuH"] == tsunami_height)]) / len(tsunami_data[tsunami_data["maxTsuH"] == tsunami_height])

tsunami_dataframe['maxTsuH'] = tsunami_dataframe.index

tsunami_dataframe = tsunami_dataframe.melt('maxTsuH', var_name='damage level',  value_name='probability of occurence')

#tsunami_dataframe = tsunami_dataframe.dropna()

sb.lmplot(x="maxTsuH", y="probability of occurence", hue="damage level", data=tsunami_dataframe, logx=True, ci=None, scatter_kws={"s":10})

tsunami_heights = tsunami_data['maxTsuVel'].unique().tolist()

tsunami_dataframe = pd.DataFrame(index=tsunami_heights)

for tsunami_height in tsunami_heights:

    tsunami_dataframe.loc[tsunami_height, 'damage 1'] = len(tsunami_data[(tsunami_data["damage"] > 0) & (tsunami_data["maxTsuVel"] == tsunami_height)]) / len(tsunami_data[tsunami_data["maxTsuVel"] == tsunami_height])

    tsunami_dataframe.loc[tsunami_height, 'damage 2'] = len(tsunami_data[(tsunami_data["damage"] > 1) & (tsunami_data["maxTsuVel"] == tsunami_height)]) / len(tsunami_data[tsunami_data["maxTsuVel"] == tsunami_height])

    tsunami_dataframe.loc[tsunami_height, 'damage 3'] = len(tsunami_data[(tsunami_data["damage"] > 2) & (tsunami_data["maxTsuVel"] == tsunami_height)]) / len(tsunami_data[tsunami_data["maxTsuVel"] == tsunami_height])                                   

    tsunami_dataframe.loc[tsunami_height, 'damage 4'] = len(tsunami_data[(tsunami_data["damage"] > 3) & (tsunami_data["maxTsuVel"] == tsunami_height)]) / len(tsunami_data[tsunami_data["maxTsuVel"] == tsunami_height])

tsunami_dataframe['maxTsuVel'] = tsunami_dataframe.index

tsunami_dataframe = tsunami_dataframe.melt('maxTsuVel', var_name='damage level',  value_name='probability of occurence')

#tsunami_dataframe = tsunami_dataframe.dropna()

sb.lmplot(x="maxTsuVel", y="probability of occurence", hue="damage level", data=tsunami_dataframe, logx=True, ci=None, scatter_kws={"s":10})
