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
import seaborn as sns

import matplotlib.pyplot as plt
placement_data = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')

placement_final = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
placement_data.head(10)
placement_data.shape
placement_data.isnull().sum()
placement_data.info()
sns.countplot(x ='gender',hue ='specialisation',data = placement_data)
sns.heatmap(placement_data.isnull())
sns.countplot(x='ssc_b',data=placement_data)
sns.countplot(y='ssc_p',data=placement_data)
sns.countplot(y='hsc_p',data=placement_data)
sns.countplot(x='hsc_s',data=placement_data)
sns.countplot(x='degree_t',data = placement_data)
sns.countplot(y='degree_p',data = placement_data)
sns.countplot(x='workex',data = placement_data)
sns.countplot(x='etest_p',data = placement_data)
sns.countplot(x='status',data=placement_data)
sns.countplot(x='mba_p',data = placement_data)
sns.countplot(x='salary',data =placement_data)
placement_data.head()
placement_gender=pd.get_dummies(placement_data.gender)
placement_gender.drop('F',axis=1,inplace=True)
placement_gender.rename(columns={'M': 'Gender_Male'},inplace= True)

placement_gender.head()
placement_final.head()
ssc_b = pd.get_dummies(placement_data.ssc_b)
ssc_b.drop('Others',axis=1,inplace= True)
ssc_b.rename(columns={'Central': 'Ssc_b_Central'},inplace= True)

ssc_b.head()
hsc_b = pd.get_dummies(placement_data.hsc_b)
hsc_b.drop('Others',axis=1,inplace= True)

hsc_b.rename(columns={'Central': 'Hsc_b_Central'},inplace= True)

placement_data.head()
drgree = pd.get_dummies(placement_data.degree_t)
drgree.drop('Others',axis=1 , inplace = True)
drgree.rename(columns={'Comm&Mgmt': 'degree_Comm&Mgmt','Sci&Tech':'degree_Sci&Tech'},inplace= True)

drgree.head()
placement_data.head()
hsc_s =pd.get_dummies(placement_data.hsc_s)
hsc_s.drop('Arts',axis=1,inplace = True)
hsc_s.rename(columns={'Commerce': 'Hsc_s_Commerce','Science':'Hsc_s_Science'},inplace= True)

hsc_s.head()
work =pd.get_dummies(placement_data.workex)
work.drop('No',axis = 1,inplace = True)
work.rename(columns={'Yes':'Working'},inplace= True)

work.head()
specialisation = pd.get_dummies(placement_data.specialisation)
specialisation.drop('Mkt&Fin',axis = 1,inplace = True)
specialisation.rename(columns = {'Mkt&HR':'specialisation_Mkt&HR'},inplace = True)

specialisation.head()
placement_final.head()
placement_final.head()
placement_final.drop(['sl_no','gender','hsc_b','hsc_s','degree_t','workex','specialisation'],axis = 1,inplace = True)
placement_final.head()
placement_final_1 = pd.concat([placement_final,drgree,hsc_b,ssc_b,placement_gender,hsc_s,work,specialisation],axis = 1)
placement_final_1.head()
from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor
y = placement_final_1.Working
x= placement_final_1.drop(['salary','status','Working','ssc_p','ssc_b','hsc_p','degree_p','etest_p','mba_p'],axis=1)

x.head()
train_X, val_X, train_y, val_y = train_test_split(x, y, random_state=1)
placement_model = DecisionTreeRegressor(random_state=1)
placement_model.fit(train_X, train_y)
# Make validation predictions and calculate mean absolute error

val_predictions = placement_model.predict(val_X)

val_mae = mean_absolute_error(val_predictions, val_y)

print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))



# Using best value for max_leaf_nodes

iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)

iowa_model.fit(train_X, train_y)

val_predictions = iowa_model.predict(val_X)

val_mae = mean_absolute_error(val_predictions, val_y)

print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))