from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score,recall_score
import pandas_profiling 

# this may need to be installed separately with
# !pip install category-encoders
import category_encoders as ce

# python general
import pandas as pd
import numpy as np
from collections import OrderedDict

#scikit learn

import sklearn
from sklearn.base import clone

# model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# ML models
from sklearn import tree
from sklearn.linear_model import LinearRegression 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# error metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
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
df = pd.read_csv('/kaggle/input/ihsmarkit-hackathon-june2020/train_data.csv',index_col='vehicle_id')
df['date'] = pd.to_datetime(df['date'])
df = df.drop(columns=['Length', 'CO2','Plugin','Global_Sales_Sub-Segment'])
pandas_profiling.ProfileReport(df)
import matplotlib.pyplot as plt
import seaborn as sns
sns.heatmap(df.corr(),annot=True,cmap='RdYlGn',linewidths=0.2)
fig=plt.gcf()
fig.set_size_inches(20,10)
plt.show()
categorical_features = [
    'Body_Type',
    'Driven_Wheels',
    'Brand',
    'Nameplate',
    'Transmission',
    'Turbo',
    'Fuel_Type',
    'PropSysDesign',
    'Registration_Type',
    'country_name'
]


numeric_features = [
    'Generation_Year',
    'Height',
    'Width',
    'Engine_KW',
    'No_of_Gears',
    'Curb_Weight',
    'Fuel_cons_combined',
    'year'
]

target =['Price_USD']
df_categorical = df[categorical_features]
df_numerical = df[numeric_features]
df_target = df[target]
df_categorical = pd.get_dummies(df_categorical,prefix=categorical_features)
df_categorical = pd.DataFrame(df_categorical)
df_categorical.head()
from sklearn import preprocessing

x = df_numerical.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df_numerical = pd.DataFrame(x_scaled,columns=numeric_features)

df_numerical.head()
df_dmy = pd.concat([df_categorical,df_numerical,df_target],axis=1,verify_integrity=True)
X_train,X_test,y_train,y_test = train_test_split(df_dmy.drop('Price_USD',axis=1),
                                                 df_dmy['Price_USD'],test_size=0.2,
                                                 random_state=200)
print(X_test.shape)
print(y_test.shape)
LogReg=LinearRegression()
LogReg.fit(X_train,y_train)
y_pred = LogReg.predict(X_test)
print(y_pred)
accuracy = LogReg.score(X_test,y_test)
print(accuracy)
#plt.scatter(X_test, y_test,  color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)