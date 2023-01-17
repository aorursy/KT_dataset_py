# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
%matplotlib inline
from matplotlib import pyplot as plt
import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from statsmodels.api import OLS
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train= pd.read_csv('../input/black-friday-sales-prediction/train.csv')
test= pd.read_csv('../input/black-friday-sales-prediction/test.csv')
sample= pd.read_csv('../input/samplesubmissionfile/sample_submission_V9Inaty.csv')
train.shape, test.shape
train.head()
sns.countplot(data=train, x="Gender", palette='husl')
train["Gender"].value_counts()
# Pie chart
plt.figure(figsize=(10,10))
labels = ['M', 'F']
counts = [414259,135809]
explode = (0, 0.1)

fig1, ax1 = plt.subplots()
ax1.pie(counts,explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.suptitle('Gender')
plt.show()
train["Stay_In_Current_City_Years"].value_counts()
# Pie chart
plt.figure(figsize=(10,10))
labels = ['0', '1', '2', '3', '4+']
counts = [74398,193821,101838,95285,84726]
explode = (0, 0.1,0,0,0)

fig1, ax1 = plt.subplots()
ax1.pie(counts,explode= explode,labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.suptitle('Stay in Current City( Years)')
plt.show()
sns.countplot(data=train, x="Marital_Status", hue="Gender")
sns.barplot(train["Marital_Status"], train["Purchase"], hue=train["Gender"])
sns.boxplot(data=train, x="Gender", y="Purchase")
sns.boxplot(data=train, x="Age", y="Purchase")
train['Age'].value_counts()
train["Age"].unique()
# Pie chart
plt.figure(figsize=(10,10))
labels = ['0-17', '55+', '26-35', '46-50', '51-55', '36-45', '18-25']
counts = [15102,21504,38501,45701,99660,110013,219587]
#explode = (0, 0.5,0,0.2,0.3,0.1,0.1)

fig1, ax1 = plt.subplots()
ax1.pie(counts,labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.suptitle('Ages representation in pie chart')
plt.show()
sns.countplot(train["City_Category"])
for col in train.columns:
    print('{}: {}'.format(col,train[col].unique()))
from sklearn.preprocessing import LabelEncoder
train['User_ID'] = train['User_ID'] - 1000000
le = LabelEncoder()
train['User_ID'] = le.fit_transform(train['User_ID'])
train['Product_ID'] = train['Product_ID'].str.replace('P00', '')
ss = StandardScaler()
train['Product_ID'] = ss.fit_transform(train['Product_ID'].values.reshape(-1, 1))
train['Gender']= train['Gender'].replace({'F':0,'M':1})

train['Age']= train['Age'].replace({'0-17':17,'18-25':25,
                                    '26-35':35,'36-45':45,
                                    '46-50':50,'51-55':55,'55+':60 })

train['Stay_In_Current_City_Years']= train['Stay_In_Current_City_Years'].replace({'4+':4})
train['City_Category']= train['City_Category'].replace({'A':1,'B':2,'C':3})
train['Gender']= train['Gender'].astype('int64')
train['Age']= train['Age'].astype('int64')
train['City_Category']= train['City_Category'].astype('category')
train['Stay_In_Current_City_Years']= train['Stay_In_Current_City_Years'].astype('int64')
plt.figure(figsize=(12,12))
corr= train.corr()
sns.heatmap(corr, linewidths=1.5, annot= True)
plt.show()
train.head()
train.isna().sum()
train["Product_Category_2"] = train["Product_Category_2"].fillna(0)
train["Product_Category_3"] = train["Product_Category_3"].fillna(0)
train.isna().sum()
train["Product_Category_2"]= train["Product_Category_2"].astype('int64')
train["Product_Category_3"]= train["Product_Category_3"].astype('int64')
train.dtypes
X= train.drop(["Purchase"], axis=1)
y= train["Purchase"]
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.3, random_state=1)
steps=[("norm",StandardScaler()),("x",XGBRegressor(n_estimators=300,learning_rate=0.4, 
                                                   max_depth=6, min_child_weight=40, seed=0))]
xgb=Pipeline(steps=steps)
xgb.fit(X_train, y_train)
xgb_preds= xgb.predict(X_test)
print('RMSE of XGB:', np.sqrt(mean_squared_error(xgb_preds, y_test)))
sample.head()
sample.dtypes
sample.shape
test.shape
test.head()
id_data= test[["User_ID","Product_ID"]]
id_data.head()
test['User_ID'] = test['User_ID'] - 1000000
le = LabelEncoder()
test['User_ID'] = le.fit_transform(test['User_ID'])
test['Product_ID'] = test['Product_ID'].str.replace('P00', '')
ss = StandardScaler()
test['Product_ID'] = ss.fit_transform(test['Product_ID'].values.reshape(-1, 1))
test['Gender']= test['Gender'].replace({'F':0,'M':1})

test['Age']= test['Age'].replace({'0-17':17,'18-25':25,
                                    '26-35':35,'36-45':45,
                                    '46-50':50,'51-55':55,'55+':60})

test['Stay_In_Current_City_Years']= test['Stay_In_Current_City_Years'].replace({'4+':4})
test['City_Category']= test['City_Category'].replace({'A':1,'B':2,'C':3})
test['Gender']= test['Gender'].astype('int64')
test['Age']= test['Age'].astype('int64')
test['City_Category']= test['City_Category'].astype('category')
test['Stay_In_Current_City_Years']= test['Stay_In_Current_City_Years'].astype('int64')
test["Product_Category_2"] = test["Product_Category_2"].fillna(0)
test["Product_Category_3"] = test["Product_Category_3"].fillna(0)
test.isna().sum()
test["Product_Category_2"]= test["Product_Category_2"].astype('int64')
test["Product_Category_3"]= test["Product_Category_3"].astype('int64')
test.dtypes
test_preds= xgb.predict(test)
len(test_preds)
test_preds
preds_df= pd.DataFrame(test_preds, columns=["Purchase"])
preds_df["User_ID"]= id_data["User_ID"]
preds_df["Product_ID"]= id_data["Product_ID"]
preds_df.head()
display(sample.dtypes)
display(preds_df.dtypes)
preds_df["Purchase"]= preds_df["Purchase"].astype('int64')
preds_df.head()
preds_df.dtypes
preds_df.shape
preds_df.to_csv('/kaggle/working/Submission(XGB).csv', index=False)
