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
train_df=pd.read_csv("/kaggle/input/breast-cancer-wisconsin-prognostic-data-set/data 2.csv")
train_df
train_df.shape
train_df["diagnosis"].unique()
train_df["diagnosis"]=train_df["diagnosis"].replace(['M','B'],[1,0])

train_df
train_df.diagnosis.unique()
del train_df["id"]
train_df.isna().sum()
train_df.skew()
#train_df["concavity_se"]=np.sqrt(train_df["concavity_se"])
train_df["fractal_dimension_se"]=np.sqrt(train_df["fractal_dimension_se"])
train_df["symmetry_se"]=np.sqrt(train_df["symmetry_se"])
train_df["area_se"]=np.sqrt(train_df["area_se"])
train_df["smoothness_se"]=np.sqrt(train_df["smoothness_se"])
train_df["area_se"]=np.sqrt(train_df["area_se"])
train_df["radius_se"]=np.sqrt(train_df["radius_se"])

train_df["perimeter_se"]=np.sqrt(train_df["perimeter_se"])
train_df["concavity_se"].skew()
train_df.skew()
train_df.columns.values


train_df["Unnamed: 32"].isna().sum()
del train_df["Unnamed: 32"]
train_df.shape
target=train_df["diagnosis"]
del train_df["diagnosis"]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(train_df,target,random_state=0)

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(X_train)
X_train=pd.DataFrame(scaler.transform(X_train))
X_test=pd.DataFrame(scaler.transform(X_test))
scaler2=MinMaxScaler()
scaler2.fit(X_train)
y_train=pd.DataFrame(scaler2.transform(X_train))
y_test=pd.DataFrame(scaler2.transform(X_test))
X_train
X_test
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,y_train)

print(reg.score(X_test,y_test)*100)
print(reg.score(X_train,y_train)*100)

