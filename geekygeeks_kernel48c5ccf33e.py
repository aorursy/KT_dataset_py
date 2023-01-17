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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline


rest = pd.read_csv("../input/train-cav/train.csv")
rest.head()

rest["Open Date"] = pd.to_datetime(rest["Open Date"],format = "%m/%d/%Y")
rest.head()

import datetime
now= datetime.datetime.now()
rest['years_old']= now.year - pd.DatetimeIndex(rest['Open Date']).year

rest.head()

fig, ax = plt.subplots(10,4,figsize=(20, 40))
col = list(rest.columns)[5:]
sns.set_style("whitegrid")
j =0
k=0
p=1
for i in col:
    if(j==4):
        j=0
        k+=1
    plt.title(i)
    sns.set_style("dark")
    sns.scatterplot(rest[i], rest["revenue"], color="g", ax=ax[k,j])
    
    j+=1
    
# to_drop = ['P3', 'P7', 'P8', 'P9', 'P10', 'P12', 'P13', 'P14', 
#            'P15', 'P16', 'P18', 'P26', 'P27', 'P29', 'P31', 'P34', 
#            'P36']
# num_data.drop(to_drop, axis=1, inplace=True)


rest.drop(columns=["P7","P9","P10","P18","P34"], inplace=True)

rest.head()

## Applying log transformed on years_old and revenue and sqrt on p1-p37

num_data = rest.iloc[:,5:]
x = np.log(num_data[["revenue","years_old"]])
y = np.sqrt(num_data.drop(columns=["revenue","years_old"]))

from sklearn.preprocessing import StandardScaler

y_columns= y.columns

scaler= StandardScaler()      
y= scaler.fit_transform(y)

y=pd.DataFrame(y, columns = y_columns)
num_data = pd.concat([y,x],axis=1)

num_data.head()

cat_data = rest.iloc[:,0:5]
cat_data.head()

cat_data.drop(columns=["Id", "Open Date","City"],inplace=True)

F_Type = pd.get_dummies(cat_data["Type"], drop_first=True)
City_Group = pd.get_dummies(cat_data["City Group"], drop_first=True)

cat_data = pd.concat([cat_data, F_Type, City_Group], axis=1)

cat_data.head()

cat_data["Type"].unique()

cat_data.drop(columns=["City Group", "Type"], inplace=True)

cat_data.head()

num_data = pd.concat([cat_data, num_data], axis=1)

num_data.head()

X = num_data.drop(columns=["revenue"])
y = num_data["revenue"]


from sklearn.decomposition import PCA

X.shape

X.head()

pca = PCA(0.95)
pca.fit(X)
X = pca.transform(X)
pca.n_components_

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error

n = pca.n_components_

randomForest = RandomForestRegressor(max_depth=3, n_estimators=n,  
          criterion='mse', random_state=42)

randomForest.fit(X, y)

y_pred = randomForest.predict(X)

mse_randomForest = mean_squared_error(y, y_pred)
rmse_randomForest = np.sqrt(mse_randomForest)
rmse_randomForest

mse_randomForest

## Applying Testing on Testing Dataset

test = pd.read_csv("../input/test-csv/test.csv")
test.head()

test["Open Date"] = pd.to_datetime(test["Open Date"],format = "%m/%d/%Y")
test.head()

import datetime
now= datetime.datetime.now()
test['years_old']= now.year - pd.DatetimeIndex(test['Open Date']).year

test.drop(columns=["P7","P9","P10","P18","P34"], inplace=True)

test.shape

train_num_data = test.iloc[:,5:]
x = np.log(train_num_data[["years_old"]])
y = np.sqrt(train_num_data.drop(columns=["years_old"]))

from sklearn.preprocessing import StandardScaler

y_columns= y.columns

scaler= StandardScaler()      
y= scaler.fit_transform(y)

y=pd.DataFrame(y, columns = y_columns)
train_num_data = pd.concat([y,x],axis=1)

train_num_data.head()

train_cat_data = test.iloc[:,0:5]
train_cat_data.head()

train_cat_data.drop(columns=["Id", "Open Date","City"],inplace=True)

def change(col):
    if col=="MB":
        return "FC"
    else:
        return col
train_cat_data["Type"] = train_cat_data["Type"].apply(change)

train_F_Type = pd.get_dummies(train_cat_data["Type"], drop_first=True)
train_City_Group = pd.get_dummies(train_cat_data["City Group"], drop_first=True)

train_cat_data = pd.concat([train_cat_data, train_F_Type, train_City_Group], axis=1)

train_cat_data.drop(columns=["City Group", "Type"], inplace=True)

train_cat_data.head()

train_num_data = pd.concat([train_cat_data, train_num_data], axis=1)

train_num_data.head()

X_train = train_num_data

X_train.shape

X_train = pca.transform(X_train)

y_train_val = randomForest.predict(X_train) #predict using the GBR model
y_train_val = np.exp(y_train_val) 

y_val_random = np.round(y_train_val, 2)
y_val_random = pd.DataFrame(y_val_random)

y_val_random.index.names = ['Id']
y_val_random.columns = ['Prediction']
pd.DataFrame(y_val_random).to_csv('forecast_random.csv')
