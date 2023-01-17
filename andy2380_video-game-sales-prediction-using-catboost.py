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
#Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings(action='ignore')
data = pd.read_csv("../input/videogamesales/vgsales.csv")
data.head()
data.describe()
data.nunique()
data.isnull().sum()
data.dropna(inplace=True)
data.info()
data.Year = data.Year.astype(int) # converting the year to int type
plt.figure(figsize=(20,10))
sns.barplot(x='Genre',y='NA_Sales',data = data)
plt.figure(figsize=(20,10))
sns.barplot(x='Genre',y='EU_Sales',data = data)
plt.figure(figsize=(25,5))
sns.lineplot(x='Year',y='Global_Sales',data= data)
plt.figure(figsize=(20,5))
sns.pointplot(x='Platform',y='Global_Sales',data=data)
sns.pairplot(data, kind="reg",
             x_vars=["NA_Sales","EU_Sales","JP_Sales","Other_Sales"],
             y_vars=["Global_Sales"],
             height=4,size=5,aspect=0.9)
train, test = train_test_split(data, test_size=0.2, random_state=42, shuffle=True)
train=train.drop('Name',1)
import catboost as cat
cat_feat = ['Platform', 'Genre', 'Publisher']
features = list(set(train.columns)-set(['Global_Sales']))
target = 'Global_Sales'
model = cat.CatBoostRegressor(random_state=100,cat_features=cat_feat,verbose=0)
model.fit(train[features],train[target])
y_true= pd.DataFrame(data=test[target], columns=['Global_Sales'])
test_temp = test.drop(columns=[target])
y_pred = model.predict(test_temp[features])
from sklearn.metrics import mean_squared_error
from math import sqrt

rmse = sqrt(mean_squared_error(y_true, y_pred))
print(rmse)
import pickle
filename = 'game_model.sav'              # saving the model
pickle.dump(model, open(filename, 'wb'))
loaded_model = pickle.load(open(filename, 'rb'))
test_temp[features].head()
loaded_model.predict(test_temp[features].head()) # predicted the global sales in millions