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
import seaborn as sns
import matplotlib.pyplot as plt
df= pd.read_csv("../input/housesalesprediction/kc_house_data.csv")
df.info()
df.drop(['id','date'],axis=1,inplace=True)
df.isnull().sum()
sns.countplot('bedrooms',data=df)
df=df[df.bedrooms >0]
df=df[df.bedrooms <7]

plt.figure(figsize=(15,5))
sns.countplot('bathrooms',data=df)
df['bathrooms']=df['bathrooms'].apply(np.round)
sns.countplot('bathrooms',data=df)
df=df[df.bathrooms >0]
df=df[df.bathrooms <5]

sns.countplot('floors',data=df)
df['floors']=df['floors'].apply(np.round)
sns.countplot('floors',data=df)
df=df[df.floors <4]

sns.countplot('waterfront',data=df)
sns.countplot('view',data=df)
df.drop('view', axis=1, inplace=True)

sns.countplot('condition',data=df)
sns.countplot('grade',data=df)
plt.figure(figsize=(25,12))
sns.heatmap(df.corr(), annot=True)
x=df.drop('price',axis=1)
y=df['price']
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor()

model.fit(x,y)
model.score(X_test,y_test)