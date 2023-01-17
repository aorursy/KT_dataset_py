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
audi=pd.read_csv("../input/used-car-dataset-ford-and-mercedes/audi.csv")
bmw=pd.read_csv("../input/used-car-dataset-ford-and-mercedes/bmw.csv")
cclass=pd.read_csv("../input/used-car-dataset-ford-and-mercedes/cclass.csv")
focus=pd.read_csv('../input/used-car-dataset-ford-and-mercedes/focus.csv')
ford=pd.read_csv('../input/used-car-dataset-ford-and-mercedes/ford.csv')
hyundi=pd.read_csv("../input/used-car-dataset-ford-and-mercedes/hyundi.csv")
merc=pd.read_csv("../input/used-car-dataset-ford-and-mercedes/merc.csv")
skoda=pd.read_csv("../input/used-car-dataset-ford-and-mercedes/skoda.csv")
toyota=pd.read_csv("../input/used-car-dataset-ford-and-mercedes/toyota.csv")
vauxhall=pd.read_csv("../input/used-car-dataset-ford-and-mercedes/vauxhall.csv")
vw=pd.read_csv("../input/used-car-dataset-ford-and-mercedes/vw.csv")

audi.info()
audi.isnull().sum()
bmw.isnull().sum()
cclass.isnull().sum()
focus.isnull().sum()
ford.isnull().sum()
hyundi.isnull().sum()
merc.isnull().sum()
skoda.isnull().sum()
toyota.isnull().sum()
vauxhall.isnull().sum()
vw.isnull().sum()
audi['manufacturer'] = 'Audi'

bmw['manufacturer'] = 'BMW'

cclass['manufacturer'] = 'Mercedes-Benz'
cclass['tax'] = 0

focus['manufacturer'] = 'Ford'
focus['tax'] = 0

hyundi['manufacturer'] = 'Hyundi Motor'
hyundi = hyundi.rename(columns={"tax(£)": "tax"})

merc['manufacturer'] = 'Mercedes'

skoda['manufacturer'] = 'Skoda'
toyota['manufacturer'] = 'Toyota'

vauxhall['manufacturer'] = 'Vauxhall'

vw['manufacturer'] = 'Volkswagen'



audi.isnull().sum()
focus.isnull().sum()
cars = pd.concat([audi, bmw, cclass, focus, hyundi, merc, skoda, toyota, vauxhall, vw], ignore_index=True)
cars.describe()
cars.head()
cars['tax']
list(cars['tax'])
cars.shape
cars
_cars=pd.get_dummies(cars,columns=['model', 'transmission','fuelType','manufacturer'])
_cars
from sklearn.model_selection import train_test_split
X = _cars.drop(['price'],axis=1)
y = _cars['price']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
import xgboost
model = xgboost.XGBRegressor(colsample_bytree=0.4,
                 gamma=0,                 
                 learning_rate=0.07,
                 max_depth=3,
                 min_child_weight=1.5,
                 n_estimators=10000,                                                                    
                 reg_alpha=0.75,
                 reg_lambda=0.45,
                 subsample=0.6,
                 seed=42) 
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
from sklearn.metrics import r2_score
accuracy=r2_score(y_pred,y_test)
# accuracy of model
accuracy*100
