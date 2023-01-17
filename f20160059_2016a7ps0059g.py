import numpy as np

import pandas as pd

data = pd.read_csv('../input/train.csv')

labels = data['AveragePrice']
data.head()
from sklearn.model_selection import train_test_split



#TODO

x_train,x_val,y_train,y_val = train_test_split(data,labels,test_size=0,random_state=0)
train=x_train.drop('AveragePrice',axis=1)
testing = pd.read_csv('../input/test.csv')
from xgboost import XGBRegressor
model = XGBRegressor(max_depth = 7,n_estimators=1000)
model.fit(train,y_train)
answ3 = model.predict(testing)
newdf = pd.DataFrame({"id": testing["id"], "AveragePrice":answ3})
columnsTitles=["id","AveragePrice"]

newdf=newdf.reindex(columns=columnsTitles)
newdf.to_csv('resultgdb1000.csv')
