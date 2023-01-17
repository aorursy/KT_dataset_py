# Code you have previously used to load data

import pandas as pd

from sklearn.tree import DecisionTreeRegressor



# Path of the file to read

mobile_train_path = '../input/mobile-price-classification/train.csv'

mobile_test_path = '../input/mobile-price-classification/test.csv'



train_df = pd.read_csv(mobile_train_path)

test_df = pd.read_csv(mobile_test_path)
max(train_df.iloc[:,-1].tolist())
min(train_df.iloc[:,-1].tolist())
test_df.info()
print(train_df.columns,test_df.columns)
X = train_df.iloc[:,:-1]

y = train_df.iloc[:,-1]
X.columns
y.shape
test_X = test_df.iloc[:,1:]
test_X.shape
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split



train_X,val_X,train_y,val_y = train_test_split(X,y,test_size = 0.2,random_state=1)

model = RandomForestClassifier(random_state=1,n_estimators=100, max_leaf_nodes = None)

model.fit(train_X, train_y)

val_pred = model.predict(val_X)

from sklearn.metrics import accuracy_score



score= accuracy_score(val_y,val_pred)

print('the accuracy score is - ',score)
rf_model = RandomForestClassifier(random_state=1,n_estimators=100, max_leaf_nodes = None)

rf_model.fit(X,y)
test_y = rf_model.predict(test_X)

import numpy as np

print(np.amin(test_y))
res_df = pd.DataFrame({"Id": test_df.id, "price_range": test_y})

res_df.to_csv('submission.csv', index=False)