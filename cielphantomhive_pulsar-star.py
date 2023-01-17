# Code you have previously used to load data

import pandas as pd

from sklearn.tree import DecisionTreeRegressor



# Path of the file to read

path_to_csv = '../input/predicting-a-pulsar-star/pulsar_stars.csv'





df = pd.read_csv(path_to_csv)
df.describe()
df.info()
X = df.iloc[:,:-1]

y = df.iloc[:,-1]
X.shape
y.shape
print(max(y.tolist()),min(y.tolist()))
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split



train_X,val_X,train_y,val_y = train_test_split(X,y,test_size = 0.2,random_state=1)

model = RandomForestClassifier(random_state=1,n_estimators=100, max_leaf_nodes = None)

model.fit(train_X, train_y)
val_pred = model.predict(val_X)

from sklearn.metrics import accuracy_score



score= accuracy_score(val_y,val_pred)

print('the accuracy score is - ',score)