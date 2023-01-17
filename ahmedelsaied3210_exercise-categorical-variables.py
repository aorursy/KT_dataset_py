# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.ml_intermediate.ex3 import *
print("Setup Complete")
import pandas as pd

# Read the data
data = pd.read_csv('../input/train.csv') 
data.head()
X=data.drop(columns='SalePrice')
X.head()
Y=data['SalePrice']
X = X.select_dtypes(exclude=['object'])
print(X)
X.dtypes
X.shape
X=X.drop(columns=['LotFrontage','GarageYrBlt'])
X.isnull().sum()
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)
X=pd.DataFrame(X)
X.head
from sklearn.model_selection import train_test_split
X=X.fillna(X.mean())
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=.15,random_state=0)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR

# function for comparing different approaches
#model = RandomForestRegressor(n_estimators=1000, random_state=0)
#model=SVR()
model=GradientBoostingRegressor(n_estimators=500,learning_rate=.1)
model.fit(x_train, y_train)
preds = model.predict(x_test)
mean_absolute_error(y_test, preds)
test = pd.read_csv('../input/test.csv')
test.head
test = test.select_dtypes(exclude=['object'])
test_id=test
print(test)
test.shape
test=test.drop(columns=['LotFrontage','GarageYrBlt'])
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
test=sc.fit_transform(test)
test=pd.DataFrame(test)
test.head
test=test.fillna(test.mean())
test.shape
# make predictions which we will submit. 

test_preds = model.predict(test)
test.head()
test.shape
test_preds.shape



# The lines below shows how to save predictions in format used for competition scoring
# Just uncomment them.

output = pd.DataFrame({'Id': test_id.Id,'SalePrice': test_preds})
output.to_csv('submission.csv', index=False)
output.head()
output.shape
