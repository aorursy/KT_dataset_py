import pandas as pd
import numpy
import sklearn
data = pd.read_csv('../input/USA_Housing.csv')
data = pd.DataFrame(data)
train = data.iloc[1:3000,:]
test = data.iloc[3000:,:]
train.shape
test.shape
train.isnull().any()
test.isnull().any()
train.head()
features_column = ['Avg. Area Income','Avg. Area House Age','Avg. Area Number of Rooms','Avg. Area Number of Bedrooms','Area Population']
label_column = ['Price']
x_train = train[features_column]
y_train = train[label_column]
x_test = test[features_column]
y_test = test[label_column]
from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(x_train,y_train)
pred_test = regr.predict(x_test)
print(pred_test)

from sklearn.metrics import mean_squared_error,r2_score
print('mean square error:', mean_squared_error(y_test,pred_test))
print(r2_score(y_test,pred_test))
