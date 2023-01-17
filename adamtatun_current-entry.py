import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

data = pd.read_csv('../input/train.csv')
data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = data.SalePrice
X =data.drop(['SalePrice'], axis =1).select_dtypes(exclude=['object'])
trainX, testX, train_y, test_y = train_test_split(X.values, y.values, test_size=0.25)
myImputer = SimpleImputer()
trainX = myImputer.fit_transform(trainX)
testX = myImputer.fit_transform(testX)
my_model=XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model.fit(trainX, train_y, early_stopping_rounds=5,
             eval_set=[(testX, test_y)], verbose=False)
test_data = pd.read_csv("../input/test.csv")
test_dataX= test_data.select_dtypes(exclude=['object'])
test_dataX=myImputer.fit_transform(test_dataX)
predictions = my_model.predict(test_dataX)
my_sub= pd.DataFrame({'Id': test_data.Id, 'SalePrice':predictions})
my_sub.to_csv('submission.csv', index=False)