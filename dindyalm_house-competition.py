# Code you have previously used to load data
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model


# Path of the file to read
iowa_file_path = '../input/house-prices-advanced-regression-techniques/train.csv'
iowa_file_path_test = '../input/house-prices-advanced-regression-techniques/test.csv'
train = pd.read_csv(iowa_file_path)
test = pd.read_csv(iowa_file_path_test)

data = train.select_dtypes(include=[np.number]).interpolate().dropna()
sum(data.isnull().sum() != 0)


y = np.log(train.SalePrice)
X = data.drop(['SalePrice', 'Id'], axis=1)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.33)

lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)

predictions = model.predict(X_test)
from sklearn.metrics import mean_squared_error
#print ('RMSE is: \n', mean_squared_error(y_test, predictions))

for i in range (-2, 3):
    alpha = 10**i
    rm = linear_model.Ridge(alpha=alpha)
    ridge_model = rm.fit(X_train, y_train)
    preds_ridge = ridge_model.predict(X_test)
    
submission = pd.DataFrame()
submission['Id'] = test.Id

feats = test.select_dtypes(
        include=[np.number]).drop(['Id'], axis=1).interpolate()

predictions = model.predict(feats)

final_predictions = np.exp(predictions)


submission['SalePrice'] = final_predictions
submission.to_csv('submission1.csv', index=False)

