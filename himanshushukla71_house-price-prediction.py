import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
train_data=pd.read_csv("../input/all//train.csv")
train_data.head()
train_data.info()
data_train_y=train_data.SalePrice
predictor_parameter=['YearBuilt', 'YearRemodAdd', '1stFlrSF', '2ndFlrSF','LowQualFinSF','GrLivArea', 'FullBath',
                     'HalfBath', 'BedroomAbvGr', 'WoodDeckSF', 'OpenPorchSF','PoolArea']
data_train_x=train_data[predictor_parameter]
data_train_x.info()

prediction_model=RandomForestRegressor()
prediction_model.fit(data_train_x,data_train_y)
test_data=pd.read_csv("../input/all//test.csv")
data_train_x=test_data[predictor_parameter]
#data_train_x.info()
predicted_price=prediction_model.predict(data_train_x)
print(predicted_price)
my_submission = pd.DataFrame({'Id': test_data.Id, 'SalePrice': predicted_price})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)
