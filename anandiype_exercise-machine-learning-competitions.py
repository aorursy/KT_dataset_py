import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split



#Reading training data

train_data_file_path = '../input/train.csv'

train_data = pd.read_csv(train_data_file_path)



#Set target(train_y) and features and train_X

train_y = train_data.SalePrice

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

train_X = train_data[features]



#Setting machine learning model - Random forest and training(fit)

ml_model = RandomForestRegressor(random_state=1)

ml_model.fit(train_X,train_y)



#Reading test data

test_data_file_path = '../input/test.csv'

test_data = pd.read_csv(test_data_file_path)



#Set test_y, test_X

test_X = test_data[features]



#Make pedictions on test data (test_X)

predictions = ml_model.predict(test_X)



#Prepare predictions for submission 

output = pd.DataFrame({'Id': test_data.Id,

                       'SalePrice': predictions})

output.to_csv('submission.csv', index=False)