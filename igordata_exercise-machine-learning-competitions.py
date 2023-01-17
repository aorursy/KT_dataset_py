# Code you have previously used to load data

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

#from sklearn.model_selection import train_test_split



data = pd.read_csv('../input/train.csv')

# Create target object and call it y

y = data.SalePrice

# Create X

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

X = data[features]



# Split into validation and training data

#train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

# To improve accuracy, create a new Random Forest model which you will train on all training data

rf_model_on_full_data = RandomForestRegressor(random_state = 0)



# fit rf_model_on_full_data on all data from the training data

rf_model_on_full_data.fit(X, y)

preds = rf_model_on_full_data.predict(X)

#save the predictions

preds_DF = pd.DataFrame(preds)

preds_DF.to_csv('submission.csv', index=False, header=False)

print("done 1")

from subprocess import check_output

print ("> ls ../working")

print(check_output(["ls", "../working"]).decode("utf8"))

output = pd.read_csv('submission.csv')

output.describe()

#output.head()




print("done 2")