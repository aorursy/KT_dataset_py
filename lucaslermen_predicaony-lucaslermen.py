import numpy as np

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor



import os

print(os.listdir("../input"))
train_data_path = '../input/train.csv'



train_data = pd.read_csv(train_data_path)



train_data = train_data.apply(pd.to_numeric, errors='coerce')



train_data = train_data.replace(np.NaN, 0)



train_data
test_data_path = '../input/test.csv'

test_data = pd.read_csv(test_data_path)



valid_data_path = '../input/valid.csv'

valid_data = pd.read_csv(valid_data_path)



merged = pd.concat([test_data, valid_data])

merged.to_csv('merged.csv', index=None)



testValid_data_path = 'merged.csv'

testValid_data = pd.read_csv(testValid_data_path)



testValid_data
y = train_data.sale_price

# Create X

features = ['gross_square_feet', 'year_built','land_square_feet']

X = train_data[features]



# Split into validation and training data

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)



# Define the model. Set random_state to 1

rf_model = RandomForestRegressor(random_state=1)

rf_model.fit(train_X, train_y)

rf_val_predictions = rf_model.predict(val_X)
testValid_data = testValid_data.apply(pd.to_numeric, errors='coerce')



testValid_data = testValid_data.replace(np.NaN, 0)



# make predictions which we will submit. 

test_preds = rf_model.predict(testValid_data[features])



output = pd.DataFrame({'sale_id': testValid_data.sale_id,

                       'sale_price': test_preds})

output.to_csv('submission.csv', index=False)