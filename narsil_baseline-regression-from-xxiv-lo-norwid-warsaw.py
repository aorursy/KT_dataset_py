import pandas as pd
import os
print(os.listdir("../input"))
pd.set_option('display.max_columns', 100) 
from sklearn.linear_model import LinearRegression as regression
path_to_data = "../input/"
train = pd.read_csv(path_to_data + "train.csv")
test = pd.read_csv(path_to_data + "test.csv")
train.head()
train.shape, test.shape
#fill na with 0, not always optimal!
train.fillna(0, inplace = True)
test.fillna(0, inplace = True)
features = ["OverallQual", "GrLivArea", "GarageCars", "GarageArea", "TotalBsmtSF",
 "FullBath", "TotRmsAbvGrd", "YearBuilt"]
target = 'SalePrice'
X_train = train[features]
X_test = test[features]
y = train[target] 
nasz_model = regression().fit(X_train, y)
nasz_model
predictions = nasz_model.predict(X_test)
predictions[:10]
predictions_table = pd.read_csv(path_to_data + 'sample_submission.csv')
predictions_table.head()
predictions_table['SalePrice'] = predictions
predictions_table.head()
predictions_table.to_csv('baseline_reg_sub.csv', index = False)
#team name: XXIV_LO_NORWID_xxx
