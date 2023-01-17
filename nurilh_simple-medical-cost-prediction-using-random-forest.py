import pandas as pd

# Load dataset
data_path = '../input/insurance.csv'
data = pd.read_csv(data_path)

# See general information about the dataset & 5 sample data points
print(data.info())
data.head(5)
from sklearn.preprocessing import LabelEncoder

# Label encoder initialization
le_sex = LabelEncoder()
le_smoker = LabelEncoder()
le_region = LabelEncoder()

# Label encoding
data['sex_encoded'] = le_sex.fit_transform(data.sex)
data['smoker_encoded'] = le_smoker.fit_transform(data.smoker)
data['region_encoded'] = le_region.fit_transform(data.region)

# See the encoding mapping 
# (categorical value encoded by the index)
print('sex column encoding mapping : %s' % list(le_sex.classes_))
print('smoker column encoding mapping : %s' % list(le_smoker.classes_))
print('region column encoding mapping : %s' % list(le_region.classes_))

# See label encoding result
data.head(5)
from sklearn.preprocessing import OneHotEncoder

# One hot encoder initialization
ohe_region = OneHotEncoder()

# One hot encoding (OHE) to array
arr_ohe_region = ohe_region.fit_transform(data.region_encoded.values.reshape(-1,1)).toarray()

# Convert array OHE to dataframe and append to existing dataframe
dfOneHot = pd.DataFrame(arr_ohe_region, columns=['region_'+str(i) for i in range(arr_ohe_region.shape[1])])
data = pd.concat([data, dfOneHot], axis=1)

# See the preprocessing result
data.head(5)
# Drop categorical features
preprocessed_data = data.drop(['sex','smoker','region',
                               'region_encoded'], axis=1)

# See the preprocessing final result
preprocessed_data.head(5)
from sklearn.model_selection import train_test_split

# Split the dataset to training and testing
train, test = train_test_split(preprocessed_data, test_size=0.2)

# Split the feature and the target
train_y = train.charges.values
train_x = train.drop(columns=['charges']).values
test_y = test.charges.values
test_x = test.drop(columns=['charges']).values

# See the size of training and testing
print('Training features : ', train_x.shape)
print('Training target : ', train_y.shape)
print('Testing features : ', test_x.shape)
print('Testing target : ', test_y.shape)
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor

# Building linear regression model
lr_model = linear_model.LinearRegression()
lr_model.fit(train_x, train_y)

# Building Random Forest model
rf_model = RandomForestRegressor()
rf_model.fit(train_x, train_y)

# Make prediction
lr_predict = lr_model.predict(test_x)
rf_predict = rf_model.predict(test_x)

# Sample the prediction
sample_id = 7
print('Actual Charges : %.2f' % test_y[sample_id])
print('Linear Regression Prediction : %.2f' % lr_predict[sample_id])
print('Random Forest Prediction : %.2f' % rf_predict[sample_id])
from sklearn.metrics import mean_squared_error, r2_score
import math

# Evaluate prediction model using MSE
lr_mse = mean_squared_error(test_y, lr_predict)
print('MSE-Linear Regression : %.2f (square-rooted)' % math.sqrt(lr_mse))
rf_mse = mean_squared_error(test_y, rf_predict)
print('MSE-Random Forest : %.2f (square-rooted)' % math.sqrt(rf_mse))

# Evaluate prediction model using R2-Score
lr_r2 = r2_score(test_y, lr_predict)
print('R2-Linear Regression : %.2f' % lr_r2)
rf_r2 = r2_score(test_y, rf_predict)
print('R2-Random Forest : %.2f' % rf_r2)