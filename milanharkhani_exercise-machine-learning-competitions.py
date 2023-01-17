# Code you have previously used to load data
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor




# Path of the file to read. We changed the directory structure to simplify submitting to a competition
iowa_file_path = '../input/train.csv'

home_data = pd.read_csv(iowa_file_path)
target = home_data.SalePrice
home_data_X=home_data.drop(['SalePrice'],axis=1)
home_data_X.head(15)
#inspactiong features
numeric_data_cols = [col for col in home_data_X.columns if home_data_X[col].dtypes != 'object']
df_numeric_features=home_data_X[numeric_data_cols]


numeric_missing_val_cols = [col for col in df_numeric_features.columns if df_numeric_features[col].isnull().any() ]

from sklearn.preprocessing import Imputer
mean_imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)
median_imputer= Imputer(missing_values='NaN',strategy='median',axis=0)
df_numeric_features['GarageYrBlt']=median_imputer.fit_transform(df_numeric_features[['GarageYrBlt']]).ravel()
print('missing val columns',numeric_missing_val_cols)
df_numeric_features_imputed=pd.DataFrame(mean_imputer.fit_transform(df_numeric_features))
df_numeric_features_imputed.columns = df_numeric_features.columns
df_numeric_features_imputed.index = df_numeric_features.index
df_numeric_features_imputed.head(15)

#inspecting non-numeric cols
df_non_numeric_featurs = home_data.drop(numeric_data_cols,axis=1)
#checking cardinality
low_cardnality_cols=[col for col in df_non_numeric_featurs.columns if df_non_numeric_featurs[col].nunique()<10 ]
df_non_numeric_featurs_with_low_card=df_non_numeric_featurs[low_cardnality_cols]
df_final = pd.concat([df_numeric_features_imputed,df_non_numeric_featurs_with_low_card],axis=1,join_axes=[df_numeric_features_imputed.index])
df_final_encoder = pd.get_dummies(df_final)
df_final.isnull().any()
train_X,test_X,train_y,test_y=train_test_split(df_final_encoder,target,test_size=0.25,random_state=1)
random_forest_model=RandomForestRegressor(random_state=1)
random_forest_model.fit(train_X,train_y)
Rf_predictions=random_forest_model.predict(test_X)
print('MAE Score rf:',mean_absolute_error(Rf_predictions,test_y))

from xgboost import XGBRegressor
xg_model=XGBRegressor(n_estimators=10000,learning_rate=0.05)
xg_model.fit(train_X,train_y)
xg_prediction = xg_model.predict(test_X)
print('XG boost MAE:',mean_absolute_error(xg_prediction,test_y))


#train model on full data
xg_model.fit(df_final_encoder,target)
#prepare test data
#using this model for prediction
test_data_path = '../input/test.csv'
test_data=pd.read_csv(test_data_path)
final_cols=df_final.columns
df_test_data=test_data[final_cols]
df_test_data['GarageYrBlt']=median_imputer.fit_transform(df_test_data[['GarageYrBlt']]).ravel()
df_test_data_numeric_features=df_test_data[numeric_data_cols]
df_test_data_numeric_features_imputed=pd.DataFrame(mean_imputer.fit_transform(df_test_data_numeric_features))
df_test_data_numeric_features_imputed.columns=df_test_data_numeric_features.columns
df_test_data_numeric_features_imputed.index = df_test_data_numeric_features.index
df_test_data_non_numeric=df_test_data[low_cardnality_cols]
df_final_test_data=pd.concat([df_test_data_numeric_features_imputed,df_test_data_non_numeric],axis=1,join_axes=[df_test_data_numeric_features_imputed.index])
df_final_test_data_encoded=pd.get_dummies(df_final_test_data)

final_train,final_test=df_final_encoder.align(df_final_test_data_encoded,axis=1,join='left')
final_test.Id=final_test.Id.astype('int64')
test_prediction=xg_model.predict(final_test)
output = pd.DataFrame({'Id': final_test.Id,
                       'SalePrice': test_prediction})
output.to_csv('submission.csv', index=False)

