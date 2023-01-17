# Import packages from numpy, CatBoost, pandas, sklearn

import numpy as np
from catboost import CatBoostRegressor, FeaturesData, Pool
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
# Since we're predicting "SalePrice", we set it to y_train then drop it
# from the X_train dataset.

train_df = pd.read_csv("../input/train.csv", encoding = 'utf-8')
test_df = pd.read_csv("../input/test.csv", encoding = 'utf-8')

y_train = train_df['SalePrice']

X_train = train_df.drop(['SalePrice'], axis=1)
X_test = test_df
# Function to determine if column in dataframe is string.
def is_str(col):
    for i in col:
        if pd.isnull(i):
            continue
        elif isinstance(i, str):
            return True
        else:
            return False
# Splits the mixed dataframe into categorical and numerical features.
def split_features(df):
    cfc = []
    nfc = []
    for column in df:
        if is_str(df[column]):
            cfc.append(column)
        else:
            nfc.append(column)
    return df[cfc], df[nfc]
# Replace all the nan categorical features with the same string "None".
# Replace all the nan numerical features with numpy nanmean.

def preprocess(cat_features, num_features):
    cat_features = cat_features.fillna("None")
    for column in num_features:
        num_features[column].fillna(np.nanmean(num_features[column]), inplace=True)
    return cat_features, num_features
# Apply the "split_features" function on the data.
cat_tmp_train, num_tmp_train = split_features(X_train)
cat_tmp_test, num_tmp_test = split_features(X_test)
# Now to apply the "preprocess" function.
# Getting a "SettingWithCopyWarning" but I usually ignore it.
cat_features_train, num_features_train = preprocess(cat_tmp_train, num_tmp_train)
cat_features_test, num_features_test = preprocess(cat_tmp_test, num_tmp_test)
train_pool = Pool(
    data = FeaturesData(num_feature_data = np.array(num_features_train.values, dtype=np.float32), 
                    cat_feature_data = np.array(cat_features_train.values, dtype=object), 
                    num_feature_names = list(num_features_train.columns.values), 
                    cat_feature_names = list(cat_features_train.columns.values)),
    label =  np.array(y_train, dtype=np.float32)
)
test_pool = Pool(
    data = FeaturesData(num_feature_data = np.array(num_features_test.values, dtype=np.float32), 
                    cat_feature_data = np.array(cat_features_test.values, dtype=object), 
                    num_feature_names = list(num_features_test.columns.values), 
                    cat_feature_names = list(cat_features_test.columns.values))
)
model = CatBoostRegressor(iterations=2000, learning_rate=0.05, depth=5)
# Fit model
model.fit(train_pool)
# Get predictions
preds = model.predict(test_pool)
indx = np.array([i for i in range(1461,2920)])
df = pd.DataFrame({'Id': indx, 'SalePrice': preds}, columns=['Id', 'SalePrice'])
df.to_csv("submission.csv", index=False)
