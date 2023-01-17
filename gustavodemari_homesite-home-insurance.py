# Loading libraries
import pandas as pd
import numpy as np
import xgboost as xgb
# Loading data
train = pd.read_csv("../input/train.csv", parse_dates=['Original_Quote_Date'])
test = pd.read_csv("../input/test.csv", parse_dates=['Original_Quote_Date'])
print(train.shape)
print(test.shape)
train.head(n=5)
test.head(n=5)
train.info()
test.info()
#print(train.columns) # X = train.iloc[:,2:], y = train.iloc[:, 1]
#print(test.columns) # X = test.iloc[:, 2:]
x_cols = train.columns
x_cols = x_cols.drop(['QuoteNumber','QuoteConversion_Flag'])
print(x_cols)
# Stratify sampling 50% of the dataset
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
     train[x_cols], train['QuoteConversion_Flag'], test_size=0.5, stratify=train['QuoteConversion_Flag'], random_state=42)
train_nulls = X_train.isnull().sum()
train_null_cols = list(train_nulls[train_nulls > 0].index)
print(train_null_cols)
test_nulls = test.isnull().sum()
test_null_cols = list(test_nulls[test_nulls > 0].index)
print(test_null_cols)
# Filling missing values with mode
X_train[train_null_cols] = X_train[train_null_cols].fillna(train[train_null_cols].mode().iloc[0])
# Filling missing values with mode
test[test_null_cols] = test[test_null_cols].fillna(test[test_null_cols].mode().iloc[0])
def feature_engineering(data):
    data['Original_Quote_Date_year'] = data['Original_Quote_Date'].dt.year
    data['Original_Quote_Date_month'] = data['Original_Quote_Date'].dt.month
    data['Original_Quote_Date_weekday'] = data['Original_Quote_Date'].dt.weekday
    return data
X_train = feature_engineering(X_train)
test = feature_engineering(test)
X_train = X_train.drop('Original_Quote_Date', axis=1)
test = test.drop('Original_Quote_Date', axis=1)
x_cols = X_train.columns
print(x_cols)
# Encoding object columns
from sklearn.preprocessing import LabelEncoder

for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(np.unique(list(X_train[c].values) + list(test[c].values)))
        X_train[c] = lbl.transform(list(X_train[c].values))
        test[c] = lbl.transform(list(test[c].values))
# Verifying data after encoding
train.describe().transpose()
from sklearn.grid_search import RandomizedSearchCV
clf = xgb.XGBClassifier()
params = {
 "learning_rate": [0.1, 0.001],
 "max_depth": [3, 6],
 "n_estimators": [100, 200, 300],
 "objective": ["binary:logistic"]
 }
random_search = RandomizedSearchCV(estimator=clf, 
                                   param_distributions=params, 
                                   scoring='roc_auc',
                                   refit=True,
                                   n_iter=3,
                                   verbose=1)
#clf.get_params()
random_search
X_train = X_train.values
y_train = y_train.values
random_search.fit(X_train, y_train)
print(random_search.best_params_)
print(random_search.best_score_)
print(random_search.best_estimator_)
# Generating predictions
predictions = random_search.predict(test[x_cols])
# Generating submission dataframe
submission = pd.DataFrame()
submission['QuoteNumber'] = test['QuoteNumber']
submission['QuoteConversion_Flag'] = predictions
import time
PREDICTIONS_FILENAME_PREFIX = 'predictions_'
PREDICTIONS_FILENAME = PREDICTIONS_FILENAME_PREFIX + time.strftime('%Y%m%d-%H%M%S') + '.csv'

print(PREDICTIONS_FILENAME)

submission.to_csv(PREDICTIONS_FILENAME, index = False)