import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
%matplotlib inline
import seaborn as sns
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
print("shape of Training  dataset", train.shape)
print("shape of test dataset", test.shape)
train.head()
test.head()
train.info()
test.info()
train.isnull().sum().sort_values(ascending=False)
dtype_df = train.dtypes.reset_index()
dtype_df.columns = ["Count", "Column Type"]
dtype_df.groupby("Column Type").aggregate('count').reset_index()
unique_df = train.nunique().reset_index()
unique_df.columns = ["col_name", "unique_count"]

unique_df.head(10)
constant_df = unique_df[unique_df["unique_count"]==1]
constant_df.shape
train.drop(columns=constant_df['col_name'],inplace=True)
test.drop(columns=constant_df['col_name'],inplace=True)
print("updated train dataset shape",train.shape)
print("update test dataset shape", test.shape)
train.drop("ID", axis = 1, inplace = True)
y_train=train['target']
train.drop("target", axis = 1, inplace = True)
test.drop("ID", axis = 1, inplace = True)


plt.figure(figsize=(10,6))
sns.distplot(y_train,kde=False, bins=20).set_title('Histogram of target');
plt.xlabel('Target')
plt.ylabel('count');


plt.figure(figsize=(10,6))
sns.distplot(np.log1p(y_train), bins=20,kde=False).set_title('Log histogram of Target');

y_train=np.log1p(y_train)
from sklearn.model_selection import train_test_split
X_train, X_valid, Y_train, Y_valid = train_test_split(train, y_train, test_size=0.05, random_state=0)
X_train.shape

RF_clf=RandomForestRegressor(random_state=42,n_jobs=-1)

RF_clf.fit(X_train,Y_train)
def evaluate(model, features, labels):
    predictions = model.predict(features)
    errors = abs(predictions - labels)
    m = 100 * np.mean(errors / labels)
    accuracy = 100 - m
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy
print('valid Accuracy')
valid_accuracy = evaluate(RF_clf, X_valid, Y_valid)
RF_target=np.expm1(RF_clf.predict(test))

NO_OF_FEATURES=1000
col = pd.DataFrame({'importance': RF_clf.feature_importances_, 'feature': train.columns}).sort_values(
    by=['importance'], ascending=[False])[:NO_OF_FEATURES]['feature'].values
train=train[col]
test=test[col]

train.shape
train["sum"] = train.sum(axis=1)
test["sum"] = test.sum(axis=1)
train["var"] = train.var(axis=1)
test["var"] = test.var(axis=1)
train["median"] = train.median(axis=1)
test["median"] = test.median(axis=1)
train["mean"] = train.mean(axis=1)
test["mean"] = test.mean(axis=1)
train["std"] = train.std(axis=1)
test["std"] = test.std(axis=1)
train["max"] = train.max(axis=1)
test["max"] = test.max(axis=1)
train["min"] =train.min(axis=1)
test["min"] = test.min(axis=1)
train["skew"] = train.skew(axis=1)
test["skew"] = test.skew(axis=1)

from sklearn.model_selection import train_test_split
X_train, X_valid, Y_train, Y_valid = train_test_split(train, y_train, test_size=0.1, random_state=0)
X_train.shape
import lightgbm as lgb


def run_lgb(X_train, Y_train, X_valid, Y_valid, test):
    params = {
        "objective" : "regression",
        "metric" : "rmse",
        "task": "train",
        "boosting type":'dart',
        "num_leaves" :100,
        "learning_rate" : 0.01,
        "bagging_fraction" : 0.8,
        "feature_fraction" : 0.8,
        "bagging_frequency" : 6,
        "bagging_seed" : 42,
        "verbosity" : -1,
        "seed": 42
    }
    
    lgtrain = lgb.Dataset(X_train, label=Y_train)
    lgval = lgb.Dataset(X_valid, label=Y_valid)
    evals_result = {}
    model = lgb.train(params, lgtrain, 5000, 
                      valid_sets=[lgtrain, lgval], 
                      early_stopping_rounds=300, 
                      verbose_eval=100, 
                      evals_result=evals_result)
    
    lgb_prediction = np.expm1(model.predict(test, num_iteration=model.best_iteration))
    return lgb_prediction, model, evals_result
lgb_pred, model, evals_result = run_lgb(X_train, Y_train, X_valid, Y_valid, test)
print("LightGBM Training Completed...")
sub=pd.read_csv('../input/sample_submission.csv')

sub_rf = pd.DataFrame()
sub_rf["target"] = RF_target
sub_rf["ID"] = sub["ID"]
sub_rf.to_csv("sub_rf.csv", index=False)

sub_lgb = pd.DataFrame()
sub_lgb["target"] = lgb_pred
sub_lgb["ID"] = sub["ID"]
sub_lgb.to_csv("sub_lgb.csv", index=False)


sub["target"] = (sub_lgb["target"] + sub_rf['target'] )/2

print(sub.head())
sub.to_csv('sub_lgb_xgb.csv', index=False)
sub.head()
