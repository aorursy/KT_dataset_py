import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

%matplotlib inline

df = pd.read_csv('../input/winequality-red.csv')
df.head()
df.describe()
correlations = df.corr()['quality'].drop('quality')
print(correlations)
_ = correlations.plot(kind='bar')
import seaborn as sns
sns.heatmap(df.corr())
train = df.sample(frac=0.8)
test_and_validation = df.loc[~df.index.isin(train.index)]
validation = test_and_validation.sample(frac=0.5)
test = test_and_validation.loc[~test_and_validation.index.isin(validation.index)]

print(train.shape, validation.shape, test.shape)
def get_features(correlation_threshold):
    abs_corrs = correlations.abs()
    high_correlations = abs_corrs[abs_corrs > correlation_threshold].index.values.tolist()
    return high_correlations
def compare_predictions(predicted, test_df, target_col):
    # Since we have to predict integer values, and the regressor will return float, let's round predicted dataframe
    predicted = predicted.round(0)
    check_df = pd.DataFrame(data=predicted, index=test_df.index, columns=["Predicted "+target_col])
    check_df = pd.concat([check_df, test_df[[target_col]]], axis=1)
    check_df["Error, %"] = np.abs(check_df["Predicted "+target_col]*100/check_df[target_col] - 100)
    check_df['Error, val'] = check_df["Predicted "+target_col] - check_df[target_col]
    return (check_df.sort_index(), check_df["Error, %"].mean())
def evaluate_predictions(model, train_df, test_df, features, target_col):
    train_pred = model.predict(train_df[features])
    train_rmse = mean_squared_error(train_pred, train_df[target_col]) ** 0.5

    test_pred = model.predict(test_df[features])
    test_rmse = mean_squared_error(test_pred, test_df[target_col]) ** 0.5

    print("RMSEs:")
    print(train_rmse, test_rmse)
    
    return test_pred
def lr_model_evaluation(feature_correlation_threshold=0):
    lr = LinearRegression()
    features = get_features(feature_correlation_threshold)
    lr.fit(train[features], train['quality'])
    lr_validation_predictions = evaluate_predictions(lr, train, validation, features, 'quality')
    check_df, avg_error = compare_predictions(lr_validation_predictions, validation, 'quality')
    print("Average validation error:", avg_error)
    return check_df
check = lr_model_evaluation()
thresholds = [x * 0.05 for x in range(1, 8)] #threshold will scale up to 0.4

for thr in thresholds:
    print('For threshold =', thr)
    _ = lr_model_evaluation(thr)
    print()
print(get_features(0.15))
