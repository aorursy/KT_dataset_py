import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Define feature column categories by column type
categorical_cols = [col for col in df.columns if df[col].dtype == 'object']
numeric_cols = [col for col in df.columns if df[col].dtype != 'object'] 

# Remove the target column (SalePrice) from our feature list
numeric_cols.remove('SalePrice')

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('scaler', MinMaxScaler()),
    ('imputer', SimpleImputer(strategy='mean'))

])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore')) # ignore set so new categories in validation set won't trigger an error post test set fit
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('numeric', numerical_transformer, numeric_cols),
        ('categorical', categorical_transformer, categorical_cols)
    ])

# Grab target as y, remove target from X
train_test = df.copy()
y = train_test.SalePrice
X = train_test.drop(columns=['SalePrice'])

# Split into train, test
train_X, val_X, train_y, val_y = train_test_split(X, y, train_size=0.8, random_state = 17)

# Fit the preprocessor using the training data
train_X_cleaned = preprocessor.fit_transform(train_X)

# Run the validation set (and all future sets) through the transform without fitting again, or else you'll end up with a different pipeline!
val_X_cleaned = preprocessor.transform(val_X)

# Printing shapes of the processed arrays to make sure we haven't gone too far wrong.
print ("Initial train_X shape: ",train_X.shape)
print ("Initial val_X shape: ",val_X.shape)
print ("Processed train_X shape: ", train_X_cleaned.shape)
print ("Processed val_X shape: ",val_X_cleaned.shape)
import xgboost as xgb

# Create an initial parameter list, which we'll tune as we go.
params = {
    'max_depth':6,
    'min_child_weight': 1,
    'eta':.3,
    'subsample': 1,
    'colsample_bytree': 1,
    'objective':'reg:squarederror',
    'eval_metric':'mae'
}

# Set num_boost_rounds for future use
NUM_BOOST_ROUNDS=999

# Take our inputs and format them as DMatrices
dtrain = xgb.DMatrix(train_X_cleaned, label=train_y)
dtest = xgb.DMatrix(val_X_cleaned, label=val_y)

# Train our first XGB model!
xbg_model = xgb.train(
    params,
    dtrain,
    num_boost_round=NUM_BOOST_ROUNDS,
    evals=[(dtest, "Test")],
    early_stopping_rounds=10
)
  
print("Best MAE: {:.2f}, found at round {}".format(
                 xbg_model.best_score,
                 xbg_model.best_iteration))
# Calculate cross validation
cv_results = xgb.cv(
    params,
    dtrain,
    num_boost_round=NUM_BOOST_ROUNDS,
    seed=42,
    nfold=5,
    metrics={'mae'},
    early_stopping_rounds=10
)

# Plot cross validation results
plt.plot(cv_results['train-mae-mean'], label='train mae')
plt.plot(cv_results['test-mae-mean'], label='test mae')
plt.title("XGB Cross Validation Error")
plt.xlabel('Round Number')
plt.ylabel('Mean Absoulte Error (MAE)')
plt.legend()
plt.show()
# Tune max_depth and min_child_weight
gridsearch_params = [
    (max_depth, min_child_weight)
    for max_depth in range(3,8)
    for min_child_weight in range(1,6)
]

# Define initial best params and MAE
min_mae = float("Inf")
best_params = None
for max_depth, min_child_weight in gridsearch_params:
    print("CV with max_depth={}, min_child_weight={}".format(
                             max_depth,
                             min_child_weight))
    # Update our parameters
    params['max_depth'] = max_depth
    params['min_child_weight'] = min_child_weight
    # Run CV
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=NUM_BOOST_ROUNDS,
        seed=42,
        nfold=5,
        metrics={'mae'},
        early_stopping_rounds=10
    )
    # Update best MAE
    mean_mae = cv_results['test-mae-mean'].min()
    boost_rounds = cv_results['test-mae-mean'].idxmin()
    print("\tMAE {} for {} rounds\n".format(mean_mae, boost_rounds))
    if mean_mae < min_mae:
        min_mae = mean_mae
        best_params = (max_depth,min_child_weight)
print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))
params['max_depth'] = 5
params['min_child_weight'] = 1
# Next, tune subsample and colsample
gridsearch_params = [
    (subsample, colsample)
    for subsample in [i/10. for i in range(7,11)]
    for colsample in [i/10. for i in range(7,11)]
]

min_mae = float("Inf")
best_params = None
# We start by the largest values and go down to the smallest
for subsample, colsample in reversed(gridsearch_params):
    print("CV with subsample={}, colsample={}".format(
                             subsample,
                             colsample))
    # We update our parameters
    params['subsample'] = subsample
    params['colsample_bytree'] = colsample
    # Run CV
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=NUM_BOOST_ROUNDS,
        seed=42,
        nfold=5,
        metrics={'mae'},
        early_stopping_rounds=10
    )
    # Update best score
    mean_mae = cv_results['test-mae-mean'].min()
    boost_rounds = cv_results['test-mae-mean'].idxmin()
    print("\tMAE {} for {} rounds\n".format(mean_mae, boost_rounds))
    if mean_mae < min_mae:
        min_mae = mean_mae
        best_params = (subsample,colsample)
print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))
params['subsample'] = 1.0
params['colsample_bytree'] = 1.0
min_mae = float("Inf")
best_params = None
for eta in [.3, .2, .1, .05, .01, .005]:
    print("CV with eta={}".format(eta))
    # We update our parameters
    params['eta'] = eta
    # Run and time CV
    cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=NUM_BOOST_ROUNDS,
            seed=42,
            nfold=5,
            metrics=['mae'],
            early_stopping_rounds=10
          )
    # Update best score
    mean_mae = cv_results['test-mae-mean'].min()
    boost_rounds = cv_results['test-mae-mean'].idxmin()
    print("\tMAE {} for {} rounds\n".format(mean_mae, boost_rounds))
    if mean_mae < min_mae:
        min_mae = mean_mae
        best_params = eta
print("Best params: {}, MAE: {}".format(best_params, min_mae))
params['eta'] = 0.05
params
# Put together a final model, using the parameters found in tuning
final_model = xgb.XGBRegressor(n_estimators=268, 
                               learning_rate=0.05, 
                               max_depth = 5, 
                               min_child_weight=1, 
                               subsample=1.0, 
                               colsample_bytree=1.0)

final_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('model', final_model)])

# Preprocessing of validation data, get predictions
final_pipeline.fit(train_X,train_y)
test_data_labels = final_pipeline.predict(test)

# Create predictions to be submitted!
pd.DataFrame({'Id': test.Id, 'SalePrice': test_data_labels}).to_csv('XGB.csv', index =False)  
print("Done :D")