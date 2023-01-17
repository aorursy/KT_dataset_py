import os

import numpy as np
import pandas as pd
import tensorflow as tf

from scipy.special import boxcox1p
from scipy.stats import skew

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Let's load the house prices dataset
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

print('Train data shape: {}'.format(train.shape))
print('Test data shape: {}'.format(test.shape))
train = train.drop(
    train[
        (train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)
    ].index
)

print('New Train data shape: {}'.format(train.shape))
# Let's save some variables that we will use in the future

test_id = test['Id']
ntrain = train.shape[0]
y_train = train.SalePrice.values
dataset = pd.concat((train, test)).reset_index(drop=True)

fill_zero = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
             'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea',
             'GarageYrBlt', 'GarageArea', 'GarageCars']

for column in fill_zero:
    dataset[column] = dataset[column].fillna(0)

fill_none = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
             'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
             'MasVnrType', 'MSSubClass', 'BsmtQual', 'BsmtCond',
             'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']

for column in fill_none:
    dataset[column] = dataset[column].fillna('None')

fill_mode = ['Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd',
             'SaleType', 'MSZoning']

for column in fill_mode:
    dataset[column] = dataset[column].fillna(
        dataset[column].mode()[0])

dataset["Functional"] = dataset["Functional"].fillna("Typ")

dataset['LotFrontage'] = dataset.groupby(
    'Neighborhood')['LotFrontage'].transform(
        lambda x: x.fillna(x.median()))
to_str_columns = ['MSSubClass', 'OverallQual', 'OverallCond', 'YrSold', 'MoSold']

for column in to_str_columns:
    dataset[column] = dataset[column].apply(str)
dataset = dataset.drop(['Utilities'], axis=1)
dataset = dataset.drop(['SalePrice'], axis=1)
dataset = dataset.drop(['Id'], axis=1)
numeric_features = dataset.dtypes[dataset.dtypes != "object"].index
skewed_features = dataset[numeric_features].apply(
    lambda x: skew(x.dropna())).sort_values(ascending=False)

skewness = pd.DataFrame({'Skew': skewed_features})
skewness = skewness[abs(skewness) > 0.5]
skewed_features = skewness.index
lam = 0.15
        
for feature in skewed_features:
    dataset[feature] = boxcox1p(dataset[feature], lam)

train = dataset[:ntrain]
test = dataset[ntrain:]
targets = pd.DataFrame(np.log(y_train))

print('Train data shape after pre-processing: {}'.format(train.shape))
print('Test data shape after pre-processing: {}'.format(test.shape))
numerical_features = train.select_dtypes(exclude=["object"]).columns

normalizer = StandardScaler()

train.loc[:, numerical_features] = normalizer.fit_transform(
    train.loc[:, numerical_features])
test.loc[:, numerical_features] = normalizer.transform(
    test.loc[:, numerical_features])
numeric_columns = []

numeric_features = train.select_dtypes(exclude=[np.object])

for feature in numeric_features:
    numeric_columns.append(
        tf.feature_column.numeric_column(
            key=feature
        )
    )

print('Number of numeric features: {}'.format(len(numeric_columns)))
categorical_columns = []
categorical_dict = {}

categorical_features = train.select_dtypes(exclude=[np.number])

for feature in categorical_features:
    categorical_dict[feature] = dataset[feature].unique()

for key, unique_values in categorical_dict.items():
    categorical_columns.append(
        tf.feature_column.categorical_column_with_vocabulary_list(
            key=key,
            vocabulary_list=unique_values
        )
    )

print('Number of categorical features: {}'.format(len(categorical_columns)))

# Verify if we have used all the available features
assert len(categorical_columns) + len(numeric_columns) == train.shape[1]
def create_model():
    linear_estimator = tf.estimator.LinearRegressor(
        feature_columns=numeric_columns + categorical_columns,
        optimizer=tf.train.FtrlOptimizer(
                        learning_rate=0.1,
                        l1_regularization_strength=0.0,
                        l2_regularization_strength=3.0)
    )
    
    return linear_estimator

linear_estimator = create_model()
batch_size = 64

def train_model(linear_estimator, x, y, should_shuffle):
    num_epochs = 25
    
    linear_estimator.train(
        input_fn=tf.estimator.inputs.pandas_input_fn(
            x=x,
            y=y,
            batch_size=batch_size,
            num_epochs=num_epochs,
            shuffle=should_shuffle
        )
    )

train_model(linear_estimator, train, targets, should_shuffle=True)
def rmse(labels, predictions):
    # Casting is used to guarantee that both labels and predictions have the same types.
    return {'rmse': tf.metrics.root_mean_squared_error(
        tf.cast(labels, tf.float32), tf.cast(predictions['predictions'], tf.float32))}
linear_estimator = tf.contrib.estimator.add_metrics(linear_estimator, rmse)
def evaluate_model(linear_estimator, validation_data, validation_targets, should_shuffle):
    evaluate_dict = linear_estimator.evaluate(
        input_fn=tf.estimator.inputs.pandas_input_fn(
            x=validation_data,
            y=validation_targets,
            batch_size=batch_size,
            shuffle=should_shuffle
        )
    )
    
    """
    The dict returns values such as the average loss as well, which you can use as well.
    However, for our example, we only need the value of the rmse metric
    """
    return evaluate_dict['rmse']

# We are using the train dataset to validate our model here just for an example
rmse_value = evaluate_model(linear_estimator, train[:100], targets[:100], should_shuffle=False)
print('Rmse metric: {}'.format(rmse_value))
#Disable TensorFlow logs for running k-fold
tf.logging.set_verbosity(tf.logging.ERROR)

k_fold = KFold(n_splits=5)
all_rmse = []

for index, (train_index, validation_index) in enumerate(k_fold.split(train)):
    print('Running fold {}'.format(index + 1))

    train_data = train.loc[train_index, :]
    train_targets = targets.loc[train_index, :]

    validation_data = train.loc[validation_index, :]
    validation_targets = targets.loc[validation_index, :]

    linear_estimator = create_model()
    linear_estimator = tf.contrib.estimator.add_metrics(linear_estimator, rmse)
    train_model(linear_estimator, train_data, train_targets, should_shuffle=True)
    rmse_value = evaluate_model(linear_estimator, validation_data, validation_targets, should_shuffle=False)
    all_rmse.append(rmse_value)

final_rmse= sum(all_rmse) / len(all_rmse)        
print('K-fold rmse: {}'.format(final_rmse))
#Enable TensorFlow logging
tf.logging.set_verbosity(tf.logging.INFO)

linear_estimator = create_model()
train_model(linear_estimator, train, targets, should_shuffle=True)
def model_predict(linear_estimator, data):
    predictions = linear_estimator.predict(
        input_fn=tf.estimator.inputs.pandas_input_fn(
            x=data,
            batch_size=batch_size,
            shuffle=False
        )
    )
    
    # all of the predictions are returned as a numpy array. We simple transform this into a list
    pred = [prediction['predictions'].item(0) for prediction in predictions]
    return pred

estimator_predictions = model_predict(linear_estimator, test)
print('Number of predictions: {}'.format(len(estimator_predictions)))
final_predictions = np.exp(estimator_predictions)

submission = pd.DataFrame()
submission['Id'] = test_id
submission['SalePrice'] = final_predictions

print('Submission shape: {}'.format(submission.shape))