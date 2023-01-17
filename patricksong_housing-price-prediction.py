import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import tensorflow as tf

from collections import OrderedDict
from IPython import display
from scipy.stats import norm
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from tensorflow.python.data import Dataset

housing_dataframe = pd.read_csv("../input/train.csv")
housing_dataframe = housing_dataframe.reindex(np.random.permutation(housing_dataframe.index))
housing_dataframe.isnull().sum().sort_values(ascending=False).head(20)
avg_price = {}
for neighbor in housing_dataframe["Neighborhood"]:
    neighbor_dataframe = housing_dataframe.loc[housing_dataframe["Neighborhood"] == neighbor]
    avg_price[neighbor] = neighbor_dataframe["SalePrice"].mean()
avg_price = OrderedDict(sorted(avg_price.items(), key=lambda k: k[1], reverse=True))
for neighbor, price in avg_price.items():
    print("{0}: {1}".format(neighbor, price))
plt.figure(figsize=(6, 5))
plt.scatter(housing_dataframe["Neighborhood"].map(avg_price), housing_dataframe["SalePrice"], s=10);
for neighbor, price in avg_price.items():
    if price > 300000.0:
        avg_price[neighbor] = 4
    elif price > 180000.0:
        avg_price[neighbor] = 3
    elif price > 120000.0:
        avg_price[neighbor] = 2
    else:
        avg_price[neighbor] = 1
for neighbor, price in avg_price.items():
    print("{0}: {1}".format(neighbor, price))
housing_dataframe = housing_dataframe.replace({"ExterQual" : {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}})
housing_dataframe["TotalArea"] = housing_dataframe["GrLivArea"] + housing_dataframe["TotalBsmtSF"]
housing_dataframe["TotalRooms"] = housing_dataframe["FullBath"] + housing_dataframe["TotRmsAbvGrd"]
housing_dataframe["Location"] = housing_dataframe["Neighborhood"].map(avg_price)
num_col = 13
mul = int(math.ceil(num_col / 10.0))
cols = housing_dataframe.corr().nlargest(num_col, "SalePrice")["SalePrice"].index
corr_mat = np.corrcoef(housing_dataframe[cols].values.T)
plt.figure(figsize=(12 * mul, 7 * mul))
sns.heatmap(corr_mat, annot=True, square=True, fmt=".2f", annot_kws={"size": 12}, yticklabels=cols.values, xticklabels=cols.values);
def scatter_plot(housing_dataframe, cols, filter_zero=False):
    num_col = 3
    num_row = math.ceil(len(cols) / 3.0)
    plt.figure(figsize=(6 * num_col, 5 * num_row))
    for i, col in enumerate(cols, 1):
        ax = plt.subplot(num_row, num_col, i)
        ax.set_title("{0} vs SalePrice".format(col))
        if filter_zero:
            plt.scatter(housing_dataframe[housing_dataframe[col] > 0][col], housing_dataframe[housing_dataframe[col] > 0]["SalePrice"], s=10);
        else:
            plt.scatter(housing_dataframe[col], housing_dataframe["SalePrice"], s=10);

scatter_plot(housing_dataframe, ["GrLivArea", "TotalBsmtSF", "GarageArea", "YearBuilt", "OverallQual", "ExterQual", "TotalRooms", "Location"])
display.display(housing_dataframe.sort_values(by="GrLivArea", ascending=False)[:2])
display.display(housing_dataframe.sort_values(by="TotalBsmtSF", ascending=False)[:1])
housing_dataframe = housing_dataframe.drop(housing_dataframe[housing_dataframe["Id"] == 1299].index)
housing_dataframe = housing_dataframe.drop(housing_dataframe[housing_dataframe["Id"] == 524].index)
def distplot(housing_dataframe, cols, filter_zero=False):
    num_col = 3
    num_row = math.ceil(len(cols) / num_col)
    plt.figure(figsize=(6 * num_col, 5 * num_row))
    for i, col in enumerate(cols, 1):
        plt.subplot(num_row, num_col, i)
        if filter_zero:
            sns.distplot(housing_dataframe[housing_dataframe[col] > 0][col], fit=norm);
        else:
            sns.distplot(housing_dataframe[col], fit=norm);

distplot(housing_dataframe, ["SalePrice", "GrLivArea", "TotalBsmtSF", "TotalArea", "YearBuilt", "GarageArea", "TotalRooms", "Location"])
def bucketized_feature(housing_dataframe, col, num_bin):
    col_bins = housing_dataframe[col].quantile(np.arange(1.0, num_bin) / num_bin)
    bin_col_series = pd.Series(data=np.digitize(housing_dataframe[col], col_bins), index=housing_dataframe.index)
    return bin_col_series
def preprocess_features(housing_dataframe):
    selected_features = housing_dataframe[
        ["GrLivArea",
         "TotalBsmtSF",
         "OverallQual",
         "ExterQual",
         "TotalArea",
         "GarageArea",
         "TotalRooms",
         "YearBuilt"]
    ]
    processed_features = selected_features.copy()
    #processed_features["GrLivArea_sqr"] = np.log(processed_features["GrLivArea"] ** 2)
    processed_features["GrLivArea"] = np.log(processed_features["GrLivArea"])
    processed_features["HasBsmt"] = pd.Series(data=0, index=processed_features.index)
    processed_features.loc[processed_features["TotalBsmtSF"] > 0, "HasBsmt"] = 1
    #processed_features["TotalBsmtSF_exp2"] = np.log(processed_features["TotalBsmtSF"] ** 2 + 1)
    processed_features["TotalBsmtSF"] = np.log(processed_features["TotalBsmtSF"] + 1)
    processed_features["TotalArea_exp2"] = np.log(processed_features["TotalArea"] ** 2 + 1)
    processed_features["TotalArea"] = np.log(processed_features["TotalArea"] + 1)
    processed_features["HasGarageArea"] = pd.Series(data=0, index=processed_features.index)
    processed_features.loc[processed_features["GarageArea"] > 0, "HasGarageArea"] = 1
    processed_features["GarageArea_exp2"] = np.log(processed_features["GarageArea"] ** 2 + 1)
    processed_features["GarageArea"] = np.log(processed_features["GarageArea"] + 1)
    processed_features[["OverallQual", "ExterQual", "TotalRooms", "YearBuilt"]] = processed_features[["OverallQual", "ExterQual", "TotalRooms", "YearBuilt"]].astype(float)
    processed_features["TotalRooms"] = np.log(processed_features["TotalRooms"])
    processed_features["TotalRooms_exp2"] = np.log(processed_features["TotalRooms"] ** 2)
    processed_features["YearBuilt_exp2"] = processed_features["YearBuilt"] ** 2
    #qual_bin_dataframe = pd.get_dummies(pd.Series(data=np.digitize(housing_dataframe["OverallQual"], [2.0, 4.0, 6.0, 8.0]), index=processed_features.index), prefix="OverallQual_Bin")
    #year_bin_dataframe = pd.get_dummies(bucketized_feature(housing_dataframe, "YearBuilt", 5), prefix="BinYearBuilt_Bin")
    #processed_features = pd.concat([processed_features, year_bin_dataframe], axis=1)
    #processed_features = pd.concat([processed_features, pd.get_dummies(housing_dataframe["ExterQual"], prefix_sep=["ExterQual_"])], axis=1)
    processed_features = pd.concat([processed_features, pd.get_dummies(housing_dataframe["Location"], prefix="Location_")], axis=1)
    #for bin_qual_col in qual_bin_dataframe:
    #    for bin_year_col in year_bin_dataframe:
    #        processed_features[bin_qual_col + "_" + bin_year_col] = qual_bin_dataframe[bin_qual_col].multiply(year_bin_dataframe[bin_year_col])
    return processed_features
processed_examples = preprocess_features(housing_dataframe)
distplot(processed_examples, ["GrLivArea", "TotalBsmtSF", "TotalArea", "TotalArea_exp2", "GarageArea", "GarageArea_exp2", "TotalRooms", "TotalRooms_exp2"], filter_zero=True)
def preprocess_targets(housing_dataframe):
    return np.log(housing_dataframe[["SalePrice"]].copy())
processed_targets = preprocess_targets(housing_dataframe)
distplot(processed_targets, ["SalePrice"], filter_zero=True)
scatter_plot(pd.concat([processed_examples, processed_targets], axis=1), ["GrLivArea", "TotalBsmtSF", "TotalRooms", "YearBuilt_exp2"], filter_zero=True)
num_examples = len(processed_targets)
num_training_examples = int(num_examples * 0.7)
num_validation_examples = num_examples - num_training_examples
training_examples = processed_examples.head(num_training_examples).copy()
training_targets = processed_targets.head(num_training_examples).copy()
validation_examples = processed_examples.tail(num_validation_examples).copy()
validation_targets = processed_targets.tail(num_validation_examples).copy()
print("number of training examples: {0}, number of validation examples: {1}".format(len(training_examples), len(validation_examples)))
numeric_features = ["GrLivArea", "TotalBsmtSF", "OverallQual", "ExterQual", "TotalArea", "TotalArea_exp2", "GarageArea", "GarageArea_exp2", "TotalRooms", "TotalRooms_exp2", "YearBuilt", "YearBuilt_exp2"]
std_sc = StandardScaler()
training_examples.loc[:, numeric_features] = std_sc.fit_transform(training_examples.loc[:, numeric_features])
validation_examples.loc[:, numeric_features] = std_sc.transform(validation_examples.loc[:, numeric_features])
print("Training examples summary:")
display.display(training_examples.describe())
print("Validation examples summary:")
display.display(validation_examples.describe())
print("Training targets summary:")
display.display(training_targets.describe())
print("Validation targets summary:")
display.display(validation_targets.describe())
def construct_feature_columns(input_features):
    return set([tf.feature_column.numeric_column(my_feature) for my_feature in input_features])
def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    features = {key:np.array(value) for key,value in dict(features).items()}
    ds = Dataset.from_tensor_slices((features,targets))
    ds = ds.batch(batch_size).repeat(num_epochs)
    if shuffle:
      ds = ds.shuffle(10000)
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels
def train_model(model, steps, 
                batch_size, 
                training_examples, 
                training_targets, 
                validation_examples=None, 
                validation_targets=None):
    calc_validation_rmse = validation_examples is not None and validation_targets is not None
    periods = 10
    steps_per_period = steps / periods
    training_input_fn = lambda: my_input_fn(training_examples, training_targets["SalePrice"], batch_size=batch_size)
    predict_training_input_fn = lambda: my_input_fn(training_examples, training_targets["SalePrice"], num_epochs=1, shuffle=False)
    if calc_validation_rmse:
        predict_validation_input_fn = lambda: my_input_fn(validation_examples, validation_targets["SalePrice"], num_epochs=1, shuffle=False)
    print("Training model...")
    print("RMSE (on training data):")
    training_rmse = []
    if calc_validation_rmse:
        validation_rmse = []
    for period in range (0, periods):
        model.train(input_fn=training_input_fn, steps=steps_per_period)
        
        training_predictions = model.predict(input_fn=predict_training_input_fn)
        training_predictions = np.array([item['predictions'][0] for item in training_predictions])
        if calc_validation_rmse:
            validation_predictions = model.predict(input_fn=predict_validation_input_fn)
            validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])
        
        training_root_mean_squared_error = math.sqrt(metrics.mean_squared_error(training_predictions, training_targets))
        if calc_validation_rmse:
            validation_root_mean_squared_error = math.sqrt(metrics.mean_squared_error(validation_predictions, validation_targets))
            print("  period %02d : training_error: %0.2f, validation_error: %0.2f" % (period, training_root_mean_squared_error, validation_root_mean_squared_error))
        else:
            print("  period %02d : train_err: %0.2f" % (period, training_root_mean_squared_error))
        
        training_rmse.append(training_root_mean_squared_error)
        if calc_validation_rmse:
            validation_rmse.append(validation_root_mean_squared_error)
    
    print("Model training finished.")
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.ylabel("RMSE")
    plt.xlabel("Periods")
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.plot(training_rmse, label="training")
    if calc_validation_rmse:
        plt.plot(validation_rmse, label="validation")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    training_final_predictions = model.predict(input_fn=predict_training_input_fn)
    training_final_predictions = np.array([item['predictions'][0] for item in training_final_predictions])
    plt.scatter(training_final_predictions, training_targets["SalePrice"], c="blue", label="Training data")
    if calc_validation_rmse:
        validation_final_predictions = model.predict(input_fn=predict_validation_input_fn)
        validation_final_predictions = np.array([item['predictions'][0] for item in validation_final_predictions])
        plt.scatter(validation_final_predictions, validation_targets["SalePrice"], c="lightgreen", label="Validation data")
    plt.ylabel("Real Values")
    plt.xlabel("Predicted Values")
    plt.legend(loc="upper left")
    plt.plot([10.5, 13.5], [10.5, 13.5], c="red")
    
    return model
def train_linear_regression_model(learning_rate, 
                                  steps, 
                                  batch_size, 
                                  training_examples, 
                                  training_targets, 
                                  l2_regularization_strength=0.0,
                                  validation_examples=None, 
                                  validation_targets=None):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    #optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate)
    optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)
    linear_regressor = tf.estimator.LinearRegressor(feature_columns=construct_feature_columns(training_examples), optimizer=optimizer)
    return train_model(linear_regressor, steps, batch_size, training_examples, training_targets, validation_examples, validation_targets)
linaer_regressor = train_linear_regression_model(
    learning_rate=0.01,
    steps=6000,
    batch_size=20,
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets
)
def train_nn_regression_model(learning_rate, 
                              steps, 
                              batch_size, 
                              hidden_units, 
                              training_examples, 
                              training_targets, 
                              l2_regularization_strength=0.0, 
                              validation_examples=None, 
                              validation_targets=None):
    #optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate, l2_regularization_strength=l2_regularization_strength)
    optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate, l2_regularization_strength=l2_regularization_strength)
    optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)
    dnn_regressor = tf.estimator.DNNRegressor(feature_columns=construct_feature_columns(training_examples), 
                                              hidden_units=hidden_units, 
                                              optimizer=optimizer)
    return train_model(dnn_regressor, 
                       steps, batch_size, 
                       training_examples, 
                       training_targets, 
                       validation_examples, 
                       validation_targets)
num_col = len(training_examples.columns)
print("{0} columns".format(num_col))
dnn_regressor = train_nn_regression_model(
    learning_rate=0.3,
    steps=30000,
    batch_size=20,
    hidden_units=[36, 18, 9],
    training_examples=training_examples,
    training_targets=training_targets,
    l2_regularization_strength=0.1,
    validation_examples=validation_examples,
    validation_targets=validation_targets
)
def train_random_forest_model(training_examples, training_targets, validation_examples, validation_targets):
    forest_regressor = RandomForestRegressor()
    forest_regressor.fit(training_examples, np.ravel(training_targets))
    training_predictions = forest_regressor.predict(training_examples)
    validation_predictions = forest_regressor.predict(validation_examples)
    training_root_mean_squared_error = math.sqrt(metrics.mean_squared_error(training_predictions, training_targets))
    validation_root_mean_squared_error = math.sqrt(metrics.mean_squared_error(validation_predictions, validation_targets))
    print("training set RMSE: %0.2f" % (training_root_mean_squared_error))
    print("validation set RMSE: %0.2f" % (validation_root_mean_squared_error))
    return forest_regressor
forest_regressor = train_random_forest_model(training_examples, training_targets, validation_examples, validation_targets)
test_housing_dataframe = pd.read_csv("../input/test.csv")
test_housing_dataframe[["GrLivArea", "TotalBsmtSF", "OverallQual", "ExterQual", "GarageArea", "YearBuilt", "FullBath", "Neighborhood"]].isnull().sum().sort_values(ascending=False)
display.display(test_housing_dataframe.loc[test_housing_dataframe["TotalBsmtSF"].isnull()])
display.display(test_housing_dataframe.loc[test_housing_dataframe["GarageArea"].isnull()])
test_housing_dataframe.loc[test_housing_dataframe["TotalBsmtSF"].isnull(), "TotalBsmtSF"] = 0.0
test_housing_dataframe.loc[test_housing_dataframe["GarageArea"].isnull(), "GarageArea"] = 0.0
for neighbor in test_housing_dataframe["Neighborhood"].unique():
    assert neighbor in avg_price
test_housing_dataframe = test_housing_dataframe.replace({"ExterQual" : {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}})
test_housing_dataframe["TotalArea"] = test_housing_dataframe["GrLivArea"] + test_housing_dataframe["TotalBsmtSF"]
test_housing_dataframe["TotalRooms"] = test_housing_dataframe["FullBath"] + test_housing_dataframe["TotRmsAbvGrd"]
test_housing_dataframe["Location"] = test_housing_dataframe["Neighborhood"].map(avg_price)
test_examples = preprocess_features(test_housing_dataframe)
test_examples.loc[:, numeric_features] = std_sc.transform(test_examples.loc[:, numeric_features])

predict_test_input_fn = lambda: my_input_fn(test_examples, test_housing_dataframe["Id"], num_epochs=1, shuffle=False)

linear_test_predictions = linaer_regressor.predict(input_fn=predict_test_input_fn)
linear_test_predictions = np.array([item['predictions'][0] for item in linear_test_predictions])
linear_predict_dataframe = pd.DataFrame()
linear_predict_dataframe["Id"] = test_housing_dataframe["Id"].copy()
linear_predict_dataframe["SalePrice"] = np.exp(linear_test_predictions)
linear_predict_dataframe.to_csv("linear_submission.csv", index=False, header=["Id", "SalePrice"])

dnn_test_predictions = dnn_regressor.predict(input_fn=predict_test_input_fn)
dnn_test_predictions = np.array([item['predictions'][0] for item in dnn_test_predictions])
dnn_predict_dataframe = pd.DataFrame()
dnn_predict_dataframe["Id"] = test_housing_dataframe["Id"].copy()
dnn_predict_dataframe["SalePrice"] = np.exp(dnn_test_predictions)
dnn_predict_dataframe.to_csv("dnn_submission.csv", index=False, header=["Id", "SalePrice"])

forest_test_predictions = forest_regressor.predict(test_examples)
forest_predict_dataframe = pd.DataFrame()
forest_predict_dataframe["Id"] = test_housing_dataframe["Id"].copy()
forest_predict_dataframe["SalePrice"] = np.exp(forest_test_predictions)
forest_predict_dataframe.to_csv("forest_submission.csv", index=False, header=["Id", "SalePrice"])