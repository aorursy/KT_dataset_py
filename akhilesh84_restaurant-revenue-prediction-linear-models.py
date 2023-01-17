import numpy as np # linear algebra
import datetime
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# import libraries for visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Library for statistical operations
import scipy
import sklearn as sk
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import cross_val_predict, cross_val_score, KFold
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error, mean_absolute_error

import os
# os.listdir("../input/")
# Any results you write to the current directory are saved as output.
# Load data from spreadsheet to pandas dataframes
train_data = pd.read_csv("../input/train.csv", parse_dates = [1])
test_data = pd.read_csv("../input/test.csv", parse_dates = [1])

# Note that we are intentionally making a copy of the raw dataset as we'll be manipulating it
# to make it appropriate for our ML algo code
X_train = train_data.iloc[:, np.arange(dtype=int, start = 1, stop = 42, step = 1)].copy()
y_train = train_data.iloc[:, [42]].copy()
X_test = test_data.iloc[:, np.arange(dtype=int, start = 1, stop = 42, step = 1)].copy()

# From the above view, we can see that "Open Date" is a date filed. We definitely cannot ue this field as an
# index as this is not a time series data that we are operating on. At the same time we cannot even ignore
# this field. One simple logic might say that how old a restaurant is might also be a factor influencing its
# sustainability and hence revenue. But this is a mere guess.
# In order to convert this field to a continuous field, we can take the difference of the date in dataset
# with that of some seed date.

seed = datetime.datetime(1990, 1, 1)
X_train["Open Date"] = X_train["Open Date"].map(lambda d: pd.Timedelta(d - seed).days)
X_test["Open Date"] = X_test["Open Date"].map(lambda d: pd.Timedelta(d - seed).days)
models = []
models.append(("OLS", linear_model.LinearRegression()))
# models.append(("SGD", linear_model.SGDRegressor(loss='squared_loss', max_iter=1000)))
le = LabelEncoder()
columns_to_encode = ['City', 'City Group', 'Type']

# Iterating over all the common columns in train and test
for col in X_test.columns.values:
    # Encoding only categorical variables
    if X_test[col].dtypes=='object':
        # Using whole data to form an exhaustive list of levels
        data=X_train[col].append(X_test[col])
        le.fit(data.values)
        X_train[col]=le.transform(X_train[col])
        X_test[col]=le.transform(X_test[col])
X_train.describe()
# We'll create X_train_1, y_train_1 and X_test_1 to be used for this model which uses plain label encoding.

X_train_1 = X_train.iloc[:].copy()
y_train_1 = y_train.iloc[:].copy()
X_test_1 = X_test.iloc[:].copy()

X_scaler = StandardScaler()
y_scaler = StandardScaler()

design_matrix_columns = X_train.columns
response_vector_columns = y_train.columns

X_train_1 = pd.DataFrame(X_scaler.fit_transform(X_train_1), columns=design_matrix_columns)
y_train_1 = pd.DataFrame(y_scaler.fit_transform(y_train_1), columns=response_vector_columns)
X_test_1 = pd.DataFrame(X_scaler.fit_transform(X_test_1), columns=design_matrix_columns)

# Now that we have converted all the columns to numerical fields, we can try and fit a linear model.

for model_name, model in models:
    model.fit(X_train_1, y_train_1)
    print("Model Name: {0}\nModel Score: {1:.2f}%".format(model_name, model.score(X_train_1, y_train_1) * 100))

# Generate the submission file.
#selected_model = models[0][1]
#predicted_df = pd.DataFrame(
#    y_scaler.inverse_transform(np.round(selected_model.predict(X_test))),
#    columns=['Prediction']
#)
#predicted_df["Id"] = predicted_df.index
#predicted_df = predicted_df[['Id', 'Prediction']]
#predicted_df
#predicted_df.to_csv("../input/linear_regression_model1_submission.csv", index=False)
X_train_2 = X_train.iloc[:].copy()
X_test_2 = X_test.iloc[:].copy()

# Perform one hot encoding on training and test data
one_hot_encoder = OneHotEncoder(dtype=np.int8, sparse=False)
columns_to_encode = ['City', 'City Group', 'Type']

for col in columns_to_encode:
    data = X_train_2[[col]].append(X_test_2[[col]])
    one_hot_encoder.fit(data)
    
    temp = one_hot_encoder.transform(X_train_2[[col]])
    temp = pd.DataFrame(temp, columns=[(col + "_" + str(i)) for i in data[col].value_counts().index.values])
    temp = temp.set_index(X_train_2.index.values)
    X_train_2 = pd.concat([X_train_2, temp], axis = 1)
    
    temp = one_hot_encoder.transform(X_test_2[[col]])
    temp = pd.DataFrame(temp, columns=[(col + "_" + str(i)) for i in data[col].value_counts().index.values])
    temp = temp.set_index(X_test_2.index.values)
    X_test_2=pd.concat([X_test_2,temp],axis=1)

# Evaluate model performance
for model_name, model in models:
    model.fit(X_train_2, y_train)
    print("Model Name: {0}\nModel Score: {1:.2f}%".format(model_name, model.score(X_train_2, y_train) * 100))

#Generate the submission file.
# selected_model = models[0][1]
# predicted_df = pd.DataFrame(
#    np.round(selected_model.predict(X_test_2)),
#    columns=['Prediction']
# )
# predicted_df["Id"] = predicted_df.index
# predicted_df = predicted_df[['Id', 'Prediction']]
# predicted_df.to_csv("../input/linear_regression_model2_submission.csv", index=False)
X_train_3 = X_train.iloc[:].copy()
X_test_3 = X_test.iloc[:].copy()
y_train_3 = y_train.iloc[:].copy()

# Perform one hot encoding on training and test data
one_hot_encoder = OneHotEncoder(dtype=np.int8, sparse=False)
columns_to_encode = ['City', 'City Group', 'Type']

for col in columns_to_encode:
    data = X_train_3[[col]].append(X_test_3[[col]])
    one_hot_encoder.fit(data)
    temp = one_hot_encoder.transform(X_train_3[[col]])
    temp = pd.DataFrame(temp, columns=[(col + "_" + str(i)) for i in data[col].value_counts().index.values])
    temp = temp.set_index(X_train_3.index.values)
    X_train_3 = pd.concat([X_train_3, temp], axis = 1)
    X_train_3.drop([col], axis = 1, inplace = True)
    
    temp = one_hot_encoder.transform(X_test_3[[col]])
    temp = pd.DataFrame(temp, columns=[(col + "_" + str(i)) for i in data[col].value_counts().index.values])
    temp = temp.set_index(X_test_3.index.values)
    X_test_3 = pd.concat([X_test_3,temp],axis=1)
    X_test_3.drop([col], axis = 1, inplace = True)

# Perform MinMax scaling to scale each feature to a range between 0 and 1
X_scaler = MinMaxScaler(feature_range=(0,1))
y_scaler = MinMaxScaler(feature_range=(0,1))

design_matrix_columns = X_train_3.columns
response_vector_columns = y_train_3.columns

X_train_3 = pd.DataFrame(X_scaler.fit_transform(X_train_3), columns=design_matrix_columns)
X_test_3 = pd.DataFrame(X_scaler.fit_transform(X_test_3), columns=design_matrix_columns)
y_train_3 = pd.DataFrame(y_scaler.fit_transform(y_train_3), columns=response_vector_columns)

# Evaluate model performance
for model_name, model in models:
    model.fit(X_train_3, y_train_3)
    print("Model Name: {0}\nModel Score: {1:.2f}%".format(model_name, model.score(X_train_3, y_train_3) * 100))

#Generate the submission file.
# selected_model = models[0][1]
# predicted_df = pd.DataFrame(
#    y_scaler.inverse_transform(np.round(selected_model.predict(X_test_3))),
#    columns=['Prediction']
# )
# predicted_df["Id"] = predicted_df.index
# predicted_df = predicted_df[['Id', 'Prediction']]
# predicted_df.to_csv("../input/linear_regression_model3_submission.csv", index=False)
# Implementing feature selection
X_train_4 = X_train.iloc[:].copy()
X_test_4 = X_test.iloc[:].copy()
y_train_4 = y_train.iloc[:].copy()

# Perform one hot encoding on training and test data
one_hot_encoder = OneHotEncoder(dtype=np.int8, sparse=False)
columns_to_encode = ['City', 'City Group', 'Type']

for col in columns_to_encode:
    data = X_train_4[[col]].append(X_test_4[[col]])
    one_hot_encoder.fit(data)
    temp = one_hot_encoder.transform(X_train_4[[col]])
    temp = pd.DataFrame(temp, columns=[(col + "_" + str(i)) for i in data[col].value_counts().index.values])
    temp = temp.set_index(X_train_4.index.values)
    X_train_4 = pd.concat([X_train_4, temp], axis = 1)
    X_train_4.drop([col], axis = 1, inplace = True)
    
    temp = one_hot_encoder.transform(X_test_4[[col]])
    temp = pd.DataFrame(temp, columns=[(col + "_" + str(i)) for i in data[col].value_counts().index.values])
    temp = temp.set_index(X_test_4.index.values)
    X_test_4 = pd.concat([X_test_4,temp],axis=1)
    X_test_4.drop([col], axis = 1, inplace = True)

# Perform MinMax scaling to scale each feature to a range between 0 and 1
X_scaler = MinMaxScaler(feature_range=(0,1))
y_scaler = MinMaxScaler(feature_range=(0,1))

design_matrix_columns = X_train_4.columns
response_vector_columns = y_train_4.columns

X_train_4 = pd.DataFrame(X_scaler.fit_transform(X_train_4), columns=design_matrix_columns)
X_test_4 = pd.DataFrame(X_scaler.fit_transform(X_test_4), columns=design_matrix_columns)
y_train_4 = pd.DataFrame(y_scaler.fit_transform(y_train_4), columns=response_vector_columns)

# Evaluate model performance
for model_name, model in models:
    rfe = RFE(estimator=model, n_features_to_select=95, step=1)
    rfe.fit(X_train_4, y_train_4)
    selected_columns = [i for i,j in zip(X_train_4.columns.values, rfe.support_) if j == True]
    X_train_4 = X_train_4.loc[:, selected_columns]
    model.fit(X_train_4, y_train_4)
    print("Model Name: {0}\nModel Score: {1:.2f}%".format(model_name, model.score(X_train_4, y_train_4) * 100))
#     print("RMSE: ", np.mean(cross_val_score(model, X=X_train_4, y=y_train, cv = 3)))

#Generate the submission file.
# selected_model = models[0][1]
# predicted_df = pd.DataFrame(
#    y_scaler.inverse_transform(np.round(selected_model.predict(X_test_4))),
#    columns=['Prediction']
# )
# predicted_df["Id"] = predicted_df.index
# predicted_df = predicted_df[['Id', 'Prediction']]
# predicted_df.to_csv("../input/linear_regression_model3_submission.csv", index=False)
X_train_5 = X_train.iloc[:].copy()
X_test_5 = X_test.iloc[:].copy()
y_train_5 = y_train.iloc[:].copy()

total_data = pd.concat([X_train_5, X_test_5])

total_records = len(total_data.index)

# Geting of missing values. We will make a simple assumption here. Any feature for which the overall percentage of 0's is more than 20% will be assumed to have zero impact on response variable.
# So we will simple drop those columns from te training set. In order to do so, we'll consider the test set as well.
def null_percentage(column):return(np.round((len(total_data[column][total_data[column] == 0])/total_records) * 100, 2))

for column in total_data.columns.values:
    threshold = null_percentage(column)
    if threshold >= 20:
        print(" Dropping column {0} as percentage of values that are zero: {1}. Much higher that threshold of 20%".format(column, threshold))
        X_train_5.drop(columns=[column], inplace=True)
        X_test_5.drop(columns=[column], inplace=True)

from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor()
model.fit(X_train_5, y_train_5)

print("Score on training set: {0:.2f}".format(model.score(X_train_5, y_train_5) * 100))

#Generate the submission file.
predicted_df = pd.DataFrame(
   np.round(model.predict(X_test_5)),
   columns=['Prediction']
)
predicted_df["Id"] = predicted_df.index
predicted_df = predicted_df[['Id', 'Prediction']]
predicted_df.to_csv("../input/linear_regression_model5_submission.csv", index=False)
# train_data.sort_index(by = ["revenue"])


