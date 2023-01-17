from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler

import numpy as np

import pandas as pd

import tensorflow as tf
responses_csv_filepath = "../input/kaggle-survey-2017/multipleChoiceResponses.csv"

all_responses = pd.read_csv(responses_csv_filepath, dtype=object, encoding="iso-8859-1")
all_responses.info()
column_names = ["CompensationAmount", "CompensationCurrency", "GenderSelect", "Country", 

                "Age", "EmploymentStatus","CodeWriter", "CurrentJobTitleSelect", 

                "FormalEducation", "MajorSelect", "Tenure", "ParentsEducation", 

                "EmployerIndustry", "EmployerSize"]



responses = all_responses[column_names]
print(responses.info())
responses = responses.dropna(axis=0, how='any')
print(responses.info())
country_vc = responses["Country"].value_counts()



min_samples_threshold = 100

country_names = country_vc[country_vc >= min_samples_threshold].index

print("Countries with at least 100 samples: " + str(country_names.tolist()))
responses = responses[responses["Country"].isin(country_names)]



print("There are now " + str(responses.shape[0]) + " rows left.")
responses = responses[responses["EmploymentStatus"] == "Employed full-time"]

responses = responses.drop("EmploymentStatus", axis=1)



print("There are now " + str(responses.shape[0]) + " rows left.")
age_outliers = ["1", "100"]

responses = responses[responses["Age"].isin(age_outliers) == False]



print("There are now " + str(responses.shape[0]) + " rows left.")
print(responses["CodeWriter"].value_counts())



responses = responses.drop("CodeWriter", axis=1)
invalid_comp_amt = ["-99", "-1", "0", "-"]

responses = responses[responses["CompensationAmount"].isin(invalid_comp_amt) == False]



responses["CompensationAmount"] = responses["CompensationAmount"].replace("140000,00", "140000.00")



print("There are now " + str(responses.shape[0]) + " rows left.")
responses["CompensationAmount"] = responses["CompensationAmount"].str.replace(",", "")

responses["CompensationAmount"] = responses["CompensationAmount"].astype(np.float64)
responses["Age"] = responses["Age"].astype(np.int8)
conversionRates_csv_filepath = "../input/kaggle-survey-2017/conversionRates.csv"

column_names = ["originCountry", "exchangeRate"]

dtype = {"originCountry": object, "exchangeRate": np.float16}



conversionRates = pd.read_csv(conversionRates_csv_filepath, usecols=column_names, dtype=dtype)
responses = responses.merge(conversionRates, left_on="CompensationCurrency", 

                                  right_on="originCountry", how="inner")



print("There are now " + str(responses.shape[0]) + " rows left.")
responses["CompensationAmountUSD"] = responses["CompensationAmount"] * responses["exchangeRate"]
column_names_to_drop = ["CompensationAmount", "CompensationCurrency", "originCountry", "exchangeRate"]

responses = responses.drop(column_names_to_drop, axis=1)
responses.info()
# filename = "multipleChoiceResponsesCleaned.csv"

# responses.to_csv(filename, index=False, encoding='utf-8')
united_states = responses[responses["Country"] == "United States"]

united_states["Tenure"].value_counts(normalize=True)
india = responses[responses["Country"] == "India"]

india["Tenure"].value_counts(normalize=True)
print("Before removing outliers: mean=" + str(np.mean(responses["CompensationAmountUSD"])))



comp_amnt = responses["CompensationAmountUSD"]

responses = responses[(comp_amnt >= 30000) & (comp_amnt <= 300000)]



print("After removing outliers: mean=" + str(np.mean(responses["CompensationAmountUSD"])))



print("There are now " + str(responses.shape[0]) + " rows left.\n")



print(responses["Country"].value_counts())
gender_select_one_hot = pd.get_dummies(responses["GenderSelect"])



print(gender_select_one_hot.info())

print("First row values: " + str(gender_select_one_hot.values[0]))
column_names_to_encode = ["GenderSelect", "Country", "CurrentJobTitleSelect", "FormalEducation", 

                          "MajorSelect", "Tenure", "ParentsEducation", "EmployerIndustry", "EmployerSize"]



columns_encoded = pd.get_dummies(responses[column_names_to_encode])

responses = responses.join(columns_encoded)



responses = responses.drop(column_names_to_encode, axis=1)
responses.columns = responses.columns.str.replace(" ", "_")

responses.columns = responses.columns.str.replace(",", "_")

responses.columns = responses.columns.str.replace("'", "_")

responses.columns = responses.columns.str.replace("(", "_")

responses.columns = responses.columns.str.replace(")", "_")
feature_names = responses.columns[responses.columns != "CompensationAmountUSD"]



scaler = StandardScaler()

responses[feature_names] = scaler.fit_transform(responses[feature_names])
np.random.seed(0) # Seed is hard coded to 0 so that the results are reproducible



total_rows = responses.shape[0]

shuffled_indices = np.random.permutation(total_rows)

test_set_size = int(total_rows * 0.2)

test_indices = shuffled_indices[:test_set_size]

train_indices = shuffled_indices[test_set_size:]



train_set = responses.iloc[train_indices]

test_set = responses.iloc[test_indices]
X_train_set = train_set.drop("CompensationAmountUSD", axis=1)

y_train_set = train_set["CompensationAmountUSD"]



X_test_set = test_set.drop("CompensationAmountUSD", axis=1)

y_test_set = test_set["CompensationAmountUSD"]
feature_names = X_train_set.columns

feature_columns = [tf.feature_column.numeric_column(feature_name) for feature_name in feature_names]



model_dir = "tmp"

regressor = tf.estimator.DNNRegressor(feature_columns=feature_columns,

                                      hidden_units=[1000, 1000, 1000],

                                      model_dir=model_dir)
input_fn_train = tf.estimator.inputs.pandas_input_fn(x=X_train_set, y=y_train_set, shuffle=True)
regressor.train(input_fn=input_fn_train, steps=1000)
input_fn_test = tf.estimator.inputs.pandas_input_fn(x=X_test_set, y=y_test_set, shuffle=False, num_epochs=1)



prediction_generators = regressor.predict(input_fn=input_fn_test)

y_predictions = [pg["predictions"] for pg in prediction_generators]



errors = []

for i in range(len(y_predictions)):

    y_prediction = y_predictions[i]

    y_actual = y_test_set.values[i]

    

    error = abs(y_prediction - y_actual) / y_prediction

    errors.append(error)

    

print("test set size: " + str(len(y_predictions)))

print("mean:" + str(np.mean(errors)))

print("median:" + str(np.median(errors)))
for i in range(len(y_predictions)):

    y_prediction = y_predictions[i]

    y_actual = y_test_set.values[i]

    

    error = abs(y_prediction - y_actual) / y_prediction

    print("Predicted: " + str(int(y_prediction)) 

          + ", Actual: " + str(int(y_actual)) 

          + ", error: " + str(round(float(error), 2)))
from sklearn.linear_model import LinearRegression



lr = LinearRegression()

lr.fit(X_train_set, y_train_set)

y_predictions = lr.predict(X_test_set)



errors = []

for i in range(len(y_predictions)):

    y_prediction = y_predictions[i]

    y_actual = y_test_set.values[i]

    

    error = abs(y_prediction - y_actual) / y_prediction

    errors.append(error)

    

print("test set size: " + str(len(y_predictions)))

print("mean:" + str(np.mean(errors)))

print("median:" + str(np.median(errors)))
for i in range(len(y_predictions)):

    y_prediction = y_predictions[i]

    y_actual = y_test_set.values[i]

    

    error = abs(y_prediction - y_actual) / y_prediction

    print("Predicted: " + str(int(y_prediction)) 

          + ", Actual: " + str(int(y_actual)) 

          + ", error: " + str(round(float(error), 2)))