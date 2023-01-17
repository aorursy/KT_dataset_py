import pandas as pd



all_data_4 = pd.read_csv("../input/covid-19-race-gender-poverty-risk-us-county/covid_data_log_200922.csv")
y = all_data_4["Cases"]



X = all_data_4.drop(["Deaths", "Cases", "FIPS", "stateFIPS"

                     , "countyFIPS_2d", "County", "State", "Risk_Cat"],  axis=1)



X
# train-test split

from sklearn.model_selection import train_test_split



# allocate 70% at random to training data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
from sklearn.ensemble import RandomForestRegressor



reg = RandomForestRegressor(max_depth=2, random_state=10, oob_score=True, bootstrap=True)
reg.fit(X_train, y_train)
# Get numerical feature importances

importances = list(reg.feature_importances_)



# List of tuples with variable and importance

feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(X_train, importances)]



# Sort the feature importances by most important first

feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)



# Print out the feature and importances 

[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
preds = reg.predict(X_test)
evaluate = pd.DataFrame({

    "actual" : y_test

    , "predicted" : preds

})



evaluate["error"] = evaluate["actual"] - evaluate["predicted"]



evaluate.head()
import numpy as np



# Calculate the absolute errors

errors = abs(preds - y_test)



# Print out the mean absolute error (mae)

print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
all_data_5 = all_data_4.copy()



y = all_data_5["Cases"]



X = all_data_5.drop(["Deaths", "Cases", "FIPS", "stateFIPS", "countyFIPS_2d", "County"

                     , "State", "Risk_Cat", "Risk_Index", "H_Male", "H_Female", "I_Male", "I_Female"

                    , "A_Male", "A_Female", "NH_Male", "NH_Female"],  axis=1)



X
# train-test split

from sklearn.model_selection import train_test_split



# allocate 70% at random to training data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
from sklearn.ensemble import RandomForestRegressor



reg = RandomForestRegressor(max_depth=2, random_state=10, oob_score=True, bootstrap=True)
reg.fit(X_train, y_train)
# Get numerical feature importances

importances = list(reg.feature_importances_)



# List of tuples with variable and importance

feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(X_train, importances)]



# Sort the feature importances by most important first

feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)



# Print out the feature and importances 

[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
preds = reg.predict(X_test)
evaluate = pd.DataFrame({

    "actual" : y_test

    , "predicted" : preds

})



evaluate["error"] = evaluate["actual"] - evaluate["predicted"]



evaluate.head()
# Calculate the absolute errors

errors = abs(preds - y_test)



# Print out the mean absolute error (mae)

print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
all_data_6 = all_data_4.copy()
y = all_data_6["Deaths"]



X = all_data_6.drop(["Deaths", "FIPS", "stateFIPS"

                     , "countyFIPS_2d", "County", "State", "Risk_Cat"],  axis=1)



X
# train-test split

from sklearn.model_selection import train_test_split



# allocate 70% at random to training data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
from sklearn.ensemble import RandomForestRegressor



reg = RandomForestRegressor(max_depth=2, random_state=10, oob_score=True, bootstrap=True)
reg.fit(X_train, y_train)
# Get numerical feature importances

importances = list(reg.feature_importances_)



# List of tuples with variable and importance

feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(X_train, importances)]



# Sort the feature importances by most important first

feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)



# Print out the feature and importances 

[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
preds = reg.predict(X_test)
evaluate = pd.DataFrame({

    "actual" : y_test

    , "predicted" : preds

})



evaluate["error"] = evaluate["actual"] - evaluate["predicted"]



evaluate.head()
# Calculate the absolute errors

errors = abs(preds - y_test)



# Print out the mean absolute error (mae)

print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
all_data_7 = all_data_4.copy()
y = all_data_7["Deaths"]



X = all_data_7.drop(["Deaths", "FIPS", "stateFIPS", "countyFIPS_2d", "County"

                     , "State", "Risk_Cat", "I_Male", "I_Female"

                    , "A_Male", "A_Female", "NH_Male", "NH_Female"],  axis=1)



X
# train-test split

from sklearn.model_selection import train_test_split



# allocate 70% at random to training data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
from sklearn.ensemble import RandomForestRegressor



reg = RandomForestRegressor(max_depth=2, random_state=10, oob_score=True, bootstrap=True)
reg.fit(X_train, y_train)
# Get numerical feature importances

importances = list(reg.feature_importances_)



# List of tuples with variable and importance

feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(X_train, importances)]



# Sort the feature importances by most important first

feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)



# Print out the feature and importances 

[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
preds = reg.predict(X_test)
evaluate = pd.DataFrame({

    "actual" : y_test

    , "predicted" : preds

})



evaluate["error"] = evaluate["actual"] - evaluate["predicted"]



evaluate.head()
# Calculate the absolute errors

errors = abs(preds - y_test)



# Print out the mean absolute error (mae)

print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
all_data_8 = all_data_4.copy()
y = all_data_8["Deaths"]



X = all_data_8.drop(["Cases", "Deaths", "FIPS", "stateFIPS", "countyFIPS_2d", "County"

                     , "State", "Risk_Cat", "W_Male", "B_Male", "H_Male"

                     , "I_Male", "A_Male", "NH_Male"],  axis=1)



X
# train-test split

from sklearn.model_selection import train_test_split



# allocate 70% at random to training data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
from sklearn.ensemble import RandomForestRegressor



reg = RandomForestRegressor(max_depth=2, random_state=10, oob_score=True, bootstrap=True)
reg.fit(X_train, y_train)
# Get numerical feature importances

importances = list(reg.feature_importances_)



# List of tuples with variable and importance

feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(X_train, importances)]



# Sort the feature importances by most important first

feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)



# Print out the feature and importances 

[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
preds = reg.predict(X_test)
evaluate = pd.DataFrame({

    "actual" : y_test

    , "predicted" : preds

})



evaluate["error"] = evaluate["actual"] - evaluate["predicted"]



evaluate.head()
# Calculate the absolute errors

errors = abs(preds - y_test)



# Print out the mean absolute error (mae)

print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')