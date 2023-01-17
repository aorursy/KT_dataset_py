import pandas as pd
from datetime import datetime
from sklearn import preprocessing
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import numpy as np 
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
path = '/kaggle/input/covid19-global-forecasting-week-5/'
train_data = pd.read_csv(path+"train.csv")
train_data
train_data["County"].count()
train_data["Province_State"].count()
train_data["Country_Region"].count()
train_data["Population"].count()
train_data["Weight"].count()
train_data["Date"].count()
train_data["Target"].count()
train_data.dropna(subset = ["County", "Province_State", "Country_Region", "Population", "Weight", "Date", "Target"], inplace=True)
train_data.reset_index(drop=True, inplace=True)
train_data
train_attributes = train_data[["County", "Province_State", "Country_Region", "Population", "Weight", "Date"]]
le_train_county = LabelEncoder()
le_train_province_state = LabelEncoder()
le_train_country_region = LabelEncoder()
le_train_date = LabelEncoder()

train_attributes["County_n"] = le_train_county.fit_transform(train_attributes["County"])
train_attributes["Province_State_n"] = le_train_province_state.fit_transform(train_attributes["Province_State"])
train_attributes["Country_Region_n"] = le_train_country_region.fit_transform(train_attributes["Country_Region"])
train_attributes["Date_n"] = le_train_date.fit_transform(train_attributes["Date"])
train_attributes
train_attributes_n = train_attributes.drop(["County", "Province_State", "Country_Region", "Date"], axis = "columns")
train_attributes_n
train_target_class = train_data[["Target"]]
le_target = LabelEncoder()
train_target_class["Target_n"] = le_target.fit_transform(train_target_class["Target"])
train_target_class
train_target_class_n = train_target_class.drop(["Target"], axis = "columns")
train_target_class_n
model = tree.DecisionTreeClassifier()
model.fit(train_attributes_n, train_target_class_n)
model.score(train_attributes_n, train_target_class_n)
test_data = pd.read_csv(path+"test.csv")
test_data
test_data.dropna(subset = ["County", "Province_State", "Country_Region", "Population", "Weight", "Date", "Target"], inplace=True)
test_data.reset_index(drop=True, inplace=True)
test_data
test_attributes = test_data[["County", "Province_State", "Country_Region", "Population", "Weight", "Date"]]
le_test_county = LabelEncoder()
le_test_province_state = LabelEncoder()
le_test_country_region = LabelEncoder()
le_test_date = LabelEncoder()

test_attributes["County_n"] = le_test_county.fit_transform(test_attributes["County"])
test_attributes["Province_State_n"] = le_test_province_state.fit_transform(test_attributes["Province_State"])
test_attributes["Country_Region_n"] = le_test_country_region.fit_transform(test_attributes["Country_Region"])
test_attributes["Date_n"] = le_test_date.fit_transform(test_attributes["Date"])
test_attributes
test_attributes_n = test_attributes.drop(["County", "Province_State", "Country_Region", "Date"], axis = "columns")
test_attributes_n
test_target_class = test_data[["Target"]]
le_test_target = LabelEncoder()
test_target_class["Target_n"] = le_test_target.fit_transform(test_target_class["Target"])
test_target_class
test_result_arr = []
score = 0
tuple_num = len(test_attributes_n)

for i in range(tuple_num):
    result = model.predict([test_attributes_n.loc[i]])[0]
    test_result_arr.append(result)
    if(result == test_target_class.loc[i][1]):
        score = score + 1
score = score / tuple_num
score
test_attributes["Calculated_Result"] = test_result_arr
test_attributes
unique_states = test_attributes.Province_State.unique()

result_table = pd.DataFrame(columns=("Provice_State", "ConfirmedCases", "Fatalities"))

for i in range(len(unique_states)):
    zero_count = len(test_attributes[(test_attributes.Province_State == unique_states[i]) & \
                          (test_attributes.Calculated_Result == 0)])
    one_count = len(test_attributes[(test_attributes.Province_State == unique_states[i]) & \
                          (test_attributes.Calculated_Result == 1)])
    result_table.loc[i] = [unique_states[i], zero_count, one_count]
result_table