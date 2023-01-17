import numpy as np

import pandas as pd

import os

import shap

import sklearn

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import preprocessing

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import LabelEncoder

import lime

import lime.lime_tabular

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score, accuracy_score

from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb

from sklearn.linear_model import Ridge

from sklearn.impute import SimpleImputer

shap.initjs()
data = pd.read_csv("/kaggle/input/szeged-weather/weatherHistory.csv")

data.describe()

# Note: The column named "Loud Cover" is not making any sense as it is only "0" I will drop it during Preprocessing.
data.info()

# Note: The data does not require any imputing or interpolation as it has no null rows at all.

# Note: Most of useful columns are numeric, no need to overthink about encoding as some task won't require any.
data.head(3)

# Note: the Dataset is designed to be "Hourly". This is good in terms of details, but I rather something less complex. So, I will change it to "Daily" on next steps.
def simplify_summaries(base_summary):

    base_split = base_summary.split(" ")

    removals_list = ["Light","Dangerously","Partly","Mostly","and"]

    to_be_replaced_list = ["Breezy","Drizzle","Overcast"]

    replacement_list = ["Windy","Rain","Cloudy"]

    for removal in removals_list: 

        if removal in base_split:

            base_split.remove(removal)

            

    for i in range(len(to_be_replaced_list)):

        if to_be_replaced_list[i] in base_split:

            base_split.remove(to_be_replaced_list[i])

            base_split.append(replacement_list[i])

        

    base_split.sort()

    return " ".join(base_split)
data.Summary = data.Summary.apply(simplify_summaries)

data.head(3)

# much better now as we reduced complexity of it dramatically.
# Dropping the column named "Loud Cover" on general dataset "data"

data.drop(columns=["Loud Cover"], inplace=True)
# Changing the original "Hourly" dataset to new and simpler "Daily"

# Fixing the Formatted Date for pandas usage.

data['Formatted Date'] = pd.to_datetime(data['Formatted Date'], utc=True)

data.sort_values(by=['Formatted Date'], inplace=True, ascending=True)
data.head(4)
# Grouping by days to achieve "Daily" dataset on what's left as numerical columns for "Sliding Windows to predict Temp" task. 

swt_data = data.groupby([data['Formatted Date'].dt.date]).mean()

swt_data["Summary"] = data["Summary"].groupby([data['Formatted Date'].dt.date]).agg(lambda x:x.value_counts().index[0])

le = LabelEncoder()

swt_data.Summary = le.fit_transform(swt_data.Summary)
# Results are sorted and daily.

swt_data.head() 
# Checking the results and it is clearly worked.

swt_data.describe()
# Plotting approx. 2 years to have an idea about what we are working with.

plt.figure(figsize=(24,8))

plt.plot(swt_data["Temperature (C)"][:740])

plt.grid()

plt.show()
ROLLING_MEAN_PARAMETER = 3

swt_data[["Temperature (C)","Apparent Temperature (C)","Humidity","Wind Speed (km/h)", "Wind Bearing (degrees)", "Visibility (km)", "Pressure (millibars)"]] = np.round(swt_data[["Temperature (C)","Apparent Temperature (C)","Humidity","Wind Speed (km/h)", "Wind Bearing (degrees)", "Visibility (km)", "Pressure (millibars)"]].rolling(ROLLING_MEAN_PARAMETER).mean(),3)

swt_data.dropna(inplace=True) # dropping the null days that are created by rolling mean
# Plotting approx. 2 years to have an idea about what we are working with after rolling mean

plt.figure(figsize=(24,8))

plt.plot(swt_data["Temperature (C)"][:740])

plt.grid()

plt.show()
# Now I will design the dataset into more trainable sliding windows format.

N_DAYS_BEFORE = 5

swt_train = pd.DataFrame()



for day in range(N_DAYS_BEFORE-1,len(swt_data)):

    for i in reversed(range(1,N_DAYS_BEFORE)):

        for j in swt_data.columns:

            col_name = str(j) + " - " + str(i)

            swt_train.loc[day, col_name] = (swt_data[j][day-i])
# each row consist from previous 5 days with details.

swt_train.head()
# first part of the shapes must be the same to labels.

print(swt_train.shape)
# Prepearing the labels for SWT task

# ignoring the first 4 days to match training data & only getting values so we won't have issues with date index later on.

swt_labels = swt_data["Temperature (C)"][N_DAYS_BEFORE-1:].values

# first part of the shapes must be the same to train.

print(swt_labels.shape)
# Temperature (C) - 1  of 22th feature should be equal to the value of 23th label (today = tomorrow of yesterday)

print(" -- Features -- \n",swt_train.iloc[23])

print("\n -- Label -- \n", swt_labels[22])
# Splitting train and test to be able to evaluate properly with some train test ratio.

swt_train_x, swt_test_x, swt_train_y, swt_test_y = train_test_split(swt_train,swt_labels, test_size=0.1)
# Checking the shapes for safety

print("shape of training dataset features: ",swt_train_x.shape)

print("shape of training dataset labels: ",swt_train_y.shape)

print("shape of testing dataset features: ",swt_test_x.shape)

print("shape of testing dataset labels: ",swt_test_y.shape)
# Prepearing the labels for SWT task

# ignoring the first 4 days to match training data & only getting values so we won't have issues with date index later on.

sws_labels = swt_data["Summary"][N_DAYS_BEFORE-1:].values

# first part of the shapes must be the same to train.

print(sws_labels.shape)
# splitting (75/25) as usual

sws_train_x, sws_test_x, sws_train_y, sws_test_y = train_test_split(swt_train, sws_labels, random_state=41, test_size=0.25)
sws_train_x
# Checking the shapes for safety

print("shape of training dataset features: ",sws_train_x.shape)

print("shape of training dataset labels: ",sws_train_y.shape)

print("shape of testing dataset features: ",sws_test_x.shape)

print("shape of testing dataset labels: ",sws_test_y.shape)
# For this approach I will only use 1 column. this will be the "Temperature (C)"

all_temps = swt_data["Temperature (C)"].values

train_temps = []

label_temps = []

for i in range(len(all_temps)-30):

    label_temps.append(all_temps[i+30])

    train_temps.append(all_temps[i:i+30])

    

train_temps = np.array(train_temps)

label_temps = np.array(label_temps)
# last of the tomorrow's array should be same as the today's label 

print(train_temps[45])

print(label_temps[44]) 
# Splitting the train and test 

sdt_train_x = train_temps[:-400]

sdt_test_x = train_temps[-400:]

sdt_train_y = label_temps[:-400]

sdt_test_y = label_temps[-400:]
# Checking the shapes for safety

print("shape of training dataset features: ",sdt_train_x.shape)

print("shape of training dataset labels: ",sdt_train_y.shape)

print("shape of testing dataset features: ",sdt_test_x.shape)

print("shape of testing dataset labels: ",sdt_test_y.shape)
rf_model = RandomForestRegressor(max_depth=10)

rf_model.fit(swt_train_x,swt_train_y)
my_imputer = SimpleImputer()

sws_train_x_imp = my_imputer.fit_transform(sws_train_x)

sws_test_x_imp = my_imputer.transform(sws_test_x)



my_model = xgb.XGBClassifier(n_estimators=1000, 

                            max_depth=4, 

                            eta=0.05, 

                            base_score=sws_train_y.mean())

hist = my_model.fit(sws_train_x_imp, sws_train_y, 

                    early_stopping_rounds=5, 

                    eval_set=[(sws_test_x_imp, sws_test_y)], eval_metric='mlogloss', 

                    verbose=10)
lr_model = Ridge()

lr_model.fit(sdt_train_x,sdt_train_y)
swt_pred_y = rf_model.predict(swt_test_x)

print("r_square score of the RandomForestRegressor model : ",r2_score(swt_test_y,swt_pred_y))
explainer = shap.TreeExplainer(rf_model)

shap_values = explainer.shap_values(swt_train_x)



shap.summary_plot(shap_values, swt_train_x, plot_type="bar");
shap.summary_plot(shap_values, swt_train_x)
a = shap.force_plot(explainer.expected_value, shap_values[100,:], swt_train_x.iloc[100,:])

display(a)



b = shap.force_plot(explainer.expected_value, shap_values[80,:], swt_train_x.iloc[80,:])

display(b)



c = shap.force_plot(explainer.expected_value, shap_values[70,:], swt_train_x.iloc[70,:])

display(c)



d = shap.force_plot(explainer.expected_value, shap_values[90,:], swt_train_x.iloc[90,:])

display(d)
shap.force_plot(explainer.expected_value, shap_values, swt_train_x)
print("prediction : ",rf_model.predict(swt_test_x.iloc[77].values.reshape(1,32)))

print("ground truth : ",swt_test_y[77])

# very accurate prediction.
y_pred = my_model.predict(sws_test_x_imp)

accuracy_score(y_pred, sws_test_y)
predict_fn = lambda x: my_model.predict_proba(x).astype(float)

explainer = lime.lime_tabular.LimeTabularExplainer(sws_test_x_imp, feature_names=sws_test_x.columns, class_names=range(0,14), verbose=True, mode='classification')
print(le.inverse_transform(my_model.predict(sws_test_x_imp)[60].ravel()))

print(le.inverse_transform(my_model.predict(sws_test_x_imp)[0].ravel()))

print(le.inverse_transform(my_model.predict(sws_test_x_imp)[124].ravel()))

# the indexes will be used later on.
foggy_instance = sws_test_x.iloc[124].values

cloudy_instance = sws_test_x.iloc[0].values

clear_instance = sws_test_x.iloc[60].values
exp1 = explainer.explain_instance(foggy_instance, predict_fn, num_features=5, labels=range(0,6))

exp2 = explainer.explain_instance(cloudy_instance, predict_fn, num_features=5, labels=range(0,6))

exp3 = explainer.explain_instance(clear_instance, predict_fn, num_features=5, labels=range(0,6))
exp1.show_in_notebook()
exp1.as_pyplot_figure(label=3)   # for class of 3, which is foggy

plt.show()
exp1.as_list(label=3)
print(exp1.as_map())
# label 0 is "clear"

exp3.as_pyplot_figure(label=0)

plt.show()
sdt_pred_y = lr_model.predict(sdt_test_x)

print("r_square score of the Ridge Regression model : ",r2_score(sdt_test_y,sdt_pred_y))   # the model performs really good.
plt.figure(figsize=(20,6))

plt.plot(sdt_pred_y)

plt.plot(sdt_test_y)

plt.tight_layout()
# efe ergün