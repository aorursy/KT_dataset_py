# Load common libraries:
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import folium

# set some display options:
sns.set(style="whitegrid")
pd.set_option("display.max_columns", 36)

# load data:
file_path = "../input/hotel-booking-demand/hotel_bookings.csv"
full_data = pd.read_csv(file_path)

# Replace missing values:
nan_replacements = {"children": 0.0, "country": "Unknown", "agent": 0, "company": 0}
full_data_cln = full_data.fillna(nan_replacements)

# "meal" contains values "Undefined", which is equal to SC.
full_data_cln["meal"].replace("Undefined", "SC", inplace=True)

# Get rid of bookings for 0 adults, 0 children, and 0 babies:
zero_guests = list(full_data_cln.loc[full_data_cln["adults"]
                   + full_data_cln["children"]
                   + full_data_cln["babies"]==0].index)
full_data_cln.drop(full_data_cln.index[zero_guests], inplace=True)

# Delete a record with ADR greater than 5000
full_data_cln = full_data_cln[full_data_cln['adr'] < 5000]
ax = sns.boxplot(x=full_data_cln['adr'])
cor_mat = full_data.corr()
fig, ax = plt.subplots(figsize=(17,7))
sns.heatmap(cor_mat, ax=ax, cmap="RdBu", center=0, linewidths=0.1)
cancel_corr = full_data.corr()["is_canceled"]
cancel_corr.sort_values(ascending=False)[1:]
#cancel_corr.abs().sort_values(ascending=False)[1:]
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier

import numpy as np
# Separate features and predicted value
y = full_data["is_canceled"] # what we want to predict
X = full_data.drop(["is_canceled"], axis=1) # remove target variable from features
# 70 % for training, 30 % for validation
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.7,
                                                                test_size=0.3, random_state=0)
num_features = ["lead_time",
                "arrival_date_week_number",
                "arrival_date_day_of_month",
                "stays_in_weekend_nights",
                "stays_in_week_nights",
                "adults",
                "children",
                "babies",
                "is_repeated_guest",
                "previous_cancellations",
                "previous_bookings_not_canceled",
                "agent",
                "company",
                "required_car_parking_spaces",
                "total_of_special_requests",
                "adr"]
cat_features = ["hotel",
                "arrival_date_month",
                "meal",
                "market_segment",
                "distribution_channel",
                "reserved_room_type",
                "deposit_type",
                "customer_type"]
X_train = X_train_full[num_features + cat_features].copy()
X_valid = X_valid_full[num_features + cat_features].copy()

# preprocess numerical features: 
num_transformer = SimpleImputer(strategy="constant") # not really necessary, as we should not have any missing values

# Preprocessing for categorical features:
cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
    ("onehot", OneHotEncoder(handle_unknown='ignore'))])

# Bundle preprocessing for numerical and categorical features:
preprocessor = ColumnTransformer(transformers=[('num', num_transformer, num_features),
                                               ('cat', cat_transformer, cat_features)])
# Define Random Forest classifier:
rfc_model = RandomForestClassifier(random_state=0,n_jobs=-1)
rfc_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', rfc_model)])

# Preprocessing of training data, fit model:
rfc_pipeline.fit(X_train, y_train)
from sklearn.metrics import accuracy_score

# Preprocessing of validation data, get predictions:
y_pred = rfc_pipeline.predict(X_valid)
y_pred
# Evaluate the model:
score = accuracy_score(y_valid, y_pred)
print("Random Forest accuracy_score: ", score)
full_data_cln["is_canceled"].mean()
from sklearn.metrics import balanced_accuracy_score

balanced_accuracy_score(y_valid, y_pred)
confusion_matrix(y_valid, y_pred)
true = pd.Categorical(list(np.where(np.array(y_valid) == 1, 'cancelled','not cancelled')), categories = ['cancelled','not cancelled'])
pred = pd.Categorical(list(np.where(np.array(y_pred) == 1, 'cancelled','not cancelled')), categories = ['cancelled','not cancelled'])

pd.crosstab(pred, true, 
            rownames=['pred'], 
            colnames=['Actual'], margins=False, margins_name="Total")
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

print("Precision: ", precision_score(y_valid, y_pred))
print("Recall: ", recall_score(y_valid, y_pred))
print("F1: ", f1_score(y_valid, y_pred))
from sklearn.metrics import plot_precision_recall_curve

disp = plot_precision_recall_curve(rfc_pipeline, X_valid, y_valid)
disp.ax_.set_title('Precision-Recall curve')
from sklearn.ensemble import RandomForestRegressor 

rfe_model = RandomForestRegressor(n_estimators = 100, random_state = 0) 
rfe_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', rfe_model)])
rfe_pipeline.fit(X_train, y_train)
y_pred = rfe_pipeline.predict(X_valid)
y_pred
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

print('Mean Absolute Error:', mean_absolute_error(y_valid, y_pred).round(4))  
print('Mean Squared Error:', mean_squared_error(y_valid, y_pred).round(4))  
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_valid, y_pred)).round(4))
print('r2_score:', r2_score(y_valid, y_pred).round(3))