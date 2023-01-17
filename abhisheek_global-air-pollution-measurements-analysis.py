import pandas as pd
from google.cloud import bigquery
from bq_helper import BigQueryHelper
QUERY = """
        SELECT location, city, country,pollutant, value, unit, source_name, latitude, longitude, averaged_over_in_hours,  timestamp
        FROM `bigquery-public-data.openaq.global_air_quality`
        ORDER BY value DESC
        """
bq_assistant = BigQueryHelper('bigquery-public-data', 'openaq')
df = bq_assistant.query_to_pandas(QUERY)
aq_data=df
aq_data.head(3)
# print a summary of the data in df 
print(aq_data.describe())
aq_value_data = aq_data.value
print(aq_value_data.head())
columns_of_interest = ['value', 'latitude', 'longitude', 'averaged_over_in_hours']
three_columns_of_data = aq_data[columns_of_interest]
three_columns_of_data.describe()
aq_data['value'].fillna(0, inplace=True)
y = aq_data.value
y
aq_data['averaged_over_in_hours'].fillna(0, inplace=True)
aq_predictors = ['latitude', 'longitude', 'averaged_over_in_hours']

X = aq_data[aq_predictors]
X
from sklearn.tree import DecisionTreeRegressor
aq_model = DecisionTreeRegressor()
aq_model.fit(X, y)
print("Making predictions for the following 5 latitudes and longitudes:")
print(X.head())
print("The predictions are")
print(aq_model.predict(X.head()))
from sklearn.metrics import mean_absolute_error

predicted_aq = aq_model.predict(X)
mean_absolute_error(y, predicted_aq)
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)
aq_model= DecisionTreeRegressor()
aq_model.fit(train_X, train_y)
val_predictions = aq_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(predictors_train, targ_train)
    preds_val = model.predict(predictors_val)
    mae = mean_absolute_error(targ_val, preds_val)
    return(mae)
# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor()
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))
