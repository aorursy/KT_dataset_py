# Install prerequisites and H2O.ai
!apt update && apt install -y default-jre 
!pip install requests tabulate "colorama>=0.3.8" future "ipywidgets>=7.5" pandas matplotlib
# Initialize an H2O instance
import h2o
from h2o.automl import H2OAutoML

h2o.init(max_mem_size="16G")
# Load data, which have lag features
oil_price_with_lag_csv_path = '../input/globaaichallenge2020-additional-data/Crude_oil_trend_Lag5days_20200706053126.csv'
oil_price_with_lag_data = h2o.import_file(oil_price_with_lag_csv_path)

print(oil_price_with_lag_data)
# Display details
print(oil_price_with_lag_data.describe())
# Drop NA
na_dropped_oil_price_with_lag = oil_price_with_lag_data.na_omit()
print(na_dropped_oil_price_with_lag, na_dropped_oil_price_with_lag.tail())

# Split into train data and validation data
import datetime
start_date = datetime.datetime(2016, 7, 18, 0, 0, 0)
split_date = datetime.datetime(2020, 4, 29, 0, 0, 0)
end_date = datetime.datetime(2020, 6, 22, 0, 0, 0)

train_mid_data = na_dropped_oil_price_with_lag[na_dropped_oil_price_with_lag["Date"] >= start_date]
train_oil_price_with_lag_data = train_mid_data[train_mid_data["Date"] < split_date]

val_mid_data = na_dropped_oil_price_with_lag[na_dropped_oil_price_with_lag["Date"] >= split_date]
val_oil_price_with_lag_data = val_mid_data[val_mid_data["Date"] <= end_date]
print(train_oil_price_with_lag_data, train_oil_price_with_lag_data.tail(), 
      val_oil_price_with_lag_data, val_oil_price_with_lag_data.tail())
# Separate an objective variable from the columns
x = train_oil_price_with_lag_data.columns
y = "Price"
x.remove(y)
print(x, y)
# Train 20 different models
auto_ml = H2OAutoML(max_models=20, seed=1, nfolds=0, 
                    stopping_metric="RMSE", sort_metric="RMSE", 
                    project_name="Demo", export_checkpoints_dir="./models/")
auto_ml.train(x=x, y=y, training_frame=train_oil_price_with_lag_data,
              validation_frame=val_oil_price_with_lag_data, leaderboard_frame=val_oil_price_with_lag_data)
# Display the leaderboard
leader_board = auto_ml.leaderboard
leader_board.head(rows=leader_board.nrows)
# Display the top ranked model
auto_ml.leader
# Prediction function that uses lag feature columns

import copy
from numpy import nan

def createChronologyData(predict_df, index, column_name='Price', period=5, interval=1):
    if predict_df.empty:
        print('DataFrame is empty!!')
        return
    
    out = copy.deepcopy(predict_df)#.set_index('Date')
    columns = out.columns
    top_index = out.index[0]
    insertIndex = index
    skip = 0

    # Make lag features
    for i, column in enumerate(columns):
        if column == 'Date' or column == column_name:
            skip = skip + 1
            continue
        insertIndex = index - interval * (i-skip) - 1
        if top_index > insertIndex:
            continue
        
        out.at[index, column] = out[column_name][insertIndex]

    print("out is {}".format(out) )
    return out#.reset_index()

def predict_and_creating_rag(model, test_df, column_name='Price'):

    out = test_df#copy.deepcopy(test_df)
    top = out.index[0]

    try:
        out = out.drop(column_name, axis=1)
    except KeyError:
        print("{} is none".format(column_name))
    out[column_name] = float(nan)
    coln = out.columns.tolist()
    coln = coln[0:1] + coln[-1:] + coln[1:-1]
    out = out[coln]
    
    # Predict one record by one record
    for index, item in test_df.iterrows():
        current = index - top
        out = createChronologyData(out, index, column_name=column_name)
        if isinstance(model, h2o.estimators.estimator_base.H2OEstimator):
            print(out[current:current+1])
            predict_h2o_frame = model.predict(h2o.H2OFrame(out[current:current+1].drop(column_name, axis=1)))
            predict = h2o.as_list(predict_h2o_frame, use_pandas=True)
            out.loc[index, column_name] = predict.iat[0,0]
        else:
            predict = model.predict(out[current:current+1].drop(column_name, axis=1))
            out.loc[index, column_name] = predict[0][0]
        print("predict is {}".format(predict))
    
    return out
model = auto_ml.leader
# Prediction
val_oil_price_with_lag_df = h2o.as_list(val_oil_price_with_lag_data, use_pandas=True)
predict_values \
    = predict_and_creating_rag(model=model, test_df=val_oil_price_with_lag_df, column_name='Price')
predict_values
# Rewrite "Date" values because when you convert an H2O frame to a Pandas Dataframe, you get "Date" values wrong
print(val_oil_price_with_lag_data[:,0])
prediction = val_oil_price_with_lag_data[:,0].cbind(h2o.H2OFrame(predict_values.iloc[:,1:]))
print(prediction)
# vice versa (when converting a Padas Dataframe to an H2O frame) 

import pandas as pd

oil_price_with_lag_df = pd.read_csv(oil_price_with_lag_csv_path)

date_format = '%Y-%m-%d'
train_df_date_mid \
    = oil_price_with_lag_df.Date[oil_price_with_lag_df.Date >= start_date.strftime(date_format)]
train_df_date \
    = train_df_date_mid[train_df_date_mid < split_date.strftime(date_format)]
val_df_date_mid \
    = oil_price_with_lag_df.Date[oil_price_with_lag_df.Date >= split_date.strftime(date_format)]
val_df_date \
    = val_df_date_mid[val_df_date_mid <= end_date.strftime(date_format)]
train_df_date
val_df_date
# Plot prediction with plotly
train_df = train_oil_price_with_lag_data.as_data_frame(use_pandas=True)
val_df = val_oil_price_with_lag_data.as_data_frame(use_pandas=True)
prediction_df = prediction.as_data_frame(use_pandas=True)
prediction_df.rename(columns={'predict': 'Price'}, inplace=True)

train_df['Date'] = train_df_date.values
val_df['Date'] = val_df_date.values
prediction_df['Date'] = val_df_date.values

train_df['Train, Validation or Predict'] = "Train"
val_df['Train, Validation or Predict'] = "Validation"
prediction_df['Train, Validation or Predict'] = "Predict"
print(train_df, val_df, prediction_df)
train_val_pred_df = pd.concat([train_df, val_df, prediction_df])
train_val_pred_df
import plotly.express as px
fig = px.line(train_val_pred_df, x='Date', y='Price', color='Train, Validation or Predict')
fig.show()
# Calculate RMSE
print('RMSE on validation term:', ((val_df.Price - prediction_df.Price) ** 2).mean() ** .5)