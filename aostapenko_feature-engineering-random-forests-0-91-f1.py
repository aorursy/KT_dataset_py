import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
import numpy as np
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, plot_confusion_matrix, precision_score, recall_score, accuracy_score
## load the data
hotels = pd.read_csv('/kaggle/input/hotel-booking-demand/hotel_bookings.csv')
hotels.columns
## drop unnecessary columns and rows with missing children, country values 
drop_cols = ['reservation_status_date', 'reservation_status', 'agent', 'company','country']
hotels = hotels.drop(columns=drop_cols)
hotels= hotels.dropna(0, subset=['children']).reset_index(drop=True)
# feature engineering -- first do one hot encodings of the hotel, meal, country, market segment, distribution
#                         channel, deposit type, customer type columns 
ohe = OneHotEncoder()
to_encode = pd.concat([hotels['hotel'], hotels['meal'], hotels['market_segment'],
                      hotels['distribution_channel'], hotels['deposit_type'], hotels['customer_type'], hotels['arrival_date_year']], axis=1)
## encoding the months... 
# we can start January at (1,0), increasing by 30degrees (pi/6)
months = ['January', 'February', 'March', 
          'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
month_enc = {}
degr = 0
for month in months: 
    month_enc[month] = (math.cos(degr), math.sin(degr))
    degr+=(math.pi)/6
# build new column names for the encoded data
unique_vals = []
for col in to_encode:
    uniquevals = ['_'.join([str(val), col]) for val in list(to_encode[col].unique())]
    unique_vals += uniquevals
# create a df of the encoded data
encoded_arr = ohe.fit_transform(to_encode).toarray()
encoded_df = pd.DataFrame(data=encoded_arr, columns = unique_vals)
# add the encoded month information to the encoded_df with the one hot vectors
all_reserved_months = hotels['arrival_date_month'].tolist()
all_month_xs = []
all_month_ys = []
for month in all_reserved_months: 
    x, y = month_enc[month]
    all_month_xs.append(x)
    all_month_ys.append(y)
encoded_df['arrival_date_month_X'] = all_month_xs
encoded_df['arrival_date_month_Y'] = all_month_ys
# z-transform the numerical columns and put them into the encoded df
ss= StandardScaler()
# add the original numerical columns back into the encoded dataframe
# ignore columns that we just encoded! 
encoded_cols = ['hotel', 'meal', 'country', 'market_segment', 
                'distribution_channel', 'deposit_type', 'customer_type',
                'assigned_room_type', 'reserved_room_type', 'arrival_date_month', 'arrival_date_year']
for col in hotels.columns: 
    if col not in encoded_cols:
        if col!= 'is_canceled':
            encoded_df[col] = ss.fit_transform(np.asarray(hotels[col]).reshape(-1,1))
        else: 
            encoded_df[col] = hotels[col]
# bootstrap the original data 
bootstrapped = encoded_df.sample(frac=1, replace=True, random_state=123)
# shuffle in place
bootstrapped=bootstrapped.sample(frac=1, random_state=123)
# split into top-level train/test , 80-20 split
train = bootstrapped[:int(len(bootstrapped)*0.8)]
test = bootstrapped[int(len(bootstrapped)*0.8):]
x_data_cols = [col for col in bootstrapped.columns if col!= 'is_canceled']
def get_feature_importances(tree_model, x_columns):
    df = pd.DataFrame({'weight':tree_model.feature_importances_.tolist(), 'features':list(x_columns)})
    df = df.sort_values(by='weight', ascending=False)
    return df
def evaluate_test(test_x, test_y, model):
    preds = model.predict(test_x)
    print("F1 score: {}, Recall Score: {}, Precision Score:{}, Accuracy: {} ".format(f1_score(test_y, preds), recall_score(test_y, preds), precision_score(test_y, preds),
                                                                                     accuracy_score(test_y, preds)))
# split into train/test
train_x = train[x_data_cols]
test_x = test[x_data_cols]
train_y = train['is_canceled']
test_y = test['is_canceled']
optimal = RandomForestClassifier(max_depth = 30).fit(train_x, train_y)
evaluate_test(test_x, test_y, optimal)
plot_confusion_matrix(optimal, test_x, test_y, values_format = 'd', display_labels=['Not Canceled', 'Canceled'])
feats_df = get_feature_importances(optimal, x_data_cols)
feats_df.head(10)