import os
import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import seaborn as sns
import matplotlib.pyplot as plt
def load_df(csv_path):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'}
                     )
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    return df
train_df = load_df('../input/train.csv')
test_df = load_df("../input/test.csv")

train_df.head()
def check_null_values(data_frame,dataset_name):
    null_count = ((data_frame[data_frame.columns] == 'not available in demo dataset') | (data_frame[data_frame.columns].isna())).sum()
    
    #print("\n--------------------{} Dataset Null Count-----------------------".format(dataset_name))
    #print(null_count)

    keys = null_count.keys()
    df_size = len(data_frame) 
    null_col = [col for col in keys if null_count[str(col)] > 0 if null_count[str(col)] == df_size]
    print("No. of columns with null values in {} dataset : {}".format(dataset_name,len(null_col)))
    return null_col,null_count
    
train_null_col,train_null_count = check_null_values(train_df,"Training")
test_null_col,test_null_count = check_null_values(test_df,"Testing")
col_labels = ['Column Name','Null Count']
null_count_df = pd.DataFrame(columns=col_labels,data={col_labels[0] : list(train_null_count.keys()),
                                                     col_labels[1] : list(train_null_count.values)})
fig,ax = plt.subplots(figsize=(20,8))
null_graph = sns.barplot(x=col_labels[0],y=col_labels[1],data=null_count_df,palette=sns.color_palette("Paired",len(null_count_df)),ax=ax,capsize=0.5)
null_graph.set_xticklabels(labels=[label.split(".")[-1] for label in null_count_df['Column Name']],rotation=60)

print("\nColumns having unnecessary information from training set \n" ,train_null_col)
print("\nColumns having unnecessary information from testing set \n" ,test_null_col)
train_df.drop(columns=train_null_col,inplace=True)
test_df.drop(columns=test_null_col,inplace=True)
train_df.drop(columns=['trafficSource.campaignCode','visitStartTime'],inplace=True)
test_df.drop(columns=['visitStartTime'],inplace=True)

print("Training Set Shape after dropping columns",train_df.shape)
print("Testing Set Shape after dropping columns",test_df.shape)
def handle_null(df,name) :
    df['totals.newVisits'].fillna(0,inplace=True)
    df['totals.pageviews'].fillna(0,inplace=True)
    
    null_col,null_count = check_null_values(df,name)
    df_len = len(df)
    
    null_percent = [round((null_count_val/df_len)* 100,2) for null_count_val in null_count.values]
    null_percent = pd.DataFrame(columns=['Column Name','Null Percent'],data={ 'Column Name' : df.keys(),'Null Percent' : null_percent})
    null_percent.set_index('Column Name',inplace=True)
    return null_percent

train_df['totals.transactionRevenue'].fillna(0,inplace=True)
train_col_null_percent = handle_null(train_df,"Training")
test_col_null_percent = handle_null(test_df,"Testing")

display(train_col_null_percent)

train_cols_to_discard = [column for column in train_col_null_percent.index.values if int(train_col_null_percent.at[column,'Null Percent']) > 22]
train_cols_to_discard.extend(['fullVisitorId','totals.visits','socialEngagementType','sessionId','visitId','geoNetwork.networkDomain','trafficSource.campaign'])
#print(train_cols_to_discard)
train_df.drop(train_cols_to_discard,axis=1,inplace=True)

test_cols_to_discard = [column for column in test_col_null_percent.index.values if int(test_col_null_percent.at[column,'Null Percent']) > 22]
test_cols_to_discard.extend(['socialEngagementType','totals.visits','sessionId','visitId','geoNetwork.networkDomain','trafficSource.campaign'])
test_df.drop(test_cols_to_discard,axis=1,inplace=True)

train_df.head()
train_quarter = [int((((date%10000)/100)/4)+1) for date in train_df['date']]
test_quarter = [int((((date%10000)/100)/4)+1) for date in test_df['date']]

def getDay(date):
    return date%100

train_df = train_df.assign(quarterOfYear=train_quarter)
test_df = test_df.assign(quarterOfYear=test_quarter)

train_df['date'] = train_df['date'].apply(getDay)
test_df['date'] = test_df['date'].apply(getDay)

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
col_to_encode = ['channelGrouping','device.browser','device.deviceCategory','device.isMobile','device.operatingSystem','geoNetwork.continent','geoNetwork.country','geoNetwork.subContinent', 'trafficSource.medium', 'trafficSource.source']
le = LabelEncoder()

def encode_cols(data_frame,columns):
    for column in col_to_encode :
        data_frame[column] = le.fit(data_frame[column]).transform(data_frame[column]) 

encode_cols(train_df,col_to_encode)
encode_cols(test_df,col_to_encode)
        
training_col = list(train_df.keys())
training_col.remove('totals.transactionRevenue')
full_visitor_ids = test_df['fullVisitorId'].values

test_col = list(test_df.keys())
test_col.remove('fullVisitorId')


X_train = train_df[training_col] 
y_train = train_df['totals.transactionRevenue']


#print(type(train_df['device.browser'][0]))
#print([col_val for col_val in train_df.columns.values if isinstance(train_df[col_val][0],str)])


#train_df.apply(le.fit_transform)
    
#print(le.fit(train_df['device.deviceCategory']).transform(train_df[['device.deviceCategory','device.browser']]))
train_df.head()
test_df.head()

#X_train.head()

from sklearn.ensemble import RandomForestRegressor

#lr_model = LinearRegression()
rf = RandomForestRegressor(n_estimators = 500, random_state = 42)
#model = lr_model.fit(X_train,y_train)
model = rf.fit(X_train,y_train)
#print(model.coef_)

prediction = rf.predict(test_df[test_col])
print(len(prediction))

import math

test_df = test_df.assign(predictedRevenue=prediction)
groupby_visitorId = test_df.groupby('fullVisitorId').sum()

def calculate_log(revenue):
    return math.log10(abs(revenue) + 1)

predicted_df = groupby_visitorId['predictedRevenue'].apply(calculate_log)
predicted_df_new = pd.DataFrame({'fullVisitorId' : list(predicted_df.keys()),'PredictedLogRevenue' : predicted_df.values})
predicted_df_new.to_csv('output_pred.csv',index=False)
