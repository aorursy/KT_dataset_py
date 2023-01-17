import numpy as np
import pandas as pd
from datetime import datetime,timedelta
raw_data=pd.read_csv("../input/shopee-code-league-20/_DA_Order_Brushing/order_brush_order.csv")
raw_data
df=raw_data.copy()
df.head()
### analysing about data
df.info()
### no missing value in any column
### check for unique values
len(df["orderid"].unique())
len(df["shopid"].unique())
len(df["userid"].unique())
type(df["event_time"][0])
### dropping orderid
dropped_df=df.drop(["orderid"],axis=1)
dropped_df.head()
### changing event_time from object to date_time object
dates_times=pd.to_datetime(dropped_df["event_time"])
dates_times.head()
date_time_df=pd.concat([dropped_df.iloc[:,:-1],dates_times],axis=1)
date_time_df.head()
date_time_df.info()
date_time_df.sort_values(["shopid","event_time"],inplace=True)
### sorting the dataframe in ascending order based on shopid 
### so that details corresponding to each shopid can be arranged together and also in ascending order of time
date_time_df.reset_index(drop=True,inplace=True)
date_time_df
date_time_df["shopid"].value_counts()
### function to analyze order brushing
def order_brushing_analysis(shopid_df):
    start_time=shopid_df["event_time"].iloc[0]
    end_time=shopid_df["event_time"].iloc[-1]
    order_brushing_dict={}
    suspected_users_str=""
    suspected_users_list=[]
    
    for i in range(shopid_df.shape[0]):
                
        if shopid_df["event_time"].iloc[i]<end_time-timedelta(hours=1):
            one_hour=shopid_df["event_time"].iloc[i]+timedelta(hours=1)
            one_hour_df=shopid_df[shopid_df["event_time"].between(shopid_df["event_time"].iloc[i],one_hour)]
            concentrate_rate=one_hour_df.shape[0]/len(one_hour_df["userid"].unique())
            
            if concentrate_rate>=3:
                unique_users=one_hour_df["userid"].value_counts()
                suspected_users=unique_users[unique_users==unique_users.max()].index
                for users in np.unique(suspected_users):
                    suspected_users_list.append(users)
                    
        else:
            one_hour=shopid_df["event_time"].iloc[i]+timedelta(hours=1)
            one_hour_df=shopid_df[shopid_df["event_time"].between(shopid_df["event_time"].iloc[i],one_hour)]
            concentrate_rate=one_hour_df.shape[0]/len(one_hour_df["userid"].unique())
            
            if concentrate_rate>=3:
                unique_users=one_hour_df["userid"].value_counts()
                suspected_users=unique_users[unique_users==unique_users.max()].index
                for users in np.unique(suspected_users):
                    suspected_users_list.append(users)
                
            break
    if len(suspected_users_list)==0:
        order_brushing_dict[shopid_df["shopid"].iloc[0]]="0"
    else:
        suspected_users_list.sort()
        for users in suspected_users_list:
            suspected_users_str+=str(users)+"&"
        order_brushing_dict[shopid_df["shopid"].iloc[0]]=suspected_users_str[:-1]
                    
    return order_brushing_dict
            
                
final_dict={}
for shoperid in date_time_df["shopid"].unique():
    new_dict=order_brushing_analysis(date_time_df[date_time_df["shopid"]==shoperid])
    final_dict.update(new_dict)
final_df=pd.DataFrame()
final_df["shopid"]=list(final_dict.keys())
final_df["userid"]=list(final_dict.values())
final_df
### This is my first submission to any kaggle task.
### Please suggest if any improvement or optimization is required.
### Please upvote
