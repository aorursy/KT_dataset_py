import pandas as pd
import datetime
train_df = pd.read_csv("/kaggle/input/shopee-code-league-2020-week-4-logistics/delivery_orders_march.csv")
train_df.head()
#convert time-based columns to datetime objects
def dt_converter(x):
    if x:
        localTime = datetime.datetime.fromtimestamp(x)
        return(datetime.datetime(localTime.year, localTime.month, localTime.day))
    else:
        localTime = datetime.datetime.fromtimestamp(0)
        return(datetime.datetime(localTime.year, localTime.month, localTime.day))
#fill NaN with zeroes
train_df['2nd_deliver_attempt'] = train_df['2nd_deliver_attempt'].fillna(0)
#apply converter function
train_df['pick'] = train_df['pick'].apply(dt_converter)
train_df['1st_deliver_attempt'] = train_df['1st_deliver_attempt'].apply(dt_converter)
train_df['2nd_deliver_attempt'] = train_df['2nd_deliver_attempt'].apply(dt_converter)
#parsing buyer address (only last 1-2 words are relevant)
def addressParser(x):
    x = x.lower()
    if 'manila' in x.lower().split(" ")[-2:] and 'metro' in x.lower().split(" ")[-2:]:
        return "metro manila"
    elif 'luzon' in x.lower().split(" ")[-2:]:
        return "luzon"
    elif 'visayas' in x.lower().split(" ")[-2:]:
        return "visayas"
    elif 'mindanao' in x.lower().split(" ")[-2:]:
        return "mindanao"
    else:
        return "unknown"
#apply parser function
train_df['buyeraddress'] = train_df['buyeraddress'].apply(addressParser)
train_df['selleraddress'] = train_df['selleraddress'].apply(addressParser)
#generate SLA matrix
sla_matrix = pd.DataFrame({'metro manila':[3,5,7,7], 'luzon':[5,5,7,7], 'visayas':[7,7,7,7], 'mindanao':[7,7,7,7]}, 
                          index=['metro manila', 'luzon', 'visayas', 'mindanao'])
#uses SLA matrix to determine days allowed for delivery
def daysAllowed(x):
    return sla_matrix.loc[x['selleraddress'], x['buyeraddress']]
#add new column
train_df['daysAllowed'] = train_df.apply(daysAllowed, axis=1)
#check if 1st attempt is not more than daysAllowed days from pick date
def first_late_checker(x):
    NON_WEEKDAYS = [datetime.datetime(2020,3,25), datetime.datetime(2020,3,30), datetime.datetime(2020,3,31)]
    picked = x['pick']
    first_atmpt= x['1st_deliver_attempt']
    
    counter = 0 
    while counter < x['daysAllowed']:
        next_day = picked + datetime.timedelta(days=1)
        if next_day.weekday() < 6 and next_day not in NON_WEEKDAYS:
            picked = next_day
            counter += 1
        else:
            picked = next_day
    if first_atmpt > picked:
        return 1
    else:
        return 0
#check if 2nd attempt is not more than 3 days from 1st attempt
def second_late_checker(x):
    NON_WEEKDAYS = [datetime.datetime(2020,3,25), datetime.datetime(2020,3,30), datetime.datetime(2020,3,31)]
    first_atmpt = x['1st_deliver_attempt']
    sec_atmpt= x['2nd_deliver_attempt']
    
    counter = 0
    if sec_atmpt > datetime.datetime(1990,1,1): 
        while counter < 3:
            next_day = first_atmpt + datetime.timedelta(days=1)
            if next_day.weekday() < 6 and next_day not in NON_WEEKDAYS:
                first_atmpt = next_day
                counter += 1
            else:
                first_atmpt = next_day
        if sec_atmpt > first_atmpt:
            return 1
        else:
            return 0
    else:
        return 0
train_df['is_late_1'] = train_df.apply(first_late_checker, axis=1)
train_df['is_late_2'] = train_df.apply(second_late_checker, axis=1)
#returns true for is late column if at least one of the above late checkers returns 1
def final_late_checker(x):
    if (x['is_late_1'] == 1) | (x['is_late_2'] == 1):
        return 1
    else:
        return 0
train_df['is_late'] = train_df.apply(final_late_checker, axis=1)
train_df[['orderid', 'is_late']].to_csv("mysubmission.csv", index=False)