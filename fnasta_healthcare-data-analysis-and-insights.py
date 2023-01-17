import datetime

from dateutil.rrule import rrule, MONTHLY



def list_months_in_date(start_date: datetime, end_date : datetime) -> list :

    strt_dt = datetime.datetime.strptime(start_date, "%d-%b-%y")

    end_dt = datetime.datetime.strptime(end_date, "%d-%b-%y")

    dates = [dt for dt in rrule(MONTHLY, dtstart=strt_dt, until=end_dt)]

    distinct_months = []

    months = [date.strftime("%B") for date in dates if date.strftime("%B") not in distinct_months]

    distinct_months = list(set(months))

    

    return distinct_months    
import pandas as pd



df_med_camps = pd.read_csv("../input/healthcare-analytics/Train/Health_Camp_Detail.csv",sep=',',delimiter=',')

#Format dates with pandas

df_med_camps
df_med_camps["Camp_Start_Date"] = pd.to_datetime(df_med_camps["Camp_Start_Date"], format="%d-%b-%y")

df_med_camps["Camp_End_Date"] = pd.to_datetime(df_med_camps["Camp_End_Date"], format="%d-%b-%y")
#map the months with a dictionary and map

s = {6:"Summer", 7:"Summer", 8:"Summer", 9:"Autumn", 10: "Autumn",11:"Autumn",12:"Winter",1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Spring"} 

df_med_camps["label"] = df_med_camps.filter(like="Date").apply(lambda d: d.dt.month.map(s)).agg(",".join, axis=1)

df_med_camps
#remove duplicates in label (seasons)

new_season = list(df_med_camps.label.str.split(","))

new_season
new_seasons2 = [str((set(season))).replace('{','').replace("'",'').replace('}','') for season in new_season]

type(new_seasons2)

df_med_camps['label'] = new_seasons2
df_med_camps.to_csv('Health_Camp_details_season.csv', encoding='utf-8',index=False)