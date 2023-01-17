import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings("ignore")



import os
df_patient_info=pd.read_csv("/kaggle/input/coronavirusdataset/PatientInfo.csv")

df_region=pd.read_csv("/kaggle/input/coronavirusdataset/Region.csv")

df_search_trend=pd.read_csv("/kaggle/input/coronavirusdataset/SearchTrend.csv")

df_time_gender=pd.read_csv("/kaggle/input/coronavirusdataset/TimeGender.csv")

df_weather=pd.read_csv("/kaggle/input/coronavirusdataset/Weather.csv")

df_case=pd.read_csv("/kaggle/input/coronavirusdataset/Case.csv")

df_time=pd.read_csv("/kaggle/input/coronavirusdataset/Time.csv")

df_time_province=pd.read_csv("/kaggle/input/coronavirusdataset/TimeProvince.csv")

df_time_age=pd.read_csv("/kaggle/input/coronavirusdataset/TimeAge.csv")

df_patient_route=pd.read_csv("/kaggle/input/coronavirusdataset/PatientRoute.csv")

df_seoul_floating=pd.read_csv("/kaggle/input/coronavirusdataset/SeoulFloating.csv")

df_policy=pd.read_csv("/kaggle/input/coronavirusdataset/Policy.csv")
df_patient_info.head()
df_patient_info.drop(["patient_id","global_num","infected_by"],axis=1, inplace=True)
df_patient_info.head()
df_patient_info.tail()
df_patient_info.info()
df_patient_info.isna().sum()
df_patient_info.drop(["birth_year","disease","infection_case","infection_order","contact_number","symptom_onset_date","released_date","released_date","deceased_date"],axis=1,inplace=True)
df_patient_info.head()
df_patient_info.dtypes
df_patient_info.head()
df_patient_info['age'] = df_patient_info['age'].str.replace(r'\D', '')
df_patient_info.head()
df_patient_info.dtypes
df_patient_info.age.isna().sum()
def impute_median(series):

    return series.fillna(series.median())
df_patient_info.age =df_patient_info.age.transform(impute_median)
df_patient_info.age.unique()
df_patient_info["age"]=df_patient_info["age"].apply(lambda x: float(x))
df_patient_info.dtypes
df_patient_info.isna().sum()
df_patient_info["sex"].fillna(str(df_patient_info["sex"].mode().values[0]),inplace=True)

df_patient_info["country"].fillna(str(df_patient_info["country"].mode().values[0]),inplace=True)

df_patient_info["city"].fillna(str(df_patient_info["city"].mode().values[0]),inplace=True)

df_patient_info["confirmed_date"].fillna(str(df_patient_info["confirmed_date"].mode().values[0]),inplace=True)
df_patient_info.isna().sum()
df_patient_info.head()
#m=1,f=0



df_patient_info["sex"]=[1 if i.strip()== "male" else 0 for i in df_patient_info.sex]
df_patient_info.head()
df_patient_info.age.describe()
len(df_patient_info[df_patient_info.age==0.0])
df_patient_info['age']=df_patient_info['age']

bins=[0,14,28,42,56,70,100]

labels=["Child","Young","Young Adult","Early Adult","Adult","Senior"]

df_patient_info['age_group']=pd.cut(df_patient_info['age'],bins,labels=labels)

df_patient_info.head()
df_patient_info.age_group.isna().sum()
df_patient_info.age_group.unique()
df_patient_info["age_group"].fillna(str(df_patient_info["age_group"].mode().values[0]),inplace=True)
df_patient_info.country.dtypes
df_patient_info.dtypes

df_patient_info.country.unique()
# Create a list to store the data

continents = []



# For each row in the column,

for row in df_patient_info['country']:

    if row =="Korea":

        continents.append('Asia')

    elif row=="China":

        continents.append('Asia')

    elif row=="United States":

        continents.append('North America')

    elif row=="France":

        continents.append('Europe')

    elif row=="Thailand":

        continents.append('Asia')

    elif row=="Canada":

        continents.append('North America')

    elif row=="Switzerland":

        continents.append('Europe')

    elif row=="Indonesia":

        continents.append('Asia')

    elif row=="Mongolia":

        continents.append("Asia")

    elif row=="Spain":

        continents.append("Europe")

    elif row=="Foreign":

        continents.append("Asia")

    else:

        continents.append('Failed')

        

# Create a column from the list

df_patient_info['continents'] = continents
df_patient_info.head()
df_patient_info.continents.unique()
df_patient_info['months']=0

for i in df_patient_info:

    df_patient_info['months']=df_patient_info['confirmed_date'].str.split('-', 0).str[1].str.strip() 

df_patient_info.head()
df_patient_info['days']=0

for i in df_patient_info:

    df_patient_info['days']=df_patient_info['confirmed_date'].str.split('-',0).str[2].str.strip() 

df_patient_info.head()
df_patient_info.dtypes
df_patient_info["months"]=df_patient_info["months"].apply(lambda x: int(x))

df_patient_info["days"]=df_patient_info["days"].apply(lambda x: int(x))
df_patient_info.dtypes
df_patient_info.isna().sum()
fig=plt.figure(figsize=(10,5))

sns.barplot(x='age_group',y='sex',data=df_patient_info)

plt.legend()

plt.show()
x=df_patient_info.sex

y=df_patient_info.age

plt.plot(x, y, '-p', color='gray',

         markersize=15, linewidth=4,

         markerfacecolor='white',

         markeredgecolor='gray',

         markeredgewidth=3)

plt.ylim(0, 100);

#man

df_patient_info[df_patient_info.sex==1].age.describe()
#women

df_patient_info[df_patient_info.sex==0].age.describe()
print("Sum of Korea:",len(df_patient_info[df_patient_info.country=="Korea"]))

print("Sum of China:",len(df_patient_info[df_patient_info.country=="China"]))

print("Sum of United States", len(df_patient_info[df_patient_info.country=="United States"]))

print("Sum of France", len(df_patient_info[df_patient_info.country=="France"]))

print("Sum of Thailand", len(df_patient_info[df_patient_info.country=="Thailand"]))

print("Sum of Canada:",len(df_patient_info[df_patient_info.country=="Canada"]))

print("Sum of Switzerland:",len(df_patient_info[df_patient_info.country=="Switzerland"]))

print("Sum of Indonesia", len(df_patient_info[df_patient_info.country=="Indonesia"]))

print("Sum of Foreign", len(df_patient_info[df_patient_info.country=="Foreign"]))

print("Sum of Mongolia", len(df_patient_info[df_patient_info.country=="Mongolia"]))

print("Sum of Spain", len(df_patient_info[df_patient_info.country=="Spain"]))
df_region.head()
df_region.tail()
df_region.drop(["code"],axis=1,inplace=True)
df_region.isna().sum()
df_region.dtypes
df_search_trend.head()
df_search_trend.tail()
df_search_trend['months']=0

for i in df_search_trend:

    df_search_trend['months']=df_search_trend['date'].str.split('-', 0).str[1].str.strip() 

df_search_trend.head()
df_search_trend['days']=0

for i in df_search_trend:

    df_search_trend['days']=df_search_trend['date'].str.split('-',0).str[2].str.strip() 

df_search_trend.head()
df_search_trend.dtypes
df_search_trend["months"]=df_search_trend["months"].apply(lambda x: int(x))

df_search_trend["days"]=df_search_trend["days"].apply(lambda x: int(x))
df_search_trend.isna().sum()
df_time_gender.head()
df_time_gender.tail()
df_time_gender['months']=0

for i in df_time_gender:

    df_time_gender['months']=df_time_gender['date'].str.split('-', 0).str[1].str.strip() 

df_time_gender.head()
df_time_gender['days']=0

for i in df_time_gender:

    df_time_gender['days']=df_time_gender['date'].str.split('-',0).str[2].str.strip() 

df_time_gender.head()
df_time_gender.dtypes
df_time_gender["months"]=df_time_gender["months"].apply(lambda x: int(x))

df_time_gender["days"]=df_time_gender["days"].apply(lambda x: int(x))
df_time_gender.dtypes
df_time_gender.isna().sum()
df_weather.head()
df_weather.drop(["code"],axis=1,inplace=True)
df_weather.head()
df_weather.isna().sum()
df_weather=df_weather.dropna()
df_weather.isna().sum()
df_weather['months']=0

for i in df_weather:

    df_weather['months']=df_weather['date'].str.split('-', 0).str[1].str.strip() 

df_weather.head()
df_weather['days']=0

for i in df_weather:

    df_weather['days']=df_weather['date'].str.split('-',0).str[2].str.strip() 

df_weather.head()
df_weather.dtypes
df_weather["months"]=df_weather["months"].apply(lambda x: int(x))

df_weather["days"]=df_weather["days"].apply(lambda x: int(x))
df_weather.dtypes
df_case.head()
df_case.drop(["case_id"],axis=1,inplace=True)
df_case.tail()
df_case.isna().sum()
df_time.head()
df_time.tail()
df_time['months']=0

for i in df_time:

    df_time['months']=df_time['date'].str.split('-', 0).str[1].str.strip() 

df_time.head()
df_time['days']=0

for i in df_time:

    df_time['days']=df_time['date'].str.split('-',0).str[2].str.strip() 

df_time.head()
df_time.dtypes
df_time["months"]=df_time["months"].apply(lambda x: int(x))

df_time["days"]=df_time["days"].apply(lambda x: int(x))
df_time.dtypes
df_time.isna().sum()
df_time_province.head()
df_time_province.tail()
df_time_province['months']=0

for i in df_time_province:

    df_time_province['months']=df_time_province['date'].str.split('-', 0).str[1].str.strip() 

df_time_province.head()
df_time_province['days']=0

for i in df_time_province:

    df_time_province['days']=df_time_province['date'].str.split('-',0).str[2].str.strip() 

df_time_province.head()
df_time_province.dtypes
df_time_province["months"]=df_time_province["months"].apply(lambda x: int(x))

df_time_province["days"]=df_time_province["days"].apply(lambda x: int(x))
df_time_province.dtypes
df_time_province.isna().sum()
df_time_age.head()
df_time_age.tail()
df_time_age.isna().sum()
df_time_age['months']=0

for i in df_time_age:

    df_time_age['months']=df_time_age['date'].str.split('-', 0).str[1].str.strip() 

df_time_age.head()
df_time_age['days']=0

for i in df_time_age:

    df_time_age['days']=df_time_age['date'].str.split('-',0).str[2].str.strip() 

df_time_age.head()
df_time_age.dtypes
df_time_age["months"]=df_time_age["months"].apply(lambda x: int(x))

df_time_age["days"]=df_time_age["days"].apply(lambda x: int(x))
df_time_age.dtypes
df_time_age['age'] = df_time_age['age'].str.replace(r'\D', '')
df_time_age.head()
df_time_age.dtypes
df_time_age["age"]=df_time_age["age"].apply(lambda x: int(x))
df_time_age.dtypes
df_patient_route.head()
df_patient_route.tail()
df_patient_route.drop(["patient_id","global_num"],axis=1, inplace=True)
df_patient_route.head()
df_patient_route['months']=0

for i in df_patient_route:

    df_patient_route['months']=df_patient_route['date'].str.split('-', 0).str[1].str.strip() 

df_patient_route.head()
df_patient_route['days']=0

for i in df_patient_route:

    df_patient_route['days']=df_patient_route['date'].str.split('-',0).str[2].str.strip() 

df_patient_route.head()
df_patient_route.dtypes
df_patient_route["months"]=df_patient_route["months"].apply(lambda x: int(x))

df_patient_route["days"]=df_patient_route["days"].apply(lambda x: int(x))
df_patient_route.dtypes
df_patient_route.isna().sum()
df_seoul_floating.head()
df_seoul_floating.tail()
df_seoul_floating['months']=0

for i in df_seoul_floating:

    df_seoul_floating['months']=df_seoul_floating['date'].str.split('-', 0).str[1].str.strip() 

df_seoul_floating.head()
df_seoul_floating['days']=0

for i in df_seoul_floating:

    df_seoul_floating['days']=df_seoul_floating['date'].str.split('-',0).str[2].str.strip() 

df_seoul_floating.head()
df_seoul_floating.dtypes
df_seoul_floating["months"]=df_seoul_floating["months"].apply(lambda x: int(x))

df_seoul_floating["days"]=df_seoul_floating["days"].apply(lambda x: int(x))
df_seoul_floating.dtypes
df_policy.head()
df_policy.drop(["policy_id"],axis=1,inplace=True)
df_policy.tail()
df_policy.isna().sum()
df_policy=df_policy.dropna()
df_policy.isna().sum()
df_policy['start_months']=0

for i in df_policy:

    df_policy['start_months']=df_policy['start_date'].str.split('-', 0).str[1].str.strip() 

df_policy.head()
df_policy['start_days']=0

for i in df_policy:

    df_policy['start_days']=df_policy['start_date'].str.split('-',0).str[2].str.strip() 

df_policy.head()
df_policy['end_months']=0

for i in df_policy:

    df_policy['end_months']=df_policy['end_date'].str.split('-', 0).str[1].str.strip() 

df_policy.head()
df_policy['end_days']=0

for i in df_policy:

    df_policy['end_days']=df_policy['end_date'].str.split('-',0).str[2].str.strip() 

df_policy.head()
df_policy.dtypes
df_policy["start_months"]=df_policy["start_months"].apply(lambda x: int(x))

df_policy["start_days"]=df_policy["start_days"].apply(lambda x: int(x))

df_policy["end_months"]=df_policy["end_months"].apply(lambda x: int(x))

df_policy["end_days"]=df_policy["end_days"].apply(lambda x: int(x))
df_policy.dtypes
df_policy.gov_policy.unique()
df_policy.detail.unique()