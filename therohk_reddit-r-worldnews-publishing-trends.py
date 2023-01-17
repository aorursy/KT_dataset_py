import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



df  = pd.read_csv("../input/reddit_worldnews_start_to_2016-11-22.csv", dtype={'date_created': object})

#print (df.info)



df['date_created'].replace(regex=True,inplace=True,to_replace="-",value="")

df['publish_month'] = df.date_created.str[:6]

df['publish_year'] = df.date_created.str[:4]

df['publish_month_only'] = df.date_created.str[4:6]

df['publish_day_only'] = df.date_created.str[6:8]



years=df['publish_year'].unique().tolist()

print (years)



df['dt_date'] = pd.to_datetime(df['date_created'], format='%Y%m%d')

df['dt_month'] = pd.to_datetime(df['publish_month'], format='%Y%m')



grp_date = df.groupby(['dt_date'])['title'].count()

grp_month = df.groupby(['dt_month'])['title'].count()



ts = pd.Series(grp_date)

ts.plot(kind='line', figsize=(20,10),title='Articles per day')

#plt.show()



ts = pd.Series(grp_month)

ts.plot(kind='line', figsize=(20,10),title='Articles per month')

plt.show()
#Year slice plotting

for year in years:

    yr_slice = df.loc[df.publish_year==year]

    grp_month = yr_slice.groupby(['publish_month_only'])['title'].count()

    month_ts = pd.Series(grp_month)

    month_ts.plot(kind='line', figsize=(20,10), style='o-', legend=True, label=year)

    

plt.show()