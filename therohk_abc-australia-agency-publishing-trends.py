import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



#from subprocess import check_output

#print(check_output(["ls", "../input/million-headlines"]).decode("utf8"))



df = pd.read_csv("../input/million-headlines/abcnews-date-text.csv", dtype={'publish_date': object})



df['publish_month'] = df.publish_date.str[:6]

df['publish_year'] = df.publish_date.str[:4]

df['publish_month_only'] = df.publish_date.str[4:6]

df['publish_day_only'] = df.publish_date.str[6:8]



years = df['publish_year'].unique().tolist()

print(years)



df['dt_date'] = pd.to_datetime(df['publish_date'], format='%Y%m%d')

df['dt_month'] = pd.to_datetime(df['publish_month'], format='%Y%m')



grp_date = df.groupby(['dt_date'])['headline_text'].count()

grp_month = df.groupby(['dt_month'])['headline_text'].count()



ts = pd.Series(grp_date)

ts.plot(kind='line', figsize=(20,10),title='articles per day')

#plt.show()



ts = pd.Series(grp_month)

ts.plot(kind='line', figsize=(20,10),title='articles per month')

plt.show()



for year in years:

    yr_slice = df.loc[df.publish_year==year]

    grp_month = yr_slice.groupby(['publish_month_only'])['headline_text'].count()

    month_ts = pd.Series(grp_month)

    month_ts.plot(kind='line', figsize=(20,10), style='o-', legend=True, label=year)

    

plt.show()
