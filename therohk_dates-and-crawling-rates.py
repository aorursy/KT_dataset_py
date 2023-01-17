# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



df  = pd.read_csv("../input/news-week-aug24.csv", dtype={'publish_time': object})



df['publish_hour'] = df.publish_time.str[:10]

df['publish_date'] = df.publish_time.str[:8]

df['publish_hour_only'] = df.publish_time.str[8:10]

df['publish_time_only'] = df.publish_time.str[8:12]

days=df['publish_date'].unique().tolist()



print (days)



#print (df.info)



df['dt_time'] = pd.to_datetime(df['publish_time'], format='%Y%m%d%H%M')

df['dt_hour'] = pd.to_datetime(df['publish_hour'], format='%Y%m%d%H')

df['dt_date'] = pd.to_datetime(df['publish_date'], format='%Y%m%d')





grp_time = df.groupby(['dt_time'])['headline_text'].count()

grp_hour = df.groupby(['dt_hour'])['headline_text'].count()





ts = pd.Series(grp_time)

ts.plot(kind='line', figsize=(20,10),title='Articles per Minute')



ts1 = pd.Series(grp_hour)

ts1.plot(kind='line', figsize=(20,10),title='Articles per Hour')



plt.show()
grp_date = df.groupby(['dt_date'])['headline_text'].count()

ts2 = pd.Series(grp_date)

ts2.plot(kind='bar', figsize=(10,5),title='Articles per Day')



plt.show()
for day in days:

    day_slice = df.loc[df.publish_date==day]

    grp_hour = day_slice.groupby(['publish_hour_only'])['headline_text'].count()

    hour_ts = pd.Series(grp_hour)

    hour_ts.plot(kind='line', figsize=(20,10), style='o-', legend=True, label=day, title='Hourly Volumes per Day')

    

plt.show()