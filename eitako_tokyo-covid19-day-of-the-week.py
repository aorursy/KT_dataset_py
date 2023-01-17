import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





dfcount = pd.read_csv('../input/tokyocount/data.csv',index_col=0, parse_dates=True,encoding='cp932', usecols=[0,1,3])

dfcount
dfcount.columns=['studycount1','studycount2']

dfcount=dfcount[(dfcount.index >= pd.datetime(2020,6,15)) & (dfcount.index <= pd.datetime(2020,8,12))]

dfcount['studycount']=dfcount['studycount1']+dfcount['studycount2']
dfcount
df1 = pd.DataFrame({'week':dfcount.index.day_name(),'studycount':dfcount.studycount})

df2=df1.sort_index(ascending=True)

df2
weeklist=["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]

df3 = df2.groupby("week")

df_sunday=df3.get_group(weeklist[0])["studycount"]

df_monday=df3.get_group(weeklist[1])["studycount"]

df_tuesday=df3.get_group(weeklist[2])["studycount"]

df_wednesday=df3.get_group(weeklist[3])["studycount"]

df_thursday=df3.get_group(weeklist[4])["studycount"]

df_friday=df3.get_group(weeklist[5])["studycount"]

df_saturday=df3.get_group(weeklist[6])["studycount"]



meanlist=np.array([df_sunday.mean(),df_monday.mean(),df_tuesday.mean(),df_wednesday.mean(),df_thursday.mean(),df_friday.mean(),df_saturday.mean()])



dfweek = pd.DataFrame({"week":weeklist,"studycountmean":meanlist})



dfweek



dfweek.plot(x='week', y=['studycountmean'],title='tokyo [covid-19] study count per day of the week',color=['green'], kind='bar',figsize=(12,6))

meanlistoffset=meanlist.mean()-meanlist

dfweekoffset = pd.DataFrame({"week":weeklist,"studycountmeanoffset":meanlistoffset.round().astype(int)})



dfweekoffset.plot(x='week', y=['studycountmeanoffset'],title='tokyo [covid-19] study count offset per day of the week',color=['green'], kind='bar',figsize=(12,6))





dfweekoffset