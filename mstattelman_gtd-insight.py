"""

A few insights uncovered:

1) The month and day that most attacks occur.

2) The types of weapons most likley used.

3) The property value destroyed by aggregate(occurrence).

4) The days of most occurrences.

5) The month and weapons types of most occerrences in the US.

6) The month and weapons types of most occerrences in California by city.

7) The month and weapons types of most occerrences in California.



**Just a few insights generated from the global terrorism dataset.

Beginning with data analysis and iPython Notebook**

"""

import pandas as pd



df = pd.read_csv('../input/globalterrorismdb_0718dist.csv', error_bad_lines=False, index_col=False,encoding='ISO-8859-1')

#df
import os

print(os.listdir("../input"))
#List attacks by weapons types count

df.weapsubtype1_txt.value_counts()
#List attacks by property value count

df.propextent_txt.value_counts()
#List attacks by day/date count

df.iday.value_counts()

#Get top 25 Month, Day of most occurrences

mon_day = df.groupby(['imonth','iday'], sort=False).size().reset_index().values.tolist()

mon_day.sort(key=lambda x: x[2], reverse=True)

df0 = pd.DataFrame(mon_day[:25], columns=['Month','Day','Count'])

df0
#List Attacks by Month, Weapons type and count

mon_weap = df.groupby(['imonth','weapsubtype1_txt'], sort=False).size().reset_index().values.tolist()

df1 = pd.DataFrame(mon_weap, columns=['Month','Weapons Type','Count'])

df1
#Sort on Month, weapon type, count (most)

mon_weap.sort(key=lambda x: x[2], reverse=True)

df2 = pd.DataFrame(mon_weap, columns=['Month','Weapons Type','Occurrences'])

df2
#In the US

df3 = pd.DataFrame(df.groupby(['country_txt','imonth','weapsubtype1_txt'], sort=True).size().reset_index().values.tolist())

df3.columns=['Country','Month','Weapons Type','Occurrences']

df3 = df3.loc[df3['Country'] == 'United States']

df3 = df3.sort_values('Occurrences',ascending=False).reset_index()

df3
#In the US

df4 = pd.DataFrame(df.groupby(['country_txt','provstate','city','imonth','weapsubtype1_txt'], sort=True).size().reset_index().values.tolist())

df4.columns=['Country','State','City','Month','Weapons Type','Occurrences']

df4 = df4.loc[df4['Country'] == 'United States']

#df4 = df4.loc[df4['State'] == 'California']

df4 = df4.sort_values('Occurrences',ascending=False).reset_index()

df4
#Month, Weapons Type and Occurrences taken from the State dataframe above

mon_weap1 = df4.groupby(['Month','Weapons Type'], sort=False).size().reset_index().values.tolist()

mon_weap1.sort(key=lambda x: x[2], reverse=True)

df5 = pd.DataFrame(mon_weap1)

df5.columns=['Month','Weapons Type','Occurrences']

df5