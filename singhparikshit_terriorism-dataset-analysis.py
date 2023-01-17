#importing important libraries

import numpy as np

import csv
file_obj=open('../input/terrorismData.csv', encoding='utf8')

file_data=csv.DictReader(file_obj, skipinitialspace=True)

days=[]

for row in file_data:

    days.append(row['Day'])

np_days=np.array(days, dtype=float)

s=np_days[(np_days>=10) & (np_days<=20)]

print(s.shape[0])
with open('../input/terrorismData.csv', encoding='utf8') as file_obj:

    file_data=csv.DictReader(file_obj, skipinitialspace=True)

    

    day=[]

    year=[]

    month=[]

    count=0

    for row in file_data:

        day.append(row['Day'])

        month.append(row['Month'])

        year.append(row['Year'])

    np_day=np.array(day, dtype='int')

    np_year=np.array(year, dtype='int')

    np_month=np.array(month, dtype='int')

    

    np_day=np_day[np_month==1]

    np_year=np_year[np_month==1]

    np_day=np_day[np_year==2010]

    for i in np_day:

        if i>=1 and i<=31:

            count+=1

    print(count)
with open("../input/terrorismData.csv") as file:

    file_obj = csv.DictReader(file,skipinitialspace = False)

    killed = []

    wounded = []

    month = []

    year = []

    city = []

    group = []

    

    for row in file_obj:

        if "1999" in row["Year"]:

            if "5" in row["Month"] or  "6" in row["Month"] or "7" in row["Month"]:

                if "Unknown" not in row["City"]:

                    if "Unknown" not in row["Group"]:

                        if "Jammu and Kashmir" in row["State"]:

                            killed.append(row["Killed"])

                            wounded.append(row["Wounded"])

                            city.append(row["City"])

                            group.append(row["Group"])

        

    killed = np.array(killed)

    wounded = np.array(wounded)

    city = np.array(city)

    group = np.array(group)

    

    killed_bool = killed == ""

    wounded_bool = wounded == ""

    

    killed[killed_bool] = "0.0"

    wounded[wounded_bool] = "0.0"

    

    killed = np.array(killed, dtype = float)

    wounded = np.array(wounded, dtype = float)

    

    casualty = (killed + wounded)

    max_casualty = (int)(casualty.max())

    max_casualty_arg = casualty.argmax()

    print(max_casualty,city[max_casualty_arg],group[max_casualty_arg])

    
with open('../input/terrorismData.csv', encoding='utf8') as file_obj:

    file_data=csv.DictReader(file_obj, skipinitialspace=True)

    killed=[]

    wounded=[]

    state=[]

    count=0

    for row in file_data:

        killed.append(row['Killed'])

        wounded.append(row['Wounded'])

        state.append(row['State'])

    np_killed=np.array(killed)

    np_killed[np_killed=='']='0'

    np_killed=np.array(np_killed, dtype='float')

    np_wounded=np.array(wounded)

    np_wounded[np_wounded=='']='0'

    np_wounded=np.array(np_wounded, dtype='float')

    np_casuality=np.array(np_killed+np_wounded, dtype='int')

    for i in range(len(state)):

        if state[i]=='Chhattisgarh' or state[i]=='Odisha' or state[i]=='Jharkhand' or state[i]=='Andhra Pradesh':

            count+=np_casuality[i]

    print(count)
with open('../input/terrorismData.csv', encoding='utf8') as file_obj:

    file_data=csv.DictReader(file_obj, skipinitialspace=True)

    killed=[]

    wounded=[]

    city=[]

    for row in file_data:

        if 'India' in row['Country'] and 'Unknown' not in row['City']:

            city.append(row['City'])

            wounded.append(row['Wounded'])

            killed.append(row['Killed'])

    np_wounded=np.array(wounded)

    np_killed=np.array(killed)

    np_city=np.array(city)

    

    np_killed[np_killed=='']='0.0'

    np_wounded[np_wounded=='']='0.0'

    np_killed=np.array(np_killed, dtype='float')

    np_wounded=np.array(np_wounded, dtype='float')

    np_casuality=np.array(np_wounded+np_killed, dtype='int')

    

    

    citydic={}

    for i in range(len(np_city)):

        if np_city[i] in citydic:

            citydic[np_city[i]]+=np_casuality[i]

        else:

            citydic[np_city[i]]=np_casuality[i]

    

    count=0

    city=''

    for i in citydic:

        if citydic[i]>count:

            count=citydic[i]

            city=i

    print(city, count)

    del citydic[city]

    

    count=0

    city=''

    for i in citydic:

        if citydic[i]>count:

            count=citydic[i]

            city=i

    print(city, count)

    del citydic[city]

    

    count=0

    city=''

    for i in citydic:

        if citydic[i]>count:

            count=citydic[i]

            city=i

    print(city, count)

    del citydic[city]

    

    count=0

    city=''

    for i in citydic:

        if citydic[i]>count:

            count=citydic[i]

            city=i

    print(city, count)

    del citydic[city]

    

    count=0

    city=''

    for i in citydic:

        if citydic[i]>count:

            count=citydic[i]

            city=i

    print(city, count)

    del citydic[city]

    
with open('../input/terrorismData.csv', encoding='utf8') as file_obj:

    file_data=csv.DictReader(file_obj, skipinitialspace=True)

    

    day=[]

    for row in file_data:

        day.append(row['Day'])

    np_day=np.array(day, dtype='int')

    day, count=np.unique(np_day, return_counts=True)

    print(day[np.argmax(count)], count[np.argmax(count)])
#importing important libraries.

import pandas as pd
td=pd.read_csv('../input/terrorismData.csv', encoding='utf8')

df=td.copy()

df=df[df.State=='Jammu and Kashmir']

df=df[df.City==df.City.describe().top]

count=df.shape[0]

df=df[df.Group!='Unknown']

city=df.City.describe().top

group=df.Group.describe().top

print('CITY    ', 'COUNT  ', 'GROUP')

print(city, count, group)

df=td.copy()

df=df[df.Country==df.Country.describe().top]

count=df.shape[0]

country=df.Country.describe().top

y={}

for i in df.Year:

    if i in y.keys():

        y[i]+=1

    else:

        y[i]=1

cnt=0

year=0

for i in y.keys():

    if cnt<y[i]:

        cnt=y[i]

        year=i

print(country, count, year)

df=td.copy()

df=df[df.Killed==df.Killed.max()]

killed=df.Killed.iloc[0]

country=df.Country.iloc[0]

group=df.Group.iloc[0]

print(int(killed), country, group)

df=td.copy()

a=df[df.Day>=26]

b=a[a.Year==2014]

c=b[b.Country=='India']

ans1=c[c.Month==5]

del a

del b

del c

a=df[df.Year==2014]

b=a[a.Country=='India']

ans2=b[b.Month>5]

del a

del b

a=df[df.Country=='India']

ans3=a[a.Year>2014]

count=ans1.shape[0]+ans2.shape[0]+ans3.shape[0]

print(count, end=' ')

ans1=ans1[ans1.Group!='Unknown']

ans2=ans2[ans2.Group!='Unknown']

ans3=ans3[ans3.Group!='Unknown']

print(ans3.Group.describe().top)
df_terrorism=td.copy()



year=len(set(df_terrorism['Year']))





df_terrorism=df_terrorism[df_terrorism['Country']=='India']



df_terrorism['Casualty']=df_terrorism['Killed']+df_terrorism['Wounded']



Jammu_state=df_terrorism[df_terrorism['State']=='Jammu and Kashmir']



red_state=df_terrorism[(df_terrorism['State']=='Jharkhand')|(df_terrorism['State']=='Odisha')

                       |(df_terrorism['State']=='Andhra Pradesh')|(df_terrorism['State']=='Chhattisgarh')]



red_casualty=int(np.sum(red_state['Casualty']))



Jammu_casualty=int(np.sum(Jammu_state['Casualty']))



print(red_casualty//year,Jammu_casualty//year)