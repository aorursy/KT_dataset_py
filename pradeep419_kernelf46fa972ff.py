#Im just started learning ML. So its a basic analysis on the given data. Please let me know if you have suggestions. Thank you

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/gtd/globalterrorismdb_0718dist.csv',encoding='ISO-8859-1')
df.head()
#Lets check highest terrorist attacks in Country, Region  and City

country_highest = df['country_txt'].value_counts()

country_highest.index[0]

region_highest = df['region_txt'].value_counts().index[0]

region_highest

city_highest = df['city'].value_counts().index[1]

city_highest



#So Country-Iraq, Region -ME&SA , City-Baghdad
#Lets check the stas of attack Type, Weapon Type and Groups Active(top 10)

attack_type = df['attacktype1_txt'].value_counts().index

attack_type[0:10]

weapon_type = df['weaptype1_txt'].value_counts().index

weapon_type[0:10]

group_name = df['gname'].value_counts().index

group_name[0:10]

#Attack Type - 'Bombing/Explosion', 'Armed Assault', 'Assassination','Hostage Taking (Kidnapping)', 'Facility/Infrastructure Attack',

               #'Unknown', 'Unarmed Assault', 'Hostage Taking (Barricade Incident)','Hijacking'

#Weapon Type - 'Explosives', 'Firearms', 'Unknown', 'Incendiary', 'Melee', 'Chemical','Sabotage Equipment','Vehicle (not to include vehicle-borne explosives, i.e., car or truck bombs)',

               #'Other', 'Biological'

#Groups Active - 'Unknown', 'Taliban', 'Islamic State of Iraq and the Levant (ISIL)','Shining Path (SL)', 'Farabundo Marti National Liberation Front (FMLN)',

                #'Al-Shabaab', 'New People's Army (NPA)', 'Irish Republican Army (IRA)','Revolutionary Armed Forces of Colombia (FARC)', 'Boko Haram'
#Lets check the attacks by year

plt.subplots(figsize=(15,6))

plot_year = sns.countplot(x='iyear',data=df)

plt.xticks(rotation = 90)

plt.show()

#There is a sharp increase in attacks from 2004 to 2014 and then decreased a bit
#Lets see how various groups in active phase through different years

for i in range(1970,2018):

    if i==1993:

        continue

    year_data=[]

    year_data = pd.DataFrame(df[df.iloc[:,1]==i])

    group = year_data['gname'].value_counts().index

    print(i,group[1])

    

#As first highest mostly unknown, I am going with second highest

#From 1973 to 77 Irish Republican Army very active

#1980 to 82 - FMLN

#1983 to 90 - Sl

#1992 to 95 - PKK

#1997 to 99 - FARC

#2004 to 08 - Taliban

#2009 to 11 - CPI M

#2014 to 17 - ISIL
#Lets see if attacks are happening on prominent days

months_highest = df['imonth'].value_counts().index

months_highest[0:12]

months_highest_array = np.array(months_highest)

#So Months with highest attacks in order - May,July,August,October,June,March,April,January,November,September,Febrauary,December

len1 = len(months_highest_array)

for i in range(len1):

    month = []

    month = pd.DataFrame(df[df.iloc[:,2]==months_highest[i]])

    days_in_month = month['iday'].value_counts().index

    days_in_month_array = np.array(days_in_month)

    len2 = len(days_in_month_array)

    

    for j in range(0,4):

        day = []

        day = pd.DataFrame(month[month.iloc[:,3]==days_in_month[j]])

        country_attacked = day['country_txt'].value_counts().index

        groups_active = day['gname'].value_counts().index

        print(months_highest[i],days_in_month[j],groups_active[0]+'\t'+country_attacked[0])

        print(months_highest[i],days_in_month[j],groups_active[1]+'\t'+country_attacked[1])

        print(months_highest[i],days_in_month[j],groups_active[2]+'\t'+country_attacked[2])

        print(months_highest[i],days_in_month[j],groups_active[3]+'\t'+country_attacked[3])

        print('========================================================================')

    print(' ')

    

#05/16 - 

#05/29 -

#07/04 - American Independence

#07/27 - 

#08/25 - Uruguay Independence

#08/15 - India's Independence day

#10/09 - Uganda Independence Day

#10/29 - 

#06/05 - 

#06/14 - 

#03/15 - 

#03/22 - 

#04/15 - 

#04/09 - 

#01/01 - New Year

#01/26 - Republic day in India

#11/13 - 

#11/18 - 

#09/18 -

#09/09 - 

#02/28 - 

#02/21 -

#12/01 - 

#12/15 - 

#Mostly Taliban,ISIL,Shining Path listed almost everywhere

#Groups and Countries list not accurate. Just tried to display top active groups and top countries attacked on those days

#Just tried to establish a relation
#Regionwise attacks

plt.subplots(figsize=(15,6))

plot_year = sns.countplot(x='region_txt',palette='inferno',order = df['region_txt'].value_counts().index,data=df)

plt.xticks(rotation = 90)

plt.show()

#As expected ME&NA and SA topped the list
#Type of attacks

plt.subplots(figsize=(15,6))

plot_year = sns.countplot(x='attacktype1_txt',palette='inferno',order = df['attacktype1_txt'].value_counts().index,data=df)

plt.xticks(rotation = 90)

plt.show()

#Bombing Explosion and Armed Assault tooped the list
#Weapon Type

plt.subplots(figsize=(15,6))

plot_year = sns.countplot(x='weaptype1_txt',palette='inferno',order = df['weaptype1_txt'].value_counts().index, data=df)

plt.xticks(rotation = 90)

plt.show()

#Explosives and Firearms tooped the list
#Lets check countries in which Chemical and Biological Weapons Used

chemical_Weapons = pd.DataFrame(df[df['weaptype1_txt']=='Chemical'])

chemical_Weapons.head()

countries_with_chemical = chemical_Weapons['country_txt'].value_counts()

print(countries_with_chemical[0:10])

groups_used_chemical = chemical_Weapons['gname'].value_counts()

print(groups_used_chemical[0:10])



bio_Weapons = pd.DataFrame(df[df['weaptype1_txt']=='Biological'])

countries_with_bio = bio_Weapons['country_txt'].value_counts()

print(countries_with_bio[0:10])

groups_used_bio = bio_Weapons['gname'].value_counts()

print(groups_used_bio[0:10])



#Suprisingly US stood top in Countries attacked with bio weapons. Also stood third in case of Chem Weapons
#Lets see the target types

target_type = df['targtype1_txt'].value_counts().index

plt.subplots(figsize=(15,6))

plot_targettype = sns.countplot(x='targtype1_txt',palette='inferno',order = df['targtype1_txt'].value_counts().index, data=df)

plt.xticks(rotation = 90)

plt.show()

len1 = len(target_type)

print(len1)

#Private Citizens,Military and Police are the first three targets

#Lets check which group targets particular section

for i in range(len1):

    target = []

    target = pd.DataFrame(df[df['targtype1_txt']==target_type[i]])

    group_target = target['gname'].value_counts().index

    print(target_type[i],group_target[0:4])

    print(' ')

    

#So from the output we can observe that, Terrorist groups like Taliban, ISIL mostly targetted Private Citizens,Military and Police

#Groups like SL mainly targetted govt properties, Business, Transportation

#Groups like CPI M targetted Police, Roads, Transportation. Also they are violent political party

#Groups like Basque Fatherland and Freedom (ETA) and Armenian Secret Army for the Liberation of Armenia targetted Tourists.

#Armenian Secret Army for the Liberation of Armenia also listed in Airport attacks. So we can presume that they mainly involve in Hijacking and Hostages

#Groups like ISIL and Boko Haram targetted Reliogious figures/Institutions. Its quite known ISIL targetted SHIA institutions

#Groups like TTP and Taliban targetted Educational Institutions. We all well aware of situtaion in Afghan and case of Malala

#There are cases of Abortion related, targetted by Anti-Abortion extremists,Army of God and Christian Liberation Army
df['nkill'].fillna(0)

df['nwound'].fillna(0)

df['affected'] = df['nkill']+df['nwound']

top_groups = df['gname'].value_counts()[1:15].index

print(top_groups)

number_affected = []



for i in range(len(top_groups)):

    

    groupby_gname = []

    

    groupby_gname = pd.DataFrame(df[df['gname']==top_groups[i]])

    number_killed = groupby_gname['nkill'].sum()

    number_wounded = groupby_gname['nwound'].sum()

    print(top_groups[i]+'-------'+str(number_killed)+'-------'+str(number_wounded)+'---------'+str(number_killed/number_wounded))

     

#In the attacks of SL and Boko Haram, number of people killed more than number of people wounded. It is quite opposite in case of ETA
