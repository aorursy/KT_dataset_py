# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

plt.rcParams["figure.figsize"]=10,6

plt.rcParams["axes.grid"]=True

plt.gray()
df = pd.read_csv("/kaggle/input/data-police-shootings/fatal-police-shootings-data.csv")
df.head()
#import pandas_profiling as pf
#from pandas_profiling import ProfileReport

#prof = ProfileReport(df)

#prof.to_file(output_file='Episode.html')
#prof
df.info()
df.columns
def missing_percentage(df):

    percentage = pd.Series(list(round((df.isnull().sum()/len(df)*100),2)),name="Percentage")

    col_name = pd.Series(df.columns,name="Col_name")

    missing_percentage = pd.concat([col_name,percentage],axis=1).sort_values(by="Percentage",ascending=False).reset_index(drop=True)

    return missing_percentage
ts = missing_percentage(df)

print(ts)
# sns.set(rc={'figure.figsize':(18,6)})

plt.figure(figsize=(18,6))

g= sns.barplot(x="Col_name",y="Percentage",data=ts)

for index, row in ts.iterrows():

    g.text(row.name,row.Percentage, round(row.Percentage), color='black', ha="center")

plt.xlabel("Columns_Name",weight="bold",fontsize=12)

plt.ylabel("Percentage",weight="bold",fontsize=12)

plt.title("Missing_values_along_different_columns",weight="bold",fontsize=14)

plt.show()
df.head()
df["month"] = pd.to_datetime(df["date"]).dt.month

df["year"] = pd.to_datetime(df["date"]).dt.year
df.head()
df['manner_of_death'].value_counts()
# sns.set(rc={'figure.figsize':(10,6)})

sns.countplot(x="gender",data=df,palette='winter')

plt.xlabel("Gender",weight="bold",fontsize=12)

plt.ylabel("Count",weight="bold",fontsize=12)

plt.title("Distribution of Gender",weight="bold",fontsize=14)

plt.show()
ds = df.loc[df["manner_of_death"]=="shot"]



# sns.set(rc={'figure.figsize':(10,6)})

g=sns.countplot(x="race",data=ds,palette='winter')

plt.xlabel("Race",weight="bold",fontsize=12)

plt.ylabel("Count",weight="bold",fontsize=12)

plt.title("Distribution of Shootout by Police",weight="bold",fontsize=14)



plt.show()
unarmed = df.loc[df["armed"]=="unarmed"]

# sns.set(rc={'figure.figsize':(10,6)})

sns.countplot(x="race",data=unarmed,palette='winter')

plt.xlabel("Race",weight="bold",fontsize=12)

plt.ylabel("Count",weight="bold",fontsize=12)

plt.title("Distribution of Shootout by Police unarmed people",weight="bold",fontsize=14)

plt.show()
shoot_by_states = df["state"].value_counts()[:10]

shoot_by_states = pd.DataFrame(shoot_by_states).reset_index()

shoot_by_states
states = shoot_by_states['index'].tolist()

data_counts  = shoot_by_states['state'].tolist()
fig, ax = plt.subplots(figsize=(10,10), subplot_kw=dict(aspect="equal"))



states = states



data = data_counts



def func(pct, allvals):

    absolute = int(pct/100.*np.sum(allvals))

    return "{:.1f}%\n({:d} )".format(pct, absolute)





wedges, texts, autotexts = ax.pie(data, autopct=lambda pct: func(pct, data),

                                  textprops=dict(color="w"))



ax.legend(wedges, states,

          title="States",

          loc="center left",

          bbox_to_anchor=(1, 0, 0.5, 1))



plt.setp(autotexts, size=8, weight="bold")



ax.set_title("Top 10 US states",weight="bold",fontsize=14)



plt.show()
df.columns
sns.distplot(df['age'])

plt.xlabel("Age",weight="bold",fontsize=12)

plt.ylabel("Frequency",weight="bold",fontsize=12)

plt.title("Age distribution",weight="bold",fontsize=14)

plt.show()
body_camera = df['body_camera'].value_counts()
sns.barplot(body_camera.index, body_camera.values, alpha=0.8)

plt.xlabel("Body Camnera",weight="bold",fontsize=12)

plt.ylabel("Frequency",weight="bold",fontsize=12)

plt.title("How many Police wearing body camera??",weight="bold",fontsize=14)

plt.show()
no_camera = df[df["body_camera"]==False]
no_camera_pol =no_camera['race'].value_counts()
g=sns.barplot(no_camera_pol.index, no_camera_pol.values, alpha=0.8)

plt.xlabel("Race",weight="bold",fontsize=12)

plt.ylabel("Frequency",weight="bold",fontsize=12)

plt.title("Nnumber Peope shot when Police not wearing camera",weight="bold",fontsize=14)





plt.show()
camera_yes = df[df["body_camera"]==True]

yes_camera_pol =camera_yes['race'].value_counts()

sns.barplot(yes_camera_pol.index, yes_camera_pol.values, alpha=0.8,palette="winter")

plt.xlabel("Race",weight="bold",fontsize=12)

plt.ylabel("Frequency",weight="bold",fontsize=12)

plt.title("Number Peope shot when Police wearing camera",weight="bold",fontsize=14)

plt.show()
daily_shootouts = df[['date']]

daily_shootouts['kills']=1

daily_shootouts=daily_shootouts.groupby('date').sum()

daily_shootouts = daily_shootouts.reset_index()

daily =daily_shootouts 
daily.set_index("date").plot()

plt.xlabel("Date",weight="bold",fontsize=12)

plt.ylabel("Count of Shootouts",weight="bold",fontsize=12)

plt.title("Distribution of daywise",weight="bold",fontsize=14)

plt.show()

plt.show()
df.head()
innocent = df[(df.signs_of_mental_illness==False) & (df.armed == "unarmed") & (df.flee=="Not fleeing")]
innnocent_people = innocent["race"].value_counts()

innnocent_people = pd.DataFrame(innnocent_people).reset_index()

innnocent_people
race = innnocent_people['index'].tolist()

data_counts  = innnocent_people['race'].tolist()
fig, ax = plt.subplots(figsize=(10,10), subplot_kw=dict(aspect="equal"))



states = race



data = data_counts



def func(pct, allvals):

    absolute = int(pct/100.*np.sum(allvals))

    return "{:.1f}%\n({:d} )".format(pct, absolute)





wedges, texts, autotexts = ax.pie(data, autopct=lambda pct: func(pct, data),

                                  textprops=dict(color="w"))



ax.legend(wedges, states,

          title="States",

          loc="center left",

          bbox_to_anchor=(1, 0, 0.5, 1))



plt.setp(autotexts, size=8, weight="bold")



ax.set_title("Innocent People by Race",weight="bold",fontsize=14)



plt.show()
year_wise = df['year'].value_counts()
g=sns.barplot(year_wise.index, year_wise.values, alpha=0.8)

plt.xlabel("Race",weight="bold",fontsize=12)

plt.ylabel("Frequency",weight="bold",fontsize=12)

plt.title("Nnumber Peope shot by every year",weight="bold",fontsize=14)





plt.show()
kills_per_year = df[['year','race']]

kills_per_year ['kills'] =1

temp = kills_per_year[["year","race",'kills']].groupby(["year","race"]).sum().add_prefix("Sum_of_").reset_index()

temp1 = temp.pivot("year","race","Sum_of_kills")

ax = temp1.plot(kind='bar',stacked=True)

plt.xticks(rotation=0)

plt.xlabel("Year",fontweight ="bold",fontsize=14)

plt.ylabel('Frequency',fontweight ="bold",fontsize=14)

plt.title("Year wise distribution of shot death",fontweight ="bold",fontsize=16)

plt.show()
kills_per_month = df[['month','race']]

kills_per_month ['kills'] =1

temp = kills_per_month[["month","race",'kills']].groupby(["month","race"]).sum().add_prefix("Sum_of_").reset_index()

temp1 = temp.pivot("month","race","Sum_of_kills")

ax = temp1.plot(kind='bar',stacked=True)

plt.xticks(rotation=0)

plt.xlabel("Monht",fontweight ="bold",fontsize=14)

plt.ylabel('Frequency',fontweight ="bold",fontsize=14)

plt.title("Month wise distribution of shot death",fontweight ="bold",fontsize=16)

plt.show()
mental_illness = df[df.signs_of_mental_illness==True]

kills_per_year = mental_illness[['year','race']]

kills_per_year ['kills'] =1

temp = kills_per_year[["year","race",'kills']].groupby(["year","race"]).sum().add_prefix("Sum_of_").reset_index()

temp1 = temp.pivot("year","race","Sum_of_kills")

ax = temp1.plot(kind='bar',stacked=True)

plt.xticks(rotation=0)

plt.xlabel("Year",fontweight ="bold",fontsize=14)

plt.ylabel('Frequency',fontweight ="bold",fontsize=14)

plt.title("Mental illnes and shooting across different races",fontweight ="bold",fontsize=16)

plt.show()
kills_per_year = df[['year','city']]

kills_per_year ['kills'] =1

temp = kills_per_year[["year","city",'kills']].groupby(["year","city"]).sum().add_prefix("Sum_of_").reset_index()[:20]

# temp1 = temp.pivot("year","city","Sum_of_kills")

# ax = temp1.plot(kind='bar',stacked=True)

# plt.xticks(rotation=0)

# plt.xlabel("Year",fontweight ="bold",fontsize=14)

# plt.ylabel('Frequency',fontweight ="bold",fontsize=14)

# plt.title("City with most Poeple killing year wise",fontweight ="bold",fontsize=16)

# plt.show()
temp
import datetime
# Feature Generation

df['date']=pd.to_datetime(df['date'])

# df['year']=pd.to_datetime(df['date']).dt.year

# df['month']=pd.to_datetime(df['date']).dt.month

df['month_name']=df['date'].dt.strftime('%B')

df['month_num']=df['date'].dt.strftime('%m')

df['weekdays']=df['date'].dt.strftime('%A')  

df['date_num']=df['date'].dt.strftime('%d').astype(int)

df['date_categ']=np.where(df['date_num']<16,"First Half","Second Half")

df['date_mon']=df.date.dt.to_period("M")
df.head()
kills_per_year_gen = mental_illness[['year','gender']]

kills_per_year_gen ['kills'] =1

temp = kills_per_year_gen[["year","gender",'kills']].groupby(["year","gender"]).sum().add_prefix("Sum_of_").reset_index()

temp1 = temp.pivot("year","gender","Sum_of_kills")

ax = temp1.plot(kind='bar',stacked=False)

plt.xticks(rotation=0)

plt.xlabel("Year",fontweight ="bold",fontsize=14)

plt.ylabel('Frequency',fontweight ="bold",fontsize=14)

plt.title("Shootouts year wise by gender",fontweight ="bold",fontsize=16)

plt.show()