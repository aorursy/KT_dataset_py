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
%matplotlib inline 



import matplotlib

import matplotlib.pyplot as plt 

import seaborn as sns

import plotly.graph_objs as go

import plotly.express as px

import plotly.io as pio

pio.templates.default = "plotly"

import warnings

from datetime import date, timedelta

warnings.filterwarnings('ignore')
df= pd.read_csv('../input/covid19-in-india/covid_19_india.csv')

df.head()
df.shape
df.info()
df['Date'] = pd.to_datetime(df['Date'])

from datetime import date

recent=df[['Date']][-1:].max()

df_update=df.loc[df.Date==pd.Timestamp(recent['Date'])]

df_update
df_update.isnull().sum()
fig = plt.figure(figsize=(10,10))

conf_per_country = df_update.groupby('State/UnionTerritory')['Confirmed'].sum().sort_values(ascending=True)

conf_sum=df_update['Confirmed'].sum()

def absolute_value(val):

    a  = val

    return (np.round(a,2))

conf_per_country.plot(kind="pie",title='Percentage of confirmed cases in India',autopct=absolute_value)



plt.show ()
fig = plt.figure(figsize=(10,10))

conf_per_country = df_update.groupby('State/UnionTerritory')['Cured'].sum().sort_values(ascending=False)

conf_sum=df_update['Confirmed'].sum()

def absolute_value(val):

    a  = val

    return (np.round(a,2))

conf_per_country.plot(kind="pie",title='Percentage of Cured cases in India',autopct=absolute_value)



plt.show ()
group_cases=df_update[['Confirmed','Cured','Deaths','State/UnionTerritory']].groupby('State/UnionTerritory').sum().sort_values('Confirmed',ascending=False).head(5)

group_cases=group_cases.reset_index()

group_cases
f, ax = plt.subplots(figsize=(15, 10))

bar1=sns.barplot(x="Confirmed",y="State/UnionTerritory",data=group_cases,label="Confirmed", color="#34495e")



bar2=sns.barplot(x="Cured", y="State/UnionTerritory", data=group_cases,label="Cured", color="#2ecc71")



bar3=sns.barplot(x="Deaths", y="State/UnionTerritory", data=group_cases,label="Deaths", color="#e74c3c")



ax.legend(loc=4, ncol = 1)

plt.xlabel("Confirmed Cases")

plt.show()
for template in [ "ggplot2", "seaborn"]:

    fig = px.scatter(df_update,

                     x="Deaths", y="Cured",size="Cured",color="State/UnionTerritory",

                     log_x=True, size_max=60,

                     template=template, title="India Fighting Covid 2020: '%s' theme" % template)

    fig.show()
group_cases=df_update[['Cured','Deaths','State/UnionTerritory']].groupby('State/UnionTerritory').sum().sort_values('Deaths',ascending=False).head()

group_cases.plot(kind='bar',width=0.5,colormap='Pastel2',figsize=(15,10))

plt.show()
fig = go.Figure(

    data=go.Surface(z=df_update.values),

    layout=go.Layout(

        title="Covid_Fights",

        width=500,

        height=500,

    ))



for template in ["plotly_white", "seaborn"]:

    fig.update_layout(template=template, title="Covid_Fight : '%s' theme" % template)

    fig.show()
df_now=df.groupby('Date')[['Confirmed','Deaths']].sum().reset_index()

df_now
dt = df[df['State/UnionTerritory']=='Karnataka']

dt
age_df= pd.read_csv('../input/covid19-in-india/AgeGroupDetails.csv')

age_df.head()
age_df.info()
explode = []

for i in  list(age_df['AgeGroup']):

    explode.append(0.05)

    

plt.figure(figsize= (15,10))

plt.pie(list(age_df['TotalCases']), labels= list(age_df['AgeGroup']), autopct='%1.1f%%', startangle=9, explode =explode, shadow = True)

centre_circle = plt.Circle((0,0),0.50,fc='white')



fig = plt.gcf()

fig.gca().add_artist(centre_circle)

plt.axis('equal')  

plt.show()
individual_details_df = pd.read_csv('../input/covid19-in-india/IndividualDetails.csv')

pattern = (list(individual_details_df['gender'].value_counts()))

plt.figure(figsize= (15,10))

plt.pie(pattern, labels = ['Male', 'Female'], autopct='%1.1f%%', shadow=True)

plt.title('Percentage of Gender (Ignoring the Missing Values)',fontsize = 20)

centre_circle = plt.Circle((0,0),0.50,fc='white')

fig = plt.gcf()

fig.gca().add_artist(centre_circle)

plt.axis('equal')  

plt.show()
start_date = date(2020, 3,25) #Phase 1 started 

end_date = date.today()

delta = timedelta(days=1)

end_date = end_date-delta

daily_mortality = []

daily_cured = []

dates = []



while start_date <= end_date:

    daily = df[df["Date"]==(str(start_date.strftime("%d/%m/%y")))]

    try:

        mor = sum(daily['Deaths']) / (sum(daily['Confirmed'])-(sum(daily['Cured'])+sum(daily['Deaths'])))

    except ZeroDivisionError:

        mor = 0.0

    try:

        cur = sum(daily['Cured']) / (sum(daily['Confirmed'])-(sum(daily['Cured'])+sum(daily['Deaths'])))

    except ZeroDivisionError:

        cur = 0.0

    start_date += delta

    dates.append(str(start_date.strftime("%d/%m/%y")))

    daily_mortality.append(mor)

    daily_cured.append(cur)

    

plt.figure(figsize=(16,7))

plt.plot(dates,daily_mortality)

plt.plot(dates,daily_cured)

plt.xticks(dates, dates)

plt.xticks(rotation=90)

plt.xlabel("dates")

plt.ylabel("Mortality rates")

plt.title("Mortality rates 26/03 to 26/05 ", size=25)

plt.legend()



plt.show()