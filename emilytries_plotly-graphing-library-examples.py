# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory





import warnings

warnings.filterwarnings('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv('../input/all-space-missions-from-1957/Space_Corrected.csv')

data.head()
data.columns = data.columns.str.strip()

data.columns
data['Rocket'] = data['Rocket'].str.replace(',', '.')

data['Rocket'] = data['Rocket'].str.split('.').str[0]
data['Rocket'] = data['Rocket'].astype(np.float64)

data['Rocket']  = data['Rocket'].values*1000000
#Drop some columns

data.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1, inplace=True)
data.info()
#Split Last Part in Location Feature

data['Country'] = data['Location'].str.split(',').str[-1]

data['City'] = data['Location'].str.split(',').str[-2]
#Extract Date Time

data['date'] = pd.to_datetime(data['Datum'])

data['year'] = data['date'].apply(lambda datetime: datetime.year)

data['month'] = data['date'].apply(lambda datetime: datetime.month)

data['day'] = data['date'].apply(lambda datetime: datetime.day)

data['weekday'] = data['date'].apply(lambda datetime: datetime.weekday())

data['hour'] = data.date.apply(lambda x : x.hour)
#Get Day Name

data.loc[data['weekday'] == 0, 'weekday_name'] = 'Monday'  

data.loc[data['weekday'] == 1, 'weekday_name'] = 'Tuesday'

data.loc[data['weekday'] == 2, 'weekday_name'] = 'Wednesday'

data.loc[data['weekday'] == 3, 'weekday_name'] = 'Thursday'

data.loc[data['weekday'] == 4, 'weekday_name'] = 'Friday'

data.loc[data['weekday'] == 5, 'weekday_name'] = 'Saturday'

data.loc[data['weekday'] == 6, 'weekday_name'] = 'Sunday'
#Get More Info to Detail Column

data['Rocket Family Member'] = data['Detail'].str.split('|').str[0]

data['Rocket Family Member'] = data['Rocket Family Member'].str.split(' ').str[0]

data['Rocket Family Member'] = data['Rocket Family Member'].str.split('-').str[0]
#Drop some columns

data.drop(['Location', 'Datum', 'Detail'], axis=1, inplace=True)
data = data.sort_values(['year', 'month','day', 'weekday'], ascending = [True, True,True,True])

data.reset_index(inplace=True)

data.head()
data.loc[(data['hour'] >=0) | (data['hour'] <12) , 'Time Period'] = 'AM'

data.loc[data['hour'] >=12 , 'Time Period'] = 'PM'
data.loc[data['hour'] == 4, 'weekday_name'] = 'Friday'
data.drop(['index','weekday'], axis=1, inplace=True)

data.head()
data['Country'] = data['Country'].str.strip()



data.loc[data['Country'] == 'Kazakhstan', 'Country'] = 'Russia'

data.loc[data['Country'] == 'New Mexico', 'Country'] = 'USA'

data.loc[data['Country'] == "Yellow Sea", 'Country'] = "China"

data.loc[data['Country'] == "Shahrud Missile Test Site", 'Country'] = "Iran"

data.loc[data['Country'] == "Pacific Missile Range Facility", 'Country'] = "USA"

data.loc[data['Country'] == "Barents Sea", 'Country'] = 'Russia'

data.loc[data['Country'] == "Gran Canaria", 'Country'] = 'USA' 
# Missing Value 

import plotly.express as px



# Data to plot

labels = data.columns

sizes = data.isnull().sum()



fig = px.pie(values=sizes, names=labels, title='Percentage of Missing Value')

fig.show()
data['date'] = pd.to_datetime(data['date'] , utc=True)

data['date'] = pd.to_datetime(data['date']).dt.date

s=pd.to_datetime(data['date'], format='%Y-%m-%d')
sv=pd.DataFrame(s)

data['date']=sv
#Not null values



xy=data.copy()

xy=xy[xy.notnull().all(axis=1)]
import plotly.express as px



fig = px.line(xy, x="date", y="Rocket",color='Country', title='Cost of the mission: in $ million without NAN values')

fig.show()
fig = px.line(xy, x="date", y="Rocket",color='Company Name', title='Cost of the mission: in $ million without NAN values')

fig.show()
abc=data[['Country','Status Mission','year']]

abc=abc.groupby(['Country','Status Mission', 'year']).size().reset_index(name='count')

abc.head()


fig = px.bar(abc, x="year", y="count",color="Country" , title="Availability of countries by years")

fig.show()


fig = px.bar(abc, x="year", y="count", color="Status Mission",  title="Completion status of the mission by year")

fig.show()
bcd=data[['Country','Status Mission','month']]

bcd=bcd.groupby(['Country','Status Mission', 'month']).size().reset_index(name='count')

bcd.head()
fig = px.bar(bcd, x="month", y="count",color= "Status Mission", title="Completion status of the mission by month")

fig.show()
fig = px.bar(bcd, x="month", y="count",color= "Country", title="Completion status of the mission by month")

fig.show()
fgh=data[['year','weekday_name']]

fgh=fgh.groupby(['year','weekday_name']).size().reset_index(name='count')



import plotly.express as px

fig = px.bar(fgh, x="year", y="count", title="Number of Space missions of Days in Years",

             color='weekday_name',

             height=400)

fig.show()
klm=xy[['Status Mission', 'Time Period', 'weekday_name','Rocket']]

klm=klm.groupby(['Status Mission', 'Time Period', 'weekday_name','Rocket']).size().reset_index(name='count')



import plotly.express as px

fig = px.scatter(klm, x="count", y="Rocket", color="Status Mission",

             facet_row="Time Period", facet_col="weekday_name",

                 category_orders={"weekday_name": ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],

                              "Time Period": ["AM", "DM"]},title="When Cost of the mission NotNull ")

fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))

fig.show()
xyz=xy.groupby(['Country','Status Rocket'])['Rocket'].sum()

xyz=pd.DataFrame(xyz)

xyz.reset_index(inplace=True)





fig = px.bar(xyz, y='Rocket', x='Country', text='Rocket',color='Status Rocket',title="When Cost of the mission NotNull ")

fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')



fig.show()
fig = px.scatter(xy, x="Rocket", y="Country", color="Status Mission",

                 labels={"Rocket":"Cost of the mission"},title="When Cost of the mission NotNull ")



fig.show()
bnm=xy.loc[xy['Status Rocket']=='StatusActive']

fig = px.scatter(bnm, x="year", y="Rocket",

                 size="Rocket", color="Company Name",

                 hover_name="Country", log_x=True, size_max=60,title="When Cost of the mission NotNull ")

fig.show()
bnm=xy.loc[xy['Status Rocket']=='StatusActive']

fig = px.scatter(bnm, x="year", y="Rocket",

                 size="Rocket", color="Country",

                 hover_name="Country", log_x=True, size_max=60,title="When Cost of the mission NotNull ")

fig.show()
fig = px.sunburst(data, path = ['Company Name','Rocket Family Member'], title = 'Company and their Rocket Name',height = 700)



fig.show()
fig = px.sunburst(data, path = ['Country','Company Name'], title = 'Country and their Space Organisations',height = 700)



fig.show()
import plotly.express as px



fig = px.sunburst(data, path=['Country', 'Status Rocket', 'Status Mission'], color='Country')

fig.show()
fig = px.treemap(xy, path=['weekday_name', 'Time Period', 'Country'], values='Rocket',title="When Cost of the mission NotNull ")

fig.show()