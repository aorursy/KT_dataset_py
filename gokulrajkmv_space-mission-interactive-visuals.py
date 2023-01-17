import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# data visual packages

import matplotlib.pyplot as plt
import seaborn as sns

#interactive visual packages
import plotly as pio

from IPython.display import HTML 

from plotly import __version__

import cufflinks as cf

from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
init_notebook_mode(connected=True)
cf.go_offline()

import plotly.express as px
df = pd.read_csv('../input/all-space-missions-from-1957/Space_Corrected.csv')

df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df.info()
plt.figure(figsize=(14,6))
plt.title('Missing values in each columns')
sns.set_context(context='notebook',font_scale=1.5)
sns.heatmap(df.isnull(),cmap='Set3',cbar=False,yticklabels=False);
# let us start with missing values

Missing_values = 100 * df.isnull().sum() / len(df) 

Missing_values.iplot(kind='bar',title='Missing values in each columns in %',theme='white',color='#3DD8AD')
# cleaning 'Company Name' column




df['Company Name'] = df['Company Name'].str.replace(r'[^\s\w]','')


df['Company Name'] = df['Company Name'].astype(str) # converting object to string
df['Company Name'] = df['Company Name'].replace(r'\d','')# removing digit
df['Company Name'] = df['Company Name'].replace(r'\W','')# removing non-alphanumeric

df['Location'] = df['Location'].str.replace(r'[^\s\w]','')

df['Location'][0].split()[-1]

df['Country'] = df['Location'].apply(lambda word : word.split(' ')[-1])

# country 
df['Country'] = df['Country'].replace('Ocean','Russia')
df['Country'] = df['Country'].replace('Site','Iran')
df['Country'] = df['Country'].replace('Facility','USA')
df['Country'] = df['Country'].replace('Zealand','New Zealand')
df['Country'] = df['Country'].replace('Canaria','Spain')

list_index = [920,957,1304]
df.loc[list_index,'Country'] = 'Russia'

chi =[133]
df.loc[chi,'Country'] = 'China'

NK =[461,623,653,775,1294]
df.loc[NK,'Country'] = 'North Korea'

SK =[619,721,757]
df.loc[SK,'Country'] = 'South Korea'
# creating new columns from Datum named it Month,Year,Time and Day

df['Month'] = df['Datum'].apply(lambda x:x.split(' ')[1])
df['Year'] = df['Datum'].apply(lambda x:x.split(' ')[3])
df['Time'] = df['Datum'].apply(lambda x:x.split(' ')[-2])
df['Day'] = df['Datum'].apply(lambda x:x.split(' ')[0])

#
df['Year'] = pd.to_numeric(df['Year'])

# cleaning  'Datum' column

df['Datum'][0].split()[-2]
df[' Rocket'] = df[' Rocket'].str.replace(r'\W','')
df[' Rocket'] = df[' Rocket'].str.strip(',')

df[' Rocket'] = pd.to_numeric(df[' Rocket'])

import plotly.io as pio

pio.templates.default = "plotly"

dc = df['Company Name'].value_counts().reset_index()
dc.columns = ('Company name','count')
dc = dc.sort_values(['count'])

fig = px.bar(dc.tail(20), x='Company name',y='count',color='count',
            title='Top companies in the space Mission')

fig.show()
# data to plot(Country)

Country_list = df['Country'].value_counts()

ISO_Code = ['RUS','USA','KAZ','FRA','CHN','JPN','IND','IRN','NZL','ISR','KEN','AUS','PRK','MEX','KOR','BRA','ESP']

Counrtry_df = pd.DataFrame(Country_list).reset_index()

Counrtry_df['ISO'] = ISO_Code

Counrtry_df.columns=('Country_Name','Number_of_Mission','ISO_Code')
# plot country



fig = px.choropleth(Counrtry_df, locations="ISO_Code",
                    color="Number_of_Mission",
                    hover_name="Country_Name", 
                    color_continuous_scale=px.colors.sequential.Plasma,
                    title = "Number of space mission by each Nation")
fig.show()

#data for plot_ani_bar

plot_bar = df.groupby(['Year','Country'])['Company Name'].count().reset_index()

plot_bar.columns = ('Year','Country','Count')


fig = px.bar(plot_bar, x='Country',y='Count',color='Count',animation_frame = 'Year',
            title='Number of space mission each year by Nations', height=700)

fig.show()
#data for the plot number of space mission success and failure

launch_plot = df.groupby(['Status Mission','Year','Month','Country'])['Company Name'].count().reset_index()


launch_plot.columns = ('Status Mission','Year','Month','Country','No. of Mission')

fig = px.sunburst(launch_plot, path=['Status Mission','Country'], values='No. of Mission',
                  color_continuous_scale='Plasma',title= 'Number of Success and failure in Space Mission',
                  height=650)


fig.show()

# who has most success made most of money and who has fail the most cost losses

plot_comp = df.groupby(['Status Mission','Company Name'])['Day'].count().reset_index()

plot_comp.columns = ('Status Mission','Company Name','Count')

fig = px.sunburst(plot_comp, path=['Status Mission', 'Company Name'], values='Count',
                  color_continuous_scale='RdBu',height=650,title = 'Success, Failure and partial Failure of each Space Agency')

fig.show()
rocket_data = df[['Status Rocket','Country','Company Name']]

plt.figure(figsize=(12,6));
sns.countplot(df['Status Rocket'],palette='Set2');
plt.title('Rocket status');
sns.set_context(context='notebook',font_scale=1.5);
df_rocket = df.groupby('Country')[' Rocket'].sum().reset_index()
df_rocket.columns = ('Country','Amount_spent')
df_rocket = df_rocket.sort_values('Amount_spent',ascending =False)


fig = px.bar(df_rocket[0:7],x='Country',y='Amount_spent',color='Amount_spent',title='Top countries by amount spent on space mission')

fig.show()
df_rocket_co = df.groupby('Company Name')[' Rocket'].sum().reset_index()
df_rocket_co.columns = ('Company Name','Amount_spent')
df_rocket_co = df_rocket_co.sort_values('Amount_spent',ascending =False)


fig = px.bar(df_rocket_co[0:17],x='Company Name',y='Amount_spent',color='Amount_spent',title='Top Space agency by amount spent on space mission')

fig.show()
# mission status
data_plot_l = df['Status Mission'].value_counts().reset_index()
data_plot_l.columns = ['status', 'count']

fig = px.pie(data_plot_l, values='count', names="status", title='Mission Status', height=500)
fig.show()