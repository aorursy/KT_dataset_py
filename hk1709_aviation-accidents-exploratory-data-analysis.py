#Import Standard Libraries and Packages

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style='darkgrid')



import plotly.express as px

import plotly.graph_objects as go

#from plotly.offline import iplot, init_notebook_mode

#init_notebook_mode()



from warnings import filterwarnings

filterwarnings("ignore")
#Read dataset

aviation1 = pd.read_csv("../input/aviation-accident-database-synopses/AviationData.csv", encoding='ISO-8859-1')
#View the dataset

aviation1.head()
#Drop irrelevant columns

columns = ['Event.Id', 'Investigation.Type','Accident.Number', 'Airport.Code', 'Airport.Name', 

           'Registration.Number', 'Air.Carrier', 'Schedule', 'FAR.Description']

aviation2 = aviation1.drop(columns, axis=1)

aviation2.head()
aviation2['City'] = aviation2['Location'].str.split(',').str[0]

aviation2['State'] = aviation2['Location'].str.split(',').str[1]

aviation2['InjurySeverityType'] = aviation2['Injury.Severity'].str.split('(').str[0]
aviation3 = aviation2.drop(['Location','Injury.Severity'],axis=1)

numeric_columns = aviation3.select_dtypes(include=['float64']).columns

aviation3[numeric_columns] = aviation3[numeric_columns].fillna(0)

aviation = aviation3.fillna("UNKNOWN")

aviation.head()
# Count of Accidents as per Countries

accidents = aviation['Country'].groupby(aviation['Country']).count()

accidents
# Plot Country

fig = plt.figure(figsize=(20,15))

accidents_Count = accidents[aviation['Country'].groupby(aviation['Country']).count() > 200]

accidents_Count_df = pd.DataFrame({'Country':accidents_Count.index,'Count':accidents_Count.values})

plt.bar(accidents_Count_df['Country'], height = accidents_Count_df['Count'], color='red')

plt.xticks(rotation=90)

plt.xlabel("Country", size=15)

plt.ylabel("Count of Accidents", size=15)

y=accidents_Count_df['Count']

for i,v in enumerate(y):

    plt.text(x=i, y=v, s=str(v), horizontalalignment='center', size=15)

plt.title("Distribution of Accidents wrt Countries", size=20)

plt.show()
aviation_State = pd.DataFrame(aviation['Total.Fatal.Injuries']+aviation['Total.Serious.Injuries']+aviation['Total.Minor.Injuries'])

aviation_State = aviation_State.rename(columns={0:"Injuries/Fatalities"})

aviation_merged = pd.concat([aviation,aviation_State], axis=1)

aviation_merged.head()
aviation_merged['text'] = aviation_merged['Country'] + '<br>Injuries/Fatalities ' + (aviation_merged['Injuries/Fatalities'].astype(str))

colors = ["darkblue","yellow","seagreen","purple","cyan","orange"]

limits = [(0,50),(50,100),(100,150),(150,175),(175,200),(350,400)]

#scale = 10

accidentState = []



fig = go.Figure()



for i in range(len(limits)):

    lim = limits[i]

    Injuries_Count = aviation_merged[((aviation_merged['Injuries/Fatalities'] > lim[0]) & (aviation_merged['Injuries/Fatalities'] <lim[1]))]

    fig.add_trace(go.Scattergeo(

        locationmode = 'country names',

        lon = Injuries_Count['Longitude'],

        lat = Injuries_Count['Latitude'],

        opacity = 0.8,

        text = Injuries_Count['text'],

        marker = dict(

            size = Injuries_Count['Injuries/Fatalities'],

            color = colors[i],

            line_color='rgb(40,40,40)',

            line_width=0.5,

            sizemode = 'area'

        ),

        name = '{0} - {1}'.format(lim[0],lim[1])))

    accidentState.append(Injuries_Count)



fig.update_layout(

        title_text = 'Distribution of Injuries/Fatalities in the World<br>(Click legend to toggle traces)',

        showlegend = True,

        geo = dict(

            scope = 'world',

            landcolor = 'rgb(217, 217, 217)',

        )

    )

fig.show()
plt.figure(figsize=(20,15))

aviation_US = aviation[aviation['Country'] == 'United States']

sns.countplot(x='State', data=aviation_US, order = aviation_US['State'].value_counts().index)

plt.xticks(rotation=90)

plt.xlabel("States", size=15)

plt.ylabel("Count of Accidents", size=15)

plt.title("Distribution of Accidents in the United States", size=20)

y=aviation_US['State'].value_counts()

for i, v in enumerate(y):

    plt.text(i,v,str(v), horizontalalignment='center', verticalalignment='bottom', fontsize=12, rotation=90)

plt.show()
# Evaluate the Notorious Companies



plt.figure(figsize=(20,15))

aviation['Make'].str.upper().value_counts().sort_values(ascending=False)[:10].plot(kind='bar', color='Green')

plt.xticks(rotation=90)

plt.xlabel("Make", size=15)

plt.ylabel("Count of Accidents", size=15)

plt.title("Top 10 Notorious Companies in the United States", size=20)

y=aviation['Make'].str.upper().value_counts().sort_values(ascending=False)[:10]

for i, v in enumerate(y):

    plt.text(i, v, str(v), fontsize=15, style='oblique', horizontalalignment='center')

plt.show()

# Load the example flights dataset and convert to long-form

data1 = aviation_merged[aviation_merged['Make'].str.upper().isin(["CESSNA","PIPER","BEECH","BELL"])]

data2 = data1.replace('Cessna','CESSNA')

data3 = data2.replace('Bell','BELL')

data4 = data3.replace('Beech','BEECH')

data = data4.replace('Piper','PIPER')

table1 = pd.pivot_table(data, index=['Model'], columns=['Make'], values='Event.Date', aggfunc=np.count_nonzero, fill_value=0)

table2 = table1[(table1.values > 300)]



# Draw a heatmap with the numeric values in each cell

fig, ax = plt.subplots(figsize=(12, 12))

sns.heatmap(table2, annot=True, fmt="d", linewidths=.1, ax=ax, linecolor='white', cmap='YlGnBu')

plt.xlabel("Make", size=15)

plt.ylabel("Model", size=15)

plt.title("Analysis of Notorious Models of Top 4 Companies involved in accidents", size=20)

fig.show()
aviation_merged['Year'] = pd.DatetimeIndex(aviation_merged['Event.Date']).year

aviation_merged['Month'] = pd.DatetimeIndex(aviation_merged['Event.Date']).month_name()

aviation_merged.head()
fig = px.scatter(aviation_merged[(aviation_merged['Injuries/Fatalities'] > 10) & (aviation_merged['Country'] == 'United States')],

                 x="Year", y="Injuries/Fatalities", animation_frame="Year", animation_group="State",

                 size="Injuries/Fatalities", color="Aircraft.Damage", hover_name="State",

                 size_max=100, range_x=[1980,2020], range_y=[0,400],

                 category_orders ={'Year':[1962,1979,1980,1981,1982,1983,1984,1985,1986,1987,1988,1989,1990,1991,

                                           1992,1993,1994,1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,

                                           2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,

                                           2020]})

fig.show()
plt.figure(figsize=(20,15))

splot = sns.countplot(data=aviation_merged, x="Broad.Phase.of.Flight", hue='Weather.Condition', edgecolor=(0,0,0), linewidth=1, palette='Reds')

plt.xticks(rotation=90)

plt.xlabel("Broad Phase of Flight", size=15)

plt.ylabel("Number of Occurrences", size=15)

plt.title("What was the Phase of Flight when Accident took place?", size=20)

for p in splot.patches:

    splot.annotate(format(p.get_height()), (p.get_x() + p.get_width() / 2., 

    p.get_height()), ha = 'center', va = 'bottom', xytext = (0, 10), textcoords = 'offset points', rotation=90, size=11.5)

plt.show()
cplot = sns.catplot(x="Engine.Type", col="InjurySeverityType", kind="count", col_wrap=2,

            data=aviation_merged[aviation_merged['InjurySeverityType'].isin(['Fatal','Non-Fatal'])], height=4,

                 legend_out=True, size=8, palette="Set3", edgecolor=(0,0,0))

cplot.set_xticklabels(rotation=90)

cplot.fig.suptitle("What was the Engine Type of Fatal or Non-Fatal accidents?", size=20)

cplot.fig.subplots_adjust(top=.9)

cplot.set_titles("Injury Severity Type is {col_name}")

cplot.set(ylim=[0,65000])

for ax in cplot.axes.ravel():

    for p in ax.patches:

        ax.annotate(format(p.get_height()), (p.get_x() + p.get_width() / 2, 

        p.get_height()), ha = 'center', va = 'bottom', xytext = (0, 10), textcoords = 'offset points', rotation=90, size=12)

plt.show()