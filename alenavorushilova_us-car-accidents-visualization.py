import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objects as go

import matplotlib.ticker as ticker





%matplotlib inline
df = pd.read_csv('../input/us-accidents/US_Accidents_Dec19.csv')

df.head()
df.describe().T
df.info()
df['Source'].unique()
fig, ax = plt.subplots(figsize=(10,6))

df.groupby('Source').size().plot(kind = 'barh', 

                                 color = 'salmon',

                                 edgecolor = 'r',

                                 linewidth = 0.8,

                                 width = 0.2,

                                 align = 'center',

                                xerr=np.std(df.groupby('Source').size()),

                                 grid = True)

ax.set_title('Reporting\nSource', fontsize=22);

fig, ax = plt.subplots(figsize=(12,8))

#fig.set_facecolor('lightgrey') #changes color around the plot area

#fig.set_axis_bgcolor('lightgrey')

df.groupby('State').size().plot(kind = 'bar', 

                                 colormap='Spectral',

                                 edgecolor = 'r',

                                 linewidth = 0.8,

                                 width = 0.8,

                                 align = 'center')

ax.set_title('Reporting by State', fontsize=22, style='italic')

ax.grid(linestyle=':', linewidth = '0.2', color ='salmon')

ax.spines['left'].set_visible(True)

ax.spines['right'].set_visible(False)

ax.spines['top'].set_visible(False)

ax.spines['bottom'].set_visible(False);

df_st_ct = pd.value_counts(df['State'])



fig = go.Figure(data=go.Choropleth(

    locations=df_st_ct.index,

    z = df_st_ct.values.astype(float),  # Data to be color-coded

    locationmode = 'USA-states',     # set of locations match entries in `locations`

    colorscale = 'YlOrRd',

    colorbar_title = "Count",

))



fig.update_layout(

    title_text = 'US Accidents by State',

    geo_scope='usa', # limite map scope to USA

)



fig.show()
df.groupby('Severity').size()
df_sev = df.groupby('Severity').size()

df_sev = df_sev[[2,3,4]]

cols = ['bisque', 'rosybrown', 'palegoldenrod']

cases = ['Severity 2','Severity 3','Severity 4']

plt.figure(figsize=(10,6))

plt.pie(df_sev,

        colors = cols,

        labels= cases,

        explode = (0,0,0.2), #moving slices apart

        autopct = ('%1.1f%%')) #to display %

plt.title('Types \nof Severity', weight='heavy', fontsize=22, style='italic');
fig, ax = plt.subplots(figsize=(10, 6))

ax.set_title('Scatterplot', fontsize=22)

ax.plot(df['Severity'], df['Visibility(mi)'], 'ko');
fig=sns.heatmap(df[['TMC','Severity','Distance(mi)',

                    'Temperature(F)','Wind_Chill(F)','Humidity(%)',

                    'Pressure(in)','Visibility(mi)','Wind_Speed(mph)']].corr(),

                annot=True,cmap='RdBu',linewidths=0.2,annot_kws={'size':15})

fig=plt.gcf()

fig.set_size_inches(12,8)

plt.xticks(fontsize=10)

plt.yticks(fontsize=10)

plt.show()
start = pd.to_datetime(df.Start_Time, format='%Y-%m-%d %H:%M:%S')

end = pd.to_datetime(df.End_Time, format='%Y-%m-%d %H:%M:%S')

laps=end-start
top_15 = laps.astype('timedelta64[m]').value_counts().nlargest(15) #Return the first n rows ordered by columns in descending order.

print('Top 15 longest accidents correspond to {:.1f}% of the data'.format(top_15.sum()*100/len(laps)))

(top_15/top_15.sum()).plot.bar(figsize=(10,8), color = 'plum')

plt.title('Top Accident Durations', fontsize = 24, color='indigo')

plt.xlabel('Duration in minutes')

plt.ylabel('% of Total Data')

plt.grid(linestyle=':', linewidth = '0.2', color ='salmon');
df_st = df.groupby('State').size().to_frame('Counts')

df_st = df_st.reset_index().sort_values('Counts', ascending = False)[:10]

df_st = df_st[::-1]   # flip values from top to bottom



colors = ['olivedrab', 'gold', 'coral', 'thistle',

     'palevioletred', 'peru', 'lightblue', 'lightsalmon', 'lightgreen']



fig, ax=plt.subplots(figsize=(15,8))

ax.barh(df_st['State'], df_st['Counts'], color = colors)



for i, (value, name) in enumerate(zip(df_st['Counts'], df_st['State'])):

        ax.text(value, i,     name,           size=14, weight=600, ha='right', va='bottom')

        ax.text(value, i-.25,     f'{value:,.0f}',  size=14, ha='left',  va='center')

        

# ... polished styles

#ax.text(1, 0.4, transform=ax.transAxes, color='#777777', size=46, ha='right', weight=800)

ax.text(0, 1.06, 'by State', transform=ax.transAxes, size=12, color='#777777')

ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

ax.xaxis.set_ticks_position('top')

ax.tick_params(axis='x', colors='#777777', labelsize=12)

ax.set_yticks([])

ax.margins(0, 0.01)

ax.grid(which='major', axis='x', linestyle='-')

ax.set_axisbelow(True)

ax.text(0, 1.12, '10 States with the Highest Accident Rate',

            transform=ax.transAxes, size=24, weight=600, ha='left')

ax.text(1, 0, 'by @alenavorushilova', transform=ax.transAxes, ha='right',

            color='#777777', bbox=dict(facecolor='white', alpha=0.8, edgecolor='white'))

plt.box(False)