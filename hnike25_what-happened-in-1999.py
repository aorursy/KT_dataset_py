import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
%matplotlib inline
from matplotlib import cm
sns.set_style('ticks')
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import warnings
from wordcloud import WordCloud
import os
print(os.listdir("../input"))
df = pd.read_csv('../input/african_conflicts.csv',encoding="latin1",low_memory=False)
# Any results you write to the current directory are saved as output.
# Lowercase all the columns name
df.rename(str.lower, axis = 'columns', inplace = True)
# I will reduce the dataset to the columns I will used the most
df = df[['actor1', 'actor2','country', 'event_date', 'event_type', 'fatalities','latitude', 'location',
       'longitude', 'notes', 'source', 'year']]
df.head()
# checking for null Values in percent
round((df.isnull().sum()/df.shape[0])*100,2)
#change a long country name to an alias
df.loc[df['country'] == 'Democratic Republic of Congo','country'] = 'DR congo'
df.country.value_counts().head()
data_1 = df.country.value_counts().sort_values(ascending=False).head(10)
x=data_1.index
y= data_1.values

trace1 = go.Bar(
    x=x,
    y=y,
    text = y,
    textposition = 'auto',
    textfont = {'size':12,'color':'black'},
    marker=dict(
    color='SlateGray'),
    opacity=0.8,
    orientation ='v',
)

data = [trace1]

layout = go.Layout (
    yaxis = dict (
    title = 'Numbers of Conflitc'),
    
    xaxis = dict (
    title = 'Country'),
    
    title = 'Top 10 Countries with Highest Conflicts'
)
fig = go.Figure (data=data, layout = layout)
py.iplot(fig)
data_2 = df.groupby('country')['fatalities'].sum().sort_values(ascending=False).head(10)
x=data_2.index
y= data_2.values

trace1 = go.Bar(
    x=x,
    y=y,
    text = y,
    textposition = 'auto',
    textfont = {'size':12,'color':'white'},
    marker=dict(
    color='darkred'),
    opacity=0.8,
    orientation ='v',
)

data = [trace1]

layout = go.Layout (
    
    xaxis = dict (
    title = 'Countries Name'),
    
    title = 'Countries with Highest Fatalities'
)
fig = go.Figure (data=data, layout = layout)
py.iplot(fig)
x=df.year.value_counts().sort_index().index
y= df.year.value_counts().sort_index().values
#Second Graph
x2=df.groupby('year')['fatalities'].sum().sort_index().index
y2= df.groupby('year')['fatalities'].sum().sort_index().values

trace1 = go.Bar(
    x=x,
    y=y,
    marker=dict(
    color='MidnightBlue'
    ),
    orientation = 'v',
    name = 'Conflicts',
)
trace2 = go.Scatter(
    x=x2,
    y=y2,
    mode = 'lines+markers',
    marker=dict(
    color='red'),
    name = 'Fatalities',
    yaxis='y2'
)
data = [trace1, trace2]
layout = go.Layout(
    
    xaxis=dict(
        title = 'Years',
        tickangle=45,
        range = [1996.5,2017.5],
        linecolor='grey',
        linewidth=2,
        autotick=False,
        ticks='outside',
    ),
    
    yaxis=dict(
        title='Numbers of Conflits',
        titlefont=dict(
            color='MidnightBlue')
    ),
    yaxis2=dict(
        title='Numbers of Fatalities',
        titlefont=dict(
            color='red'
        ),
        
    overlaying='y',
    side='right',
    ),
    barmode='group',
    bargap=0.15,
    bargroupgap=0.1,
    
    legend=dict(
        x=.5,
        y=1,
    ),
    title = 'Africa Conflits vs. Fatalities'
    )

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)

# Convert Event date from a str to datetime format
df.event_date = pd.to_datetime(df.event_date)
#Let extract Month, and ,years
df['year_extract']= df['event_date'].apply(lambda x: x.year) #this will extract years from the event_date format
df['month']= df['event_date'].apply(lambda x: x.month) #this will extract Month from the event_date format
df['day of week']= df['event_date'].apply(lambda x: x.dayofweek)
dmonth = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',
         9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
df['month'] = df['month'].map(dmonth)
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
df['day of week']= df['day of week'].map(dmap)
rt = df.groupby('country').sum()['fatalities'].sort_values(ascending=False)

data = [dict(
        type='choropleth',
        locations= rt.index,
        locationmode='country names',
        z=rt.values,
        text=rt.index,
        colorscale='Reds',
        marker=dict(line=dict(width=0.7)),
        colorbar=dict(autotick=False, tickprefix='', title=' Fatalities Scale'),
)]
layout = dict(title = 'Fatalities reported in Africa from [1997 - 2017]',
             geo = dict(
            showframe = False,
            showcoastlines = False,
            projection = dict(
                type = 'Mercator'
            )
        ),)
fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False)
rt2 = df.country.value_counts().sort_values(ascending=False)

data = [dict(
        type='choropleth',
        locations= rt2.index,
        locationmode='country names',
        z=rt2.values,
        text=rt2.index,
        colorscale='Viridis',
        marker=dict(line=dict(width=0.7)),
        colorbar=dict(autotick=False, tickprefix='', title=' Countries with Conflicts'),
)]
layout = dict(title = 'Conflicts reported in Africa from [1997 - 2017]',
             geo = dict(
            showframe = False,
            showcoastlines = False,
            scope = 'africa',
            showcountries = True,
            projection = dict(
                type = 'Mercator'
            )
        ),)
fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False)
import re
from nltk.corpus import stopwords
clean_1 = re.compile('[/(){}\[\]\|@,;]')
clean_2 = re.compile('[^0-9a-z #+_]')
def clean_text (text):
    text = text.lower()
    text = clean_1.sub(' ',text) # compile and replace those symbole by empty scpace
    text = clean_2.sub('',text)
    text_2 = [word.strip() for word in text.split() if not word in set(stopwords.words('english'))]
    new_text = ''
    for i in text_2:
        new_text +=i+' '
    text = new_text
    return text.strip()
# remove null value from column 'notes'
note_data = df.dropna(subset=['notes'])
note_data['notes'] = note_data['notes'].apply(lambda x: " ".join(x.lower() for x in x.split()))
note_data['notes'] = note_data['notes'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))
from wordcloud import WordCloud

wc = WordCloud(max_font_size=50, width=600, height=300,colormap='Blues')
wc.generate(' '.join(note_data['notes'].values))

plt.figure(figsize=(15,8))
plt.imshow(wc,interpolation="bilinear")
plt.title("Most Used Words by New Agencies", fontsize=35)
plt.axis("off")
plt.show() 
# Lower all word in event_type
df.event_type = df.event_type.apply(lambda x: x.lower())
event_data = df.groupby('event_type').sum().reset_index()
# Create a new columns that count the numbers of counflicts 
d = dict(df.event_type.value_counts())
event_data['conflicts'] = event_data['event_type'].map(d)
# Sort the data by Fatalities
event_data.sort_values(by='fatalities', ascending=False,inplace=True)
#reduce the data to only 8 event type
event_data = event_data.head(8)
f, ax = plt.subplots(1,1,figsize = (10,10))
ax = event_data[['fatalities', 'conflicts']].plot(kind='barh',ax=ax,width=0.8,
              color=['dodgerblue', 'slategray'], fontsize=13);

ax.set_title("Causes of Conflicts in Africa",fontsize=20)
ax.set_ylabel("Event Type", fontsize=15)

ax.set_yticklabels(event_data.event_type.values)

# set individual bar lables using above list
for i in ax.patches:
    # get_width pulls left or right; get_y pushes up or down
    ax.text(i.get_width()+750, i.get_y()+.25, \
            str(int(round(((i.get_width())/1000))))+'k', fontsize=12, color='black')

# invert for largest on top 
ax.invert_yaxis()
sns.despine(bottom=True)
x_axis = ax.axes.get_xaxis().set_visible(False) # turn off the y axis label
plt.legend(loc=(1.0,0.98),fontsize=13,ncol=2)
plt.show()
## Extract 1999 dataset
df_99 = df[df.year==1999]
rt3 = df_99.country.value_counts().sort_values(ascending=False)

data = [dict(
        type='choropleth',
        locations= rt3.index,
        locationmode='country names',
        z=rt3.values,
        #text=rt3.index,
        colorscale='Portland',
        marker=dict(line=dict(width=0.7)),
        colorbar=dict(autotick=False, tickprefix='', title=' Countries with Conflicts'),
)]
layout = dict(title = 'Conflicts reported in Africa in 1999',
             geo = dict(
            showframe = False,
            showcoastlines = False,
            scope = 'africa',
            showcountries = True,
            projection = dict(
                type = 'Mercator'
            )
        ),)
fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False)
rt4 = df_99.groupby('country').sum()['fatalities'].sort_values(ascending=False)

data = [dict(
        type='choropleth',
        locations= rt4.index,
        locationmode='country names',
        z=rt4.values,
        text=rt4.index,
        colorscale='Reds',
        marker=dict(line=dict(width=0.7)),
        colorbar=dict(autotick=False, tickprefix='', title=' Fatalities Scale'),
)]
layout = dict(title = 'Fatalities reported in Africa in 1999',
             geo = dict(
            showframe = False,
            showcoastlines = False,
            projection = dict(
                type = 'Mercator'
            )
        ),)
fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False)
df_99.event_type = df_99.event_type.apply(lambda x: x.lower())
event_data_99 = df_99.groupby('event_type').sum().reset_index()
# Create a new columns that count the numbers of counflicts 
d_99 = dict(df_99.event_type.value_counts())
event_data_99['conflicts'] = event_data_99['event_type'].map(d_99)
# Sort the data by Fatalities
event_data_99.sort_values(by='fatalities', ascending=False,inplace=True)
#reduce the data to only 8 event type
event_data = event_data.head(8)
f, ax = plt.subplots(1,1,figsize = (10,10))

ax = event_data_99[['fatalities', 'conflicts']].plot(kind='barh',ax=ax,width=0.8,
              color=['dodgerblue', 'slategray'], fontsize=13);

ax.set_title("Causes of Conflicts in 1999",fontsize=20)

ax.set_ylabel("Event Type", fontsize=15)

ax.set_yticklabels(event_data_99.event_type.values)

# set individual bar lables using above list
for i in ax.patches:
    # get_width pulls left or right; get_y pushes up or down
    ax.text(i.get_width()+750, i.get_y()+.25, \
            str(round((i.get_width())/1000,2))+'k', fontsize=12, color='black')

# invert for largest on top 
ax.invert_yaxis()
sns.despine(bottom=True)
x_axis = ax.axes.get_xaxis().set_visible(False) # turn off the y axis label
plt.legend(loc=(1.0,0.98),fontsize=13,ncol=2)
plt.show()

source_99 = df_99.source.value_counts().head(10)
f, ax = plt.subplots(1,1,figsize = (10,10))

ax.barh(source_99.index,source_99.values,color='DarkCyan',edgecolor='black')
ax.set_title("Top 10 News Source in 1999 ",
fontsize=15)

ax.set_ylabel("News Source", fontsize=15)
ax.tick_params(length=3, width=1, colors='black',labelsize='large',axis='y')
ax.set_yticklabels(source_99.index)

# set individual bar lables using above list
for i in ax.patches:
    # get_width pulls left or right; get_y pushes up or down
    ax.text(i.get_width()+4, i.get_y()+ 0.51, \
            str(round((i.get_width()),2)), fontsize=12, color='black')

# invert for largest on top 
ax.invert_yaxis()
sns.despine(bottom=True)
x_axis = ax.axes.get_xaxis().set_visible(False) # turn off the y axis label
plt.show()
note_data_99 = df_99.dropna(subset=['notes'])

note_data_99['notes'] = note_data_99['notes'].apply(lambda x: " ".join(x.lower() for x in x.split()))

note_data_99['notes'] = note_data_99['notes'].map(clean_text)
wc_99 = WordCloud(max_font_size=50, width=600, height=300, background_color='black',colormap='Reds')
wc_99.generate(' '.join(note_data_99['notes'].values))

plt.figure(figsize=(15,8))
plt.imshow(wc_99)
plt.title("WordCloud of News Angencies Notes in 1999", fontsize=35)
plt.axis("off")
plt.show() 
cat = ['Jan', 'Feb', 'Mar', 'Apr','May','Jun', 'Jul', 'Aug','Sep', 'Oct', 'Nov', 'Dec']
#Sort the Number of Conflicts by Month 
df_99_sort = df_99.groupby('month').count()['event_date']
df_99_sort.index = pd.CategoricalIndex(df_99_sort.index,categories=cat ,sorted=True)
df_99_sort = df_99_sort.sort_index()

#Sort the Number of Fatalities by Month
df_99_sort_2 = df_99.groupby('month').sum()['fatalities']
df_99_sort_2.index = pd.CategoricalIndex(df_99_sort_2.index,categories= cat,sorted=True)
df_99_sort_2 = df_99_sort_2.sort_index()
x=df_99_sort.index
y= df_99_sort.values
#Second Graph
x2=df_99_sort_2.index
y2= df_99_sort_2.values

trace1 = go.Bar(
    x=x,
    y=y,
    marker=dict(
    color='gray'
    ),
    orientation = 'v',
    name = 'Conflicts',
)
trace2 = go.Scatter(
    x=x2,
    y=y2,
    mode = 'lines+markers',
    marker=dict(
    color='red'),
    name = 'Fatalities',
    yaxis='y2'
)
data = [trace1, trace2]
layout = go.Layout(
    
    xaxis=dict(
        title = 'Months',
        linecolor='grey',
        linewidth=2,
        autotick=False,
        ticks='outside',
    ),
    
    yaxis=dict(
        title='Numbers of Conflicts',
        titlefont=dict(
            color='MidnightBlue')
    ),
    yaxis2=dict(
        title='Numbers of Fatalities',
        titlefont=dict(
            color='red'
        ),
        
    overlaying='y',
    side='right',
    ),
    barmode='group',
    bargap=0.15,
    bargroupgap=0.1,
    
    legend=dict(
        x=.8,
        y=1,
    ),
    title = 'Conflicts & Fatalities by Month in 1999'
    )

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
data_99 = df_99.groupby('country')['fatalities'].sum().sort_values(ascending=False).head(10)
x=data_99.index
y= data_99.values

trace1 = go.Bar(
    x=x,
    y=y,
    text = y,
    textposition = 'auto',
    textfont = {'size':12,'color':'black'},
    marker=dict(
    color='DarkOrange'),
    opacity=0.8,
    orientation ='v',
)

data = [trace1]

layout = go.Layout (
    
    xaxis = dict (
    title = 'Countries Name'),
    
    title = 'Top 10 Countries with Highest Fatalities in 1991'
)
fig = go.Figure (data=data, layout = layout)
py.iplot(fig)
# Eritrea Dataset
df_99_er = df_99[df_99.country =='Eritrea']
cat = ['Jan', 'Feb', 'Mar', 'Apr','May','Jun', 'Jul', 'Aug','Sep', 'Oct', 'Nov', 'Dec']
#Sort the Number of Conflicts by Month 
df_99_sort_er = df_99_er.groupby('month').count()['event_date']
df_99_sort_er.index = pd.CategoricalIndex(df_99_sort_er.index,categories=cat ,sorted=True)
df_99_sort_er = df_99_sort_er.sort_index()

#Sort the Number of Fatalities by Month
df_99_sort_2_er = df_99_er.groupby('month').sum()['fatalities']
df_99_sort_2_er.index = pd.CategoricalIndex(df_99_sort_2_er.index,categories= cat,sorted=True)
df_99_sort_2_er = df_99_sort_2_er.sort_index()
x=df_99_sort_er.index
y= df_99_sort_er.values
#Second Graph
x2=df_99_sort_2_er.index
y2= df_99_sort_2_er.values

trace1 = go.Bar(
    x=x,
    y=y,
    marker=dict(
    color='gray'
    ),
    orientation = 'v',
    name = 'Conflicts',
)
trace2 = go.Scatter(
    x=x2,
    y=y2,
    mode = 'lines+markers',
    marker=dict(
    color='red'),
    name = 'Fatalities',
    yaxis='y2'
)
data = [trace1, trace2]
layout = go.Layout(
    
    xaxis=dict(
        title = 'Months',
        linecolor='grey',
        linewidth=2,
        autotick=False,
        ticks='outside',
    ),
    
    yaxis=dict(
        title='Numbers of Conflicts',
        titlefont=dict(
            color='MidnightBlue')
    ),
    yaxis2=dict(
        title='Numbers of Fatalities',
        titlefont=dict(
            color='red'
        ),
        
    overlaying='y',
    side='right',
    ),
    barmode='group',
    bargap=0.15,
    bargroupgap=0.1,
    
    legend=dict(
        x=.8,
        y=1,
    ),
    title = 'Eritrea Conflits vs. Fatalities in 1991'
    )

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
w1= df_99_er.pivot_table(index='actor1',columns='actor2',values='fatalities',aggfunc='sum')

f, ax = plt.subplots(figsize=(20,7))

ax = sns.heatmap(w1, cmap='Reds',linecolor='black',linewidths=0.2,annot=True,fmt='.1f',annot_kws={'fontsize':15},square=False)

ax.tick_params(axis = 'y',length=3, width=1, colors='black',labelsize=13)
ax.tick_params(axis = 'x',length=3, width=1, colors='black',labelsize=13)
kwargs= {'fontsize':15, 'color':'black'}
ax.set_xlabel('Actor2',**kwargs)
ax.set_ylabel('Actor1',**kwargs)
ax.set_title('Fatalities btw Actors in Eritrea Conflicts',fontsize=20, color='black')
sns.despine(top=False,right = False)
plt.show()
week_month_1= df_99_er.pivot_table(index='month',columns='day of week',values='fatalities',aggfunc='sum')
#Sorting the month Chronologically
week_month_1.index = pd.CategoricalIndex(week_month_1.index,
        categories= ['Jan', 'Feb', 'Mar', 'Apr','May','Jun', 'Jul', 'Aug','Sep', 'Oct', 'Nov', 'Dec'],sorted=True)
week_month_1 = week_month_1.sort_index()
f, ax = plt.subplots(figsize=(15,10))

ax = sns.heatmap(week_month_1, cmap='gist_heat_r',linecolor='black',linewidths=0.4,annot=True,fmt='.1f',annot_kws={'fontsize':13})
ax.tick_params(axis = 'y',length=3, width=1, colors='black',labelsize=13)
ax.tick_params(axis = 'x',length=3, width=1, colors='black',labelsize=13)
kwargs= {'fontsize':13, 'color':'black'}
ax.set_xlabel('Day of Week',**kwargs)
ax.set_ylabel('Months',**kwargs)
ax.set_title('Months vs. Weekday Fatalities Distribution in Eritrea',fontsize=20, color = 'black')
sns.despine(top=False,right = False)
plt.show()
note_data_99_er = df_99_er.dropna(subset=['notes'])

note_data_99_er['notes'] = note_data_99_er['notes'].apply(lambda x: " ".join(x.lower() for x in x.split()))

note_data_99_er['notes'] = note_data_99_er['notes'].map(clean_text)

wc_99_er = WordCloud(max_font_size=50, width=600, height=300,colormap='Oranges')
wc_99_er.generate(' '.join(note_data_99_er['notes'].values))

plt.figure(figsize=(15,8))
plt.imshow(wc_99_er)
plt.title("WordCloud of News Angencies Notes of Eritrea in 1999", fontsize=30)
plt.axis("off")
plt.show() 
#Angola dataset in 1991
df_99_ang = df_99[df_99.country =='Angola']
w_2 = df_99_ang.groupby(['actor1','actor2']).sum()['fatalities'].reset_index().sort_values('fatalities',ascending=False).head(6)
#Cleaning up long name by replacing with abbreviation
w_2.actor2 = w_2.actor2.replace({'UNITA: National Union for the Total Independence of Angola':'UNITA'})
w_2.actor1 = w_2.actor1.replace({'UNITA: National Union for the Total Independence of Angola':'UNITA',
                                'Military Forces of Democratic Republic of Congo (1997-2001)':'Military Forces of DRC'})
w_ang= w_2.pivot_table(index='actor1',columns='actor2',values='fatalities',aggfunc='sum')

f, ax = plt.subplots(figsize=(15,7))

ax = sns.heatmap(w_ang, cmap='Reds',linecolor='black',linewidths=0.2,annot=True,fmt='.1f',annot_kws={'fontsize':15})

ax.tick_params(axis = 'y',length=3, width=1, colors='black',labelsize=13)
ax.tick_params(axis = 'x',length=3, width=1, colors='black',labelsize=13)
kwargs= {'fontsize':15, 'color':'black'}
ax.set_xlabel('Actor2',**kwargs)
ax.set_ylabel('Actor1',**kwargs)
ax.set_title('Fatalities btw Actors in Angola conflicts',fontsize=20, color='black')
sns.despine(top=False,right = False)
plt.show()
cat = ['Jan', 'Feb', 'Mar', 'Apr','May','Jun', 'Jul', 'Aug','Sep', 'Oct', 'Nov', 'Dec']
#Sort the Number of Conflicts by Month 
df_99_sort_ang = df_99_ang.groupby('month').count()['event_date']
df_99_sort_ang.index = pd.CategoricalIndex(df_99_sort_ang.index,categories=cat ,sorted=True)
df_99_sort_ang = df_99_sort_ang.sort_index()

#Sort the Number of Fatalities by Month
df_99_sort_2_ang = df_99_ang.groupby('month').sum()['fatalities']
df_99_sort_2_ang.index = pd.CategoricalIndex(df_99_sort_2_ang.index,categories= cat,sorted=True)
df_99_sort_2_ang = df_99_sort_2_ang.sort_index()
x=df_99_sort_ang.index
y= df_99_sort_ang.values
#Second Graph
x2=df_99_sort_2_ang.index
y2= df_99_sort_2_ang.values

trace1 = go.Bar(
    x=x,
    y=y,
    marker=dict(
    color='gray'
    ),
    orientation = 'v',
    name = 'Conflicts',
)
trace2 = go.Scatter(
    x=x2,
    y=y2,
    mode = 'lines+markers',
    marker=dict(
    color='red'),
    name = 'Fatalities',
    yaxis='y2'
)
data = [trace1, trace2]
layout = go.Layout(
    
    xaxis=dict(
        title = 'Months',
        linecolor='grey',
        linewidth=2,
        autotick=False,
        ticks='outside',
    ),
    
    yaxis=dict(
        title='Numbers of Conflicts',
        titlefont=dict(
            color='MidnightBlue')
    ),
    yaxis2=dict(
        title='Numbers of Fatalities',
        titlefont=dict(
            color='red'
        ),
        
    overlaying='y',
    side='right',
    ),
    barmode='group',
    bargap=0.15,
    bargroupgap=0.1,
    
    legend=dict(
        x=.8,
        y=1,
    ),
    title = 'Angola Conflits vs. Fatalities in 1999'
    )

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
week_month= df_99_ang.pivot_table(index='month',columns='day of week',values='fatalities',aggfunc='sum')
#Sorting the month Chronologically
week_month.index = pd.CategoricalIndex(week_month.index,
        categories= ['Jan', 'Feb', 'Mar', 'Apr','May','Jun', 'Jul', 'Aug','Sep', 'Oct', 'Nov', 'Dec'],sorted=True)
week_month = week_month.sort_index()
f, ax = plt.subplots(figsize=(15,10))

ax = sns.heatmap(week_month, cmap='gist_heat_r',linecolor='black',linewidths=0.4,annot=True,fmt='d',annot_kws={'fontsize':13})
ax.tick_params(axis = 'y',length=3, width=1, colors='black',labelsize=13)
ax.tick_params(axis = 'x',length=3, width=1, colors='black',labelsize=13)
kwargs= {'fontsize':13, 'color':'black'}
ax.set_xlabel('Day of Week',**kwargs)
ax.set_ylabel('Months',**kwargs)
ax.set_title('Months vs. Weekday Fatalities Distribution in Angola',fontsize=20, color = 'black')
sns.despine(top=False,right = False)
plt.show()
note_data_99_ang = df_99_ang.dropna(subset=['notes'])

note_data_99_ang['notes'] = note_data_99_ang['notes'].apply(lambda x: " ".join(x.lower() for x in x.split()))

note_data_99_ang['notes'] = note_data_99_ang['notes'].map(clean_text)
wc_99_ang = WordCloud(max_font_size=50, width=600, height=300, background_color='black',colormap='Reds')
wc_99_ang.generate(' '.join(note_data_99_ang['notes'].values))

plt.figure(figsize=(15,8))
plt.imshow(wc_99_ang)
plt.title("WordCloud of News Angencies Notes of Angola in 1999", fontsize=30)
plt.axis("off")
plt.show() 
