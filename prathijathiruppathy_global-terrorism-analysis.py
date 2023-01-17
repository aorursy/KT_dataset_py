import math

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns



import plotly.express as px

import plotly.graph_objs as go



from matplotlib.ticker import FuncFormatter,ScalarFormatter

import matplotlib.animation as animation

from IPython.display import HTML



from plotly.offline import plot, iplot, init_notebook_mode

init_notebook_mode(connected=True)

%matplotlib inline

pd.options.mode.chained_assignment = None

sns.set(style="darkgrid")
gt = pd.read_csv('/kaggle/input/gtd/globalterrorismdb_0718dist.csv',encoding='ISO-8859-1',

                 usecols=['eventid', 'iyear', 'imonth', 'iday', 'region','region_txt', 'country', 

               'country_txt', 'provstate', 'city', 'latitude', 'longitude', 'extended', 'suicide', 

               'attacktype1', 'attacktype1_txt', 'targtype1', 'targtype1_txt','weaptype1','weaptype1_txt',

               'nkill','nwound','gname'])

                 
gt.rename(columns={'country':'country_code','country_txt':'country','region':'region_code','region_txt':'region_name',

                   'attacktype1':'attack_type_code','attacktype1_txt':'attack_type',

                   'targtype1':'target_type_code','targtype1_txt':'target_type','weaptype1':'weapon_type_code',

                   'weaptype1_txt':'weapon_type'},inplace = True)
#Creating date column

temp1 = gt[['iyear', 'imonth', 'iday']]

temp1.rename(columns={'iyear':'year','imonth':'month','iday':'day'},inplace = True)

gt['date']= pd.to_datetime(temp1[['year', 'month', 'day']],errors='coerce').dt.date.astype('datetime64')
gt.info()
#Filling missing values

gt['nkill'] = gt['nkill'].fillna(0).astype(np.int64)

gt['nwound'] = gt['nwound'].fillna(0).astype(np.int64)



#Adding new columns

gt['casualities'] = gt['nwound'] + gt['nkill']



#Altering large values

gt['weapon_type'].replace({"Vehicle (not to include vehicle-borne explosives, i.e., car or truck bombs)":"Bombless Vehicle"},inplace =True)
gt.sample(10)
#Function to obtain formats such as 1K,10K,1M for large numbers 

def count_format(num,pos):

    num = float('{:.3g}'.format(num))

    mag = 0

    while abs(num) >= 1000:

        mag += 1

        num /= 1000.0

    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M'][mag])





formatter = FuncFormatter(count_format)

year_group = gt.groupby('iyear',as_index = False)['eventid'].count()

plt.figure(figsize=(12,8))

plt.boxplot(year_group['eventid'],patch_artist = True)

plt.title('Total count of terrorist attacks over the years ')

plt.xticks([1],['Total count of terrorist attacks over the years '])

plt.ylabel('Count')

ax = plt.gca()

ax.yaxis.set_major_formatter(formatter)

plt.show()
print('Statistics on total count of terrorist attacks every year from',year_group['iyear'].min(), 'to', year_group['iyear'].max(),':')

print('\t Total: ',year_group['eventid'].sum())

print('\t Average: ',round(year_group['eventid'].mean()))

print('\t Maximum: ',year_group['eventid'].max())

print('\t Minimum: ',year_group['eventid'].min())

plt.figure(figsize=(12,8))

ax=sns.distplot(year_group['eventid'], 

             bins=20, color='blue',rug= True)

plt.xlabel('Count of terrorist attacks every year')

plt.ylabel('Probability density')

plt.xticks(range(0,20001,1000),rotation =70)

ax.xaxis.set_major_formatter(formatter)

plt.title('Probability density distribution of total count of terrorist attacks from {0} to {1}'.format(year_group['iyear'].min(),

                                                                                             year_group['iyear'].max()))

fig = go.Figure()

fig.add_trace(go.Scatter(x=year_group['iyear'], y=year_group['eventid'],

                    mode='lines+markers',

                    name='lines+markers'))

fig.update_layout(xaxis_rangeslider_visible=True)

fig.update_layout(xaxis = dict(title='Year',

        tickmode = 'array',

        tickvals = list(range(year_group['iyear'].min(),year_group['iyear'].max()+1,5))),

        yaxis=dict(title='Total attacks'),

        height=600,width=600,

        title="Spread of total count of terrorist attacks every year")



fig.data[0].hovertemplate = '<b>Year: %{x} </b><br>Attack count: %{y}<extra></extra>'



fig.show()
print('The highest number of terrorist attacks took place in the year',

      year_group.loc[year_group['eventid']==year_group['eventid'].max(),'iyear'].iloc[0])

     

print('The lowest number of terrorist attacks took place in the year',

      year_group.loc[year_group['eventid']==year_group['eventid'].min(),'iyear'].iloc[0])
#Dateframe for daywise terrorist attacks and their resulting casualities

date_cas = gt.groupby('date',as_index = False)['casualities'].sum()

date_group = gt.groupby('date',as_index = False)['eventid'].count()

result = pd.concat([date_group,date_cas], axis=1, sort=False)

cols = [x for x in range(result.shape[1])] 

cols.remove(2)

daywise_attacks_cas = result.iloc[:, cols]



def cas_category(x):

    if x == 0:

        return 'No casualities'

    elif 1 <=  x <= 100:

        return '1 to 100'

    elif 101 <= x <= 500:

        return '101 to 500'

    elif 501 <= x <= 999:

        return '501 to 999'

    elif  x >= 1000:

        return '1000 and above'





c_color={'No casualities':'rgb(60,179,113)','1 to 100':'rgb(93, 164, 214)',

         '101 to 500' :'rgb(238,130,238)','501 to 999':'rgb(255,215,0)',

         '1000 and above': 'rgb(255,99,71)'}

c_size= {'No casualities':10,'1 to 100' :20,'101 to 500': 30,'501 to 999':40,

         '1000 and above':50}



daywise_attacks_cas['cas_stats'] = daywise_attacks_cas['casualities'].map(cas_category)

daywise_attacks_cas['cas_color'] = daywise_attacks_cas['cas_stats'].map(c_color)

daywise_attacks_cas['cas_size'] = daywise_attacks_cas['cas_stats'].map(c_size)



# Dictionary with dataframes for each casualities groups

cas_stats_list = list(daywise_attacks_cas.cas_stats.unique())

cas_stats_data = {cas_stats:daywise_attacks_cas.query("cas_stats == '%s'" %cas_stats)

                              for cas_stats in cas_stats_list}



fig = go.Figure()

for cas_stats_name, cas_stats in cas_stats_data.items():

    fig.add_trace(go.Scattergl(

        x=cas_stats['date'], y=cas_stats['eventid'],

        name=cas_stats_name,

        marker_size=cas_stats['cas_size'], 

        mode='markers',

        customdata = cas_stats['casualities'],

        marker_color= cas_stats['cas_color'],

        hovertemplate =  '<b>%{x} </b><br>Number of attacks: %{y}<br>Casualities: %{customdata}<extra></extra>'))

    

fig.update_layout(height=600,width=1000,

                title="How terrorism and resulting casualities spread through days?",

                xaxis_range=[daywise_attacks_cas['date'].min(),daywise_attacks_cas['date'].max()],

                xaxis = dict(title='Timeline',

                rangeselector=dict(buttons=list([                    

                        dict(count=1,label="1M",step="month",stepmode="backward"),

                        dict(count=6,label="6M",step="month",stepmode="backward"),

                        dict(count=1,label="YTD",step="year",stepmode="todate"),

                        dict(count=1,label="1Y",step="year",stepmode="backward"),

                        dict(count=5,label="5Y",step="year",stepmode="backward"),

                        dict(count=10,label="10Y",step="year",stepmode="backward"),

                        dict(step="all",label='All')]))),

                yaxis=dict(title='Count of terroirst attacks',

                           range=[0,daywise_attacks_cas['eventid'].max()+30]),

                legend = dict(title= 'Casualities'),

                plot_bgcolor='rgb(243, 243, 243)',)                 

fig.show()
cg = gt.groupby('iyear',as_index =False).sum()

cas_group = cg[['iyear','casualities','nkill','nwound']]

cas_group_array = cas_group[['nkill','nwound']].values



sum_of_cas = pd.DataFrame(np.array([['Casualities: '+str(cas_group['casualities'].sum()),

                        cas_group['nkill'].sum(), cas_group['nwound'].sum()]]),    

                        columns=['Casualities','Dead', 'Wounded'])

temp = sum_of_cas.melt(id_vars="Casualities", value_vars=['Dead', 'Wounded'])



fig = go.Figure(go.Treemap(

    labels=temp['variable'],

    parents =temp['Casualities'],values= temp['value'],

    textinfo='label+text+value+percent parent',

    hoverinfo='label+text+value'))



fig.update_layout(height=275,title='People killed vs People wounded')

fig.show()
plt.figure(figsize=(12,8))

cas_group_columns = ['Number of people killed','Number of people wounded']

bp = plt.boxplot(cas_group_array,patch_artist = True,labels=cas_group_columns)

colors = ['r','b']

for i in range(len(bp['boxes'])):

    bp['boxes'][i].set(facecolor = colors[i])

plt.title('Count of casualities over the years ')

plt.ylabel('Count')

ax= plt.gca()

ax.yaxis.set_major_formatter(formatter)

plt.yticks(range(0,cas_group['nkill'].max()+10001,5000))

plt.show()
print('Statistics on people killed from the year',cas_group['iyear'].min(),

      'to', cas_group['iyear'].max(),'due to terrorism: ')

print('\t Total:', cas_group['nkill'].sum())

print('\t Average:' ,round(cas_group['nkill'].mean()))

print('\t Minimum:',cas_group['nkill'].min())

print('\t Maximum:',cas_group['nkill'].max())

print('')

print('Statistics on people wounded from the year',cas_group['iyear'].min(),

      'to', cas_group['iyear'].max(),'due to terrorism: ')

print('\t Total:',cas_group['nwound'].sum())

print('\t Average:',round(cas_group['nwound'].mean()))

print('\t Minimum:',cas_group['nwound'].min())

print('\t Maximum:',cas_group['nwound'].max())
fig = go.Figure()

fig.add_trace(go.Scatter(x=cas_group['iyear'], y=cas_group['nkill'],

                    mode='lines+markers',marker_color='rgba(255, 0, 0, 0.7)',

                    name='Number of people killed'))

fig.add_trace(go.Scatter(x=cas_group['iyear'], y=cas_group['nwound'],

                    mode='lines+markers',marker_color='rgba(0, 0, 255, 0.7)',

                    name='Number of people wounded'))

fig.update_layout(xaxis_rangeslider_visible=True)

fig.update_layout(xaxis = dict(title = 'Year',

        tickmode = 'array',

        tickvals = list(range(cas_group['iyear'].min(),cas_group['iyear'].max()+1,5))),

        height=600,width=800,

        title="Total count of casualities over the years",

        yaxis=dict(title='Total count'))



fig.data[0].hovertemplate = '<b>Year: %{x} </b><br>People killed: %{y}<extra></extra>'

fig.data[1].hovertemplate = '<b>Year: %{x} </b><br>People wounded: %{y}<extra></extra>'

fig.show()
def generate_tenth_power(max_value):

    return math.pow(10, len(str(max_value)));



def plot_count(col):

    plt.figure(figsize=(15, 12))



    ax = sns.countplot(y=col,

                  data=gt, order = gt[col].value_counts().index)

    total = len(gt[col])

    for p in ax.patches:

        percentage = '{:.2f}%'.format(p.get_width()/total*100)

        x= p.get_x()+p.get_width()

        y= p.get_y()+p.get_height()

        ax.annotate(percentage, (x, y))

    plt.title('Count and Percentage of different '+col.split('_')[0]+' types')

    ax.set_xscale('log')

    max_value = generate_tenth_power(gt[col].value_counts().max())

    ax.set_xticks([max_value*0.0005,max_value*0.001,max_value*0.005,max_value*0.01, 

                   max_value*0.05,max_value*0.1,max_value*0.5,max_value])

    ax.xaxis.set_major_formatter(formatter)

    plt.ylabel(col.split('_')[0].title()+' types')

    plt.show()
plot_count('attack_type')
plot_count('weapon_type')
plot_count('target_type')
plot_count('region_name')
def plot_double_category(col,a,b):

    plt.figure(figsize=(6, 6))

    ax = sns.countplot(y=col,data=gt,order=gt[col].value_counts().index)

    total = len(gt[col])

    for p in ax.patches:

        percentage = '{:.2f}%'.format(p.get_width()/total*100)

        x= p.get_x()+p.get_width()

        y= p.get_y()+p.get_height()

        ax.annotate(percentage, (x, y))

    plt.title('Count and Percentage of '+a.lower()+' vs '+b.lower())

    ax.set_xscale('log')

    max_value = generate_tenth_power(gt[col].value_counts().max())

    ax.set_xticks([max_value*0.0005,max_value*0.001,max_value*0.005,max_value*0.01, 

                   max_value*0.05,max_value*0.1,max_value*0.5,max_value])

    ax.xaxis.set_major_formatter(formatter)

    plt.yticks([0,1],[a,b])

    plt.ylabel('')

    plt.show()  
plot_double_category('extended','Attacks lasting one day','Extended attacks')
plot_double_category('suicide','Non suicidal attacks','Suicidal attacks')
wg = gt.groupby('weapon_type',as_index=False).count()

wg.sort_values('eventid',ascending=False,inplace=True)

wg.rename(columns={'eventid':'Total attacks'},inplace =True)

weapon_group= wg[['weapon_type','Total attacks']]

top_weapons = weapon_group.loc[weapon_group["Total attacks"]>10000]

top_weapons_list = list(top_weapons['weapon_type'])



plt.figure(figsize=(18, 12))

ax= sns.countplot(x='attack_type',hue='weapon_type',data=gt,hue_order=top_weapons_list)

plt.title('Count of top weapons used in different attack types')

plt.xticks(rotation = 60)

ax.set_yscale('log')

ax.yaxis.set_major_formatter(formatter)

plt.ylabel('Count of weapons used')

plt.xlabel('Attack types')

plt.legend(title='Top weapon types',loc='upper left',bbox_to_anchor=(1.0, 1.0))

plt.show()
def plot_tree(col,title):

    temp1 = gt.groupby(col,as_index = False).sum()[[col,'casualities']]

    temp1["node"] =  title.title()

    fig = px.treemap(temp1, path=['node', col], values='casualities',template="seaborn",

                     title="Different "+title+' and their resulting casualities')

    fig.data[0].texttemplate = '<b>%{label} </b> <br>Casualities: %{value}'

    fig.data[0].hovertemplate = '<b>%{label} </b> <br>Casualities: %{value}'

    fig.show()
plot_tree('attack_type','attack types')
plot_tree('weapon_type','weapon types')
plot_tree('target_type','target types')
plot_tree('gname','terrorist groups')
plot_tree('region_name','regions of the world')
#Functions to generate custom groups based on total casualites or attack count



def create_yearwise_cas_group(col):

    cas_sum = gt.groupby(['iyear',col],as_index=False).sum()

    temp = cas_sum[['iyear',col,'casualities']]

    cas_df= temp.loc[(temp.casualities > 0)]

    cas_df.rename(columns={'iyear':'year'},inplace =True)

    return cas_df



def create_yearwise_attacks_group(col):

    att_sum = gt.groupby(['iyear',col],as_index=False).count()

    temp = att_sum[['iyear',col,'eventid']]

    att_df = temp.loc[(temp.eventid > 0)]

    att_df.rename(columns={'iyear':'Year','eventid':'Number of attacks'},inplace =True)

    return att_df
def plot_heatmap(col):

    temp_df = create_yearwise_cas_group(col)

    fig = go.Figure(data=go.Heatmap(

        z=temp_df['casualities'],

        x=temp_df['year'],

        y=temp_df[col],

        zmin=10,

        zmax=10000,

        colorscale='burg', 

        colorbar=dict(

            title='Casualities',

            tickmode="array",

            tickvals = [100,1000,2500,5000,7500,10000],

            ticktext=[100,1000,2500,5000,7500,'10000 & above'],

            ticks='outside' ),

        hoverongaps = False,

        text=temp_df['casualities'],

        hovertemplate = '<b>%{y} </b><br>Year: %{x} <br>Casualities: %{z}<extra></extra>'))        

    fig.update_layout(

    yaxis = dict(dtick = 1,title=str(col.split('_')[0]).title()+' types',

                 categoryarray=temp_df[col].sort_values(ascending=False)),

    xaxis= dict(title='Year'),

    height=500,

    title='Timeline of how casualities spread by different '+str(col.split('_')[0])+'s')

    fig.show()
plot_heatmap('attack_type')
plot_heatmap('weapon_type')
plot_heatmap('target_type')
plot_heatmap('region_name')
world_cas = create_yearwise_cas_group('country')

world_attacks =  create_yearwise_attacks_group('country')



#Dataframe containing terrorist attacks count and their resulting casualities

result = pd.concat([world_attacks,world_cas], axis=1, sort=False)

cols = [x for x in range(result.shape[1])] 

cols.remove(3)

cols.remove(4)



world_attacks_cas = result.iloc[:, cols]

world_attacks_cas['casualities'] = world_attacks_cas['casualities'].fillna(0).astype('int64')

world_attacks_cas = world_attacks_cas.rename(columns={'country':'Country'})



fig = px.choropleth(world_attacks_cas, color='Number of attacks', locations='Country', locationmode='country names', 

                    color_continuous_scale=px.colors.sequential.amp, 

                    title='Terrorism and their resulting casualities around the world',

                    custom_data=['casualities'],

                    animation_frame=world_attacks_cas["Year"])



fig.data[0].hovertemplate = str(fig.data[0].hovertemplate)+'<br><b>Casualities= %{customdata}<b>'



#updating year frames to include casualities count

for i in range(len(fig.frames)):

    fig.frames[i].data[0].hovertemplate =str(fig.frames[i].data[0].hovertemplate)+'<br><b>Casualities= %{customdata}<b>'

fig.show()
usa_event_count = gt.groupby(['iyear','country','provstate'],as_index=False).count()

temp= usa_event_count.loc[(usa_event_count.country=='United States')&(usa_event_count.eventid > 0)]

usa_event_group = temp[['iyear','country','provstate','eventid']]

usa_cas_sum = gt.groupby(['iyear','country','provstate'],as_index=False).sum()

temp= usa_cas_sum.loc[(usa_cas_sum.country=='United States')&(usa_cas_sum.casualities > 0)]

usa_cas_group = temp[['iyear','country','provstate','casualities']]



#Dataframe containing terrorist attacks count and their resulting casualities

result = pd.concat([usa_event_group, usa_cas_group], axis=1, sort=False)

cols = [x for x in range(result.shape[1])] 

cols.remove(4)

cols.remove(5)

cols.remove(6)

us_states_attacks_cas = result.iloc[:, cols]

us_states_attacks_cas['casualities'] = us_states_attacks_cas['casualities'].fillna(0).astype('int64')

us_states_attacks_cas = us_states_attacks_cas.rename(columns={'iyear':'Year','eventid':'Number of attacks'})



us_code = {'Alabama': 'AL', 'Alaska': 'AK', 'American Samoa': 'AS','Arizona': 'AZ', 'Arkansas': 'AR', 

    'California': 'CA','Colorado': 'CO','Connecticut': 'CT','Delaware': 'DE', 'District of Columbia': 'DC', 

    'Florida': 'FL', 'Georgia': 'GA', 'Guam': 'GU', 'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL',

    'Indiana': 'IN','Iowa': 'IA','Kansas': 'KS','Kentucky': 'KY','Louisiana': 'LA','Maine': 'ME',

    'Maryland': 'MD','Massachusetts': 'MA','Michigan': 'MI','Minnesota': 'MN','Mississippi': 'MS',

    'Missouri': 'MO','Montana': 'MT','Nebraska': 'NE','Nevada': 'NV','New Hampshire': 'NH', 'New Jersey': 'NJ',

    'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Northern Mariana Islands':'MP',

    'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Puerto Rico': 'PR',

    'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX',

    'Utah': 'UT', 'Vermont': 'VT', 'Virgin Islands': 'VI', 'Virginia': 'VA', 'Washington': 'WA',

    'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'}





us_states_attacks_cas['State'] = us_states_attacks_cas['provstate'].map(us_code)



fig = px.choropleth(us_states_attacks_cas, color='Number of attacks', locations='State', locationmode="USA-states", 

                    scope="usa", 

                    color_continuous_scale=px.colors.sequential.amp, 

                    title='Terrorism and their resulting casualities in USA',

                    custom_data=['casualities'],

                    animation_frame=us_states_attacks_cas["Year"])



fig.data[0].hovertemplate = str(fig.data[0].hovertemplate)+'<br><b>Casualities= %{customdata}<b>'



#updating year frames to include casualities count

for i in range(len(fig.frames)):

    fig.frames[i].data[0].hovertemplate =str(fig.frames[i].data[0].hovertemplate)+'<br><b>Casualities= %{customdata}<b>'

fig.show()
ycr_group = gt.groupby(['iyear','region_name','country'],as_index= False).count() 

ycr_events= ycr_group[['iyear','region_name','country','eventid']]

all_regions = list(ycr_events['region_name'].unique())

color_code = ['#C39953','#A17A74','#6D9BC3','#CD607E','#6EAEA1','#E97451','#FC80A5','#C9A0DC',

                '#76D7EA','#FFCBA4','#FCD667','#29AB87']

colors = dict(zip(all_regions, color_code))

group_cr = ycr_events.set_index('country')['region_name'].to_dict()



fig, ax = plt.subplots(figsize=(16, 12))



def plot_racing_bars(current_year):

    year_df = ycr_events.loc[ycr_events['iyear']==current_year].sort_values(by='eventid').tail(10)

    ax.clear()

    ax.barh(year_df['country'], year_df['eventid'], color=[colors[group_cr[x]] for x in year_df['country']])

    dx = year_df['eventid'].max() / 200

    for i, (value, name) in enumerate(zip(year_df['eventid'], year_df['country'])):

        ax.text(value-dx, i,     name,           size=14, weight=600, ha='right', va='bottom')

        ax.text(value-dx, i-.25, group_cr[name], size=10, color='#444444', ha='right', va='baseline')

        ax.text(value+dx, i,     value,  size=14, ha='left',  va='center')

    ax.text(1, 0.4, current_year, transform=ax.transAxes, color='#777777', size=46, ha='right', weight=800)

    ax.text(0, 1.06, 'Total terrorist attacks', transform=ax.transAxes, size=12, color='#777777')    

    ax.set_xticks(range(0,(year_df['eventid'].max()+(int(year_df['eventid'].max()/20))),int(year_df['eventid'].max()/20)))

    ax.xaxis.set_ticks_position('top')

    ax.tick_params(axis='x', colors='#777777', labelsize=12)

    ax.set_yticks([])   

    ax.margins(0, 0.01)

    ax.grid(which='major', axis='x', linestyle='-')

    ax.set_axisbelow(True)

    ax.text(0, 1.15, 'Top 10 countries affected by terrorism every year from '

            + str(ycr_events['iyear'].min())+' to '+ str(ycr_events['iyear'].max()),

            transform=ax.transAxes, size=24, weight=600, ha='left', va='top')

    plt.box(False)

    plt.legend()

    plt.close()



animator = animation.FuncAnimation(fig, plot_racing_bars, frames=ycr_events['iyear'].unique(),

                                   save_count=ycr_events['iyear'].nunique())

HTML(animator.to_jshtml())