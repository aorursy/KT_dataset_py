import pandas as pd
import numpy as np
import numbers
import plotly
from plotly.offline import init_notebook_mode, iplot
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
import folium 
# from folium import plugins

init_notebook_mode(connected=True)
gun_violence_df = pd.read_csv('../input/gun-violence-data_01-2013_03-2018.csv') 
gun_violence_df.head(3)
gun_violence_df.describe() ##describes only numeric data
# Function to describe more information for all the attributes
def brief(data):
    
    df = data.copy()
    
    print("This dataset has {} Rows {} Attributes".format(df.shape[0],df.shape[1]), end='')
    print('\n')
    
    real_valued = {}
    symbolics = {}
    
    
    for i,col in enumerate(df.columns, 1):
        Missing = len(df[col]) - df[col].count()
        
        counter = 0
        for val in df[col].dropna():
            if isinstance(val, numbers.Number):
                    counter += 1
        
        if counter != len(df[col].dropna()):
            arity = len(df[col].dropna().unique())
            symbolics[i] = [i, col, Missing, arity]  
        else:
            Mean, Median, Sdev, Min, Max = df[col].mean(), df[col].median(), df[col].std(), df[col].min(), df[col].max()
            real_valued[i] =  [i, col, Missing, Mean, Median, Sdev, Min, Max]
            
    
    #Create array containing list of real valued
    real_valued_array = [real_valued[keys] for keys in real_valued.keys()]
    real_valued_transformed = np.array(real_valued_array).T
    
    symbolic_array = [symbolics[keys] for keys in symbolics.keys()]
    symbolic_transformed = np.array(symbolic_array).T
    
    # return symbolic_transformed
    real_cols = ['Attribute_ID', 'Attribute_Name', 'Missing', 'Mean', 'Median', 'Sdev', 'Min', 'Max']
    sym_cols = ['Attribute_ID', 'Attribute_Name', 'Missing','arity']
    
    
   
    index = range(1, len(real_valued.keys())+1)
    real_val_df = pd.DataFrame(data={unit[0]:unit[1] for unit in zip(real_cols, real_valued_transformed)}, index = index, columns=real_cols)
    

    index_sym = range(1, len(symbolics.keys())+1)
    sym_val_df = pd.DataFrame(data={unit[0]:unit[1] for unit in zip(sym_cols, symbolic_transformed)}, index = index_sym, columns = sym_cols)
    
    text = ("real valued attributes" + "\n" + "---------------------" 
            + "\n" + str(real_val_df) + "\n"  + "non-real valued attributes"  
            + "\n" + "-------------------" + "\n" + str(sym_val_df))
        
    return text

%time
print(brief(gun_violence_df))
gun_violence_df.info()
# added important missing data point found in the description on Kaggle
missing =  ['sban_1', '2017-10-01', 'Nevada', 'Las Vegas', 'Mandalay Bay 3950 Blvd S', 59, 489, 'https://en.wikipedia.org/wiki/2017_Las_Vegas_shooting', 'https://en.wikipedia.org/wiki/2017_Las_Vegas_shooting', '-', '-', '-', '-', '-', '36.095', 'Hotel', 
            '-115.171667', 47, 'Route 91 Harvest Festiva; concert, open fire from 32nd floor. 47 guns seized; TOTAL:59 kill, 489 inj, number shot TBD,girlfriend Marilou Danley POI', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-']
gun_violence_df.loc[len(gun_violence_df)] = missing

print(gun_violence_df.shape)
drop_columns = gun_violence_df.columns[gun_violence_df.apply(lambda col: col.isnull().sum() >= (0.5 * len(gun_violence_df)))]
gun_violence_filtered = gun_violence_df.drop(drop_columns, axis=1)
print(gun_violence_filtered.shape)
print('Dropped Columns:', list(drop_columns))
gun_violence_filtered['date'] = pd.to_datetime(gun_violence_filtered['date'])
gun_violence_filtered = gun_violence_filtered.assign(year = gun_violence_filtered['date'].map(lambda dates: dates.year))
gun_violence_filtered = gun_violence_filtered.assign(month = gun_violence_filtered['date'].map(lambda dates: dates.month))
gun_violence_filtered = gun_violence_filtered.assign(day = gun_violence_filtered['date'].map(lambda dates: dates.weekday()))

y_yrs = gun_violence_filtered.groupby('year')['incident_id'].count().values
x_yrs = gun_violence_filtered.groupby('year')['incident_id'].count().index.values

y_months = gun_violence_filtered.\
            groupby(by=['year','month']).\
            agg('count').\
            groupby('month')['incident_id'].\
            mean().\
            values

x_months = ['Jan','Feb','Mar','Apr','May','June','July','Aug','Sep','Oct','Nov','Dec']

y_days = gun_violence_filtered.\
            groupby(['year','day']).\
            agg('count').\
            groupby('day')['incident_id'].\
            mean().\
            values

x_days = ['Mon','Tues','Wed','Thurs','Fri','Sat','Sun']


trace1 = go.Bar(
    x=x_yrs,
    y=y_yrs
)
trace2 = go.Bar(
    x=x_months,
    y=y_months,
    xaxis='x2',
    yaxis='y2'
)
trace3 = go.Bar(
    x=x_days,
    y=y_days,
    xaxis='x3',
    yaxis='y3'
)

data = [trace1, trace2, trace3]
fig = plotly.tools.make_subplots(rows=3, cols=1, specs = [[{}], [{}],[{}]],vertical_spacing = 0.25, subplot_titles=('Number of Incidents per Year', 
                                                                 'Average Number of Incidents per Month over Years',
                                                                 'Average Number of Incidents per Day over Years'))

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 2, 1)
fig.append_trace(trace3, 3, 1)

fig['layout']['xaxis1'].update(title='Years')
fig['layout']['xaxis2'].update(title='Months')
fig['layout']['xaxis3'].update(title='Days')


fig['layout']['yaxis1'].update(title='Count')
fig['layout']['yaxis2'].update(title='Avg. Frequency')
fig['layout']['yaxis3'].update(title='Avg. Frequency')


fig['layout'].update(showlegend=False, height=800, width=800, title='Incidents Over Time')
iplot(fig)
n_killed = gun_violence_filtered.\
                groupby('date').\
                sum()['n_killed'].values

n_injured = gun_violence_filtered.\
                groupby('date').\
                sum()['n_injured'].values

dates = gun_violence_filtered.\
                groupby('date').\
                count().\
                index

trace1 = go.Scatter(
    x = dates,
    y = n_killed,
    name = 'Number Killed',
    line = dict(
        dash = 'dot'
    )
)

trace2 = go.Scatter(
    x = dates,
    y = n_injured,
    name = 'Number Injured',
    line = dict(
        dash = 'dot'
    )
)

data = [trace1, trace2]

layout = dict(height=400,
              width=1000,
              title = 'Number of Total Incidents',
              xaxis = dict(title = 'Time'),
              yaxis = dict(title = 'Count'),
              )

fig = dict(data = data, layout=layout)
iplot(fig)

n_killed_2017 = gun_violence_filtered[gun_violence_filtered.loc[:,'year'] == 2017].\
                    groupby('date').\
                    sum()['n_killed'].values

n_injured_2017 = gun_violence_filtered[gun_violence_filtered.loc[:,'year'] == 2017].\
                    groupby('date').\
                    sum()['n_injured'].values

dates_2017 = gun_violence_filtered[gun_violence_filtered.loc[:,'year'] == 2017].\
                    groupby('date').\
                    count().\
                    index



trace1 = go.Scatter(
    x = dates_2017,
    y = n_killed_2017,
    name = 'Number Killed',
    line = dict(
        dash = 'dot'
    )
)

trace2 = go.Scatter(
    x = dates_2017,
    y = n_injured_2017,
    name = 'Number Injured',
    line = dict(
        dash = 'dot'
    )
)

data = [trace1, trace2]

layout = dict(height=400,
              width=1000,
              title = 'Number of Incidents in 2017',
              xaxis = dict(title = 'Time'),
              yaxis = dict(title = 'Count'),
              )

fig = dict(data = data, layout=layout)
iplot(fig)
state = gun_violence_filtered.groupby('state')
state_incidents = state.count().sort_values(by='incident_id',ascending=False)['incident_id']
state_killed = state.sum()['n_killed']
state_injured = state.sum()['n_injured']

city = gun_violence_filtered.groupby('city_or_county')
city_incidents= city.count().sort_values(by='incident_id',ascending=False)['incident_id'].head(20)



trace = go.Bar(
    x = state_incidents.index,
    y = state_incidents,
)

layout = dict(height=400,
              width=1000,
              title =  'Top States with Highest Number of Gun Violence Incidents',
              yaxis = dict(title = 'Number of Incidents'),
              )

data = [trace]

fig = dict(data = data, layout=layout)
iplot(fig)

trace = go.Bar(
    x = city_incidents.index[:20],
    y = city_incidents,
)
    
layout = dict(height=400,
              width=1000,
              title = 'Top Twenty Cities with Highest Number of Gun Violence Incidents',
              yaxis = dict(title = 'Number of Incidents'),
             )
    
data = [trace]

fig = dict(data = data, layout=layout)
iplot(fig)
trace1 = go.Bar(
    x = state_killed.index,
    y = state_killed,
    name = 'Number Killed'
)

trace2 = go.Bar(
    x = state_injured.index,
    y = state_injured,
    name = 'Number Injured'
)


data = [trace1, trace2]

layout = dict(height=400,
              width=1000,
              title = 'Number of People Injured/Killed Across States',
              yaxis = dict(title = 'Frequency'),
              )

fig = dict(data = data, layout=layout)
iplot(fig)
population_adjusted_data = pd.read_html('https://www.enchantedlearning.com/usa/states/population.shtml')[1] #population data
population_adjusted_data['State'] = population_adjusted_data['State'].apply(lambda val: val[3:].strip())
pop_adj_dic = {k:v for k,v in population_adjusted_data.to_dict('split')['data']}

state_incidents = pd.DataFrame(state_incidents)
state_incidents['population'] = state_incidents.index.map(lambda states : pop_adj_dic[states])
state_incidents['adj_incidents'] = (state_incidents['incident_id']/state_incidents['population']) * 100000

state_incidents = state_incidents.sort_values(by='adj_incidents',ascending=False)['adj_incidents']

trace = go.Bar(
    x = state_incidents.index,
    y = state_incidents,
)

layout = dict(height=400,
              width=1000,
              title =  'Top States with Highest Number of Gun Violence Incidents Adjusted For Population',
              yaxis = dict(title = 'Number of Incidents'),
              )

data = [trace]

fig = dict(data = data, layout=layout)
iplot(fig)
state_killed = pd.DataFrame(state_killed)
state_killed['population'] = state_killed.index.map(lambda states : pop_adj_dic[states])
state_killed['adj_killings'] = (state_killed['n_killed']/state_killed['population']) * 100000

state_injured = pd.DataFrame(state_injured)
state_injured['population'] = state_injured.index.map(lambda states : pop_adj_dic[states])
state_injured['adj_injuries'] = (state_injured['n_injured']/state_injured['population']) * 100000




trace1 = go.Bar(
    x = state_killed.index,
    y = state_killed['adj_killings'],
    name = 'Number Killed'
)

trace2 = go.Bar(
    x = state_injured.index,
    y = state_injured['adj_injuries'],
    name = 'Number Injured'
)


data = [trace1, trace2]

layout = dict(height=400,
              width=1000,
              title = 'Number of People Injured/Killed Across States Adjusted for Population',
              yaxis = dict(title = 'Frequency'),
              )

fig = dict(data = data, layout=layout)
iplot(fig)
gun_violence_filtered['total_damage'] = gun_violence_filtered['n_injured'] + gun_violence_filtered['n_killed']

gun_violence_filtered.\
        loc[:,['date','year','state', 'city_or_county', 'address', 'total_damage']].\
        sort_values(by='total_damage', ascending = False).\
        head(10)
df = gun_violence_filtered[gun_violence_filtered['total_damage'] >= 10][['latitude', 'longitude', 'total_damage', 'n_killed']].dropna()
maps = folium.Map([39.50, -98.35],  zoom_start=4, tiles='Stamen Toner')
markers = []
for idx, row in df.iterrows():
    total = row['total_damage'] * 0.30   
    folium.CircleMarker([float(row['latitude']), float(row['longitude'])], radius=float(total), color='#ef4f61', fill=True).add_to(maps)
maps
gun_violence_filtered[['participant_age','participant_type','participant_gender']].head(4)
# Convert string into dictionary
def StringToDic(S1):
    dic1 = {}
    list1 = str(S1).split('||')
    for i in list1:
        try:
            index = i.split('::')[0]
            value = i.split('::')[1]
            dic1[index] = value
        except:
            pass
        
    return dic1
        
    
# Apply the function above to each column, creating new column
gun_violence_filtered['participant_age_dic'] \
= gun_violence_filtered['participant_age'].apply(lambda x: StringToDic(x))

gun_violence_filtered['participant_type_dic'] \
= gun_violence_filtered['participant_type'].apply(lambda x: StringToDic(x)) 

gun_violence_filtered['participant_gender_dic'] \
= gun_violence_filtered['participant_gender'].apply(lambda x: StringToDic(x)) 


# Create another two new column, with new dictionary mapping type and age, type and gender
mappingCol1='participant_type_dic'
def MapThroughRow(df,mappingCol1,mappingCol2):
    newDic = {'Victim':[],'Suspect':[]}
    for rowName,row in df.iterrows():
        for keys,values in row[mappingCol1].items():
            if (keys in row[mappingCol2]) and (values =='Victim'):
                newDic['Victim'].append(row[mappingCol2][keys])
            elif (keys in row[mappingCol2]) and ('Suspect' in values):
                newDic['Suspect'].append(row[mappingCol2][keys])
                
    return newDic
%time
mappingCol2 = 'participant_age_dic'
mappingCol3 = 'participant_gender_dic'
df = gun_violence_filtered
MapTypeAge = MapThroughRow(df,mappingCol1,mappingCol2)
for key,values in MapTypeAge.items():
    MapTypeAge[key] = [int(i) for i in values]
    
MapTypeGender = MapThroughRow(df,mappingCol1,mappingCol3)

print(len(MapTypeAge['Victim']))
print(len(MapTypeAge['Suspect']))
print(len(MapTypeGender['Victim']))
print(len(MapTypeGender['Suspect']))
def countDic(L):
    dic = {}
    for i in L:
        if i not in dic:
            dic[i] = 1
        else:
            dic[i] += 1
    return dic
VicageList = list(countDic(MapTypeAge['Victim']).keys())
VicageCount = list(countDic(MapTypeAge['Victim']).values())
SusageList = list(countDic(MapTypeAge['Suspect']).keys())
SusageCount = list(countDic(MapTypeAge['Suspect']).values())
# For Victim
trace1 = go.Bar(
    x=VicageList,
    y=VicageCount,
    name='Age distribution of Victim',
    marker=dict(
        color='rgb(55, 83, 109)'
    )
)


data = [trace1]
layout = go.Layout(
    title='Age Distribution of Victims',
    xaxis=dict(
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)',
        ),
        range=[0,100]
    ),
    yaxis=dict(
        title='Count',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    legend=dict(
        x=0,
        y=1.0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    barmode='group',
    bargap=0.15,
    bargroupgap=0.1
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
# For Suspects
trace1 = go.Bar(
    x=SusageList,
    y=SusageCount,
    name='Age distribution of Suspects',
    marker=dict(
        color='maroon'
    )
)


data = [trace1]
layout = go.Layout(
    title='Age Distribution of Suspects',
    xaxis=dict(
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)',
        ),
        range=[0,100]
    ),
    yaxis=dict(
        title='Count',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    legend=dict(
        x=0,
        y=1.0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    barmode='group',
    bargap=0.15,
    bargroupgap=0.1
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
VicGenderList = list(countDic(MapTypeGender['Victim']).keys())
VicGenderCount = list(countDic(MapTypeGender['Victim']).values())
SusGenderList = list(countDic(MapTypeGender['Suspect']).keys())
SusGenderCount = list(countDic(MapTypeGender['Suspect']).values())
# It has a incorrectly recorded data here, but it is fine
print((VicGenderList,VicGenderCount))
print(sum(VicGenderCount))
print((SusGenderList,SusGenderCount))
print(sum(SusGenderCount))
import plotly.plotly as py
import plotly.graph_objs as go

fig = {
  "data": [
    {
      "values": [136394, 30630],
      "labels": [
        "Male",
        "Female"
      ],
      "domain": {"x": [0, .48]},
      "name": "Victims",
      "hoverinfo":"label+percent+name",
      "hole": .4,
      "type": "pie"
    },
    {
      "values": [167708,11746],
      "labels": [
         "Male",
        "Female"
      ],
      "text":["Suspects"],
      "textposition":"inside",
      "domain": {"x": [.52, 1]},
      "name": "Proportion of Gender for Victims and Suspects",
      "hoverinfo":"label+percent+name",
      "hole": .4,
      "type": "pie"
    }],
  "layout": {
        "title":"Proportion of Gender for Victims and Suspects",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Victims",
                "x": 0.20,
                "y": 0.5
            },
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Suspects",
                "x": 0.8,
                "y": 0.5
            }
        ]
    }
}
iplot(fig, filename='donut')
gun_violence_filtered[['gun_type']].head(7)
# Apply the function have defined above to each column, creating a new dictionary column
gun_violence_filtered['gun_type_dic'] \
= gun_violence_filtered['gun_type'].apply(lambda x: StringToDic(x))
def CountDfValue(df,col='gun_type_dic'):
    newDic = {}
    for index,row in df.iterrows():
        for key,value in row[col].items():
            if value not in newDic:
                newDic[value] = 1
            else:
                newDic[value] += 1
                
    return newDic

dicGun = CountDfValue(gun_violence_filtered)
del dicGun['Unknown']
gun_violence_filtered[['gun_type_dic']].head(7)
gunList = []
gunCount = []
for i in sorted(dicGun.items(),key=lambda items:items[1],reverse=True):
    gunList.append(i[0])
    gunCount.append(i[1])
# For Victim
trace1 = go.Bar(
    x=gunList,
    y=gunCount,
    marker=dict(
        color='orange'
    )
)


data = [trace1]
layout = go.Layout(
    title='Distribution of Types of Gun',
    xaxis=dict(
        tickfont=dict(
            size=12,
            color='rgb(107, 107, 107)',
        )
    ),
    yaxis=dict(
        title='Count',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    barmode='group',
    bargap=0.15,
    bargroupgap=0.1,
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
df['gun_type_appear'] = df['gun_type_dic'].apply(lambda x: set(x.values()))
def FurthurColCal(df=gun_violence_filtered,colToCal='n_injured',colToMap='gun_type_appear'):
    dicGunCal = {}
    
    for index,row in df.iterrows():
        for item in row[colToMap]:
            if item not in dicGunCal:
                dicGunCal[item] = [1]
                dicGunCal[item].append(int(row[colToCal]))
            else:
                dicGunCal[item][0] += 1
                dicGunCal[item][1] += int(row[colToCal])
    return dicGunCal  
dicGunInjured = FurthurColCal()
del dicGunInjured['Unknown']
dicGunKilled = FurthurColCal(colToCal='n_killed')
del dicGunKilled['Unknown']
GunInjuredList = [(key,values[1]/values[0],values[1]) for key,values in list(dicGunInjured.items())]
GunKilledList = [(key,values[1]/values[0],values[1]) for key,values in list(dicGunKilled.items())]
GunTotalList = [(injured[0],injured[1]+kill[1],injured[2]+kill[2]) for injured,kill in zip(GunInjuredList,GunKilledList)]
GunType = [i[0] for i in GunTotalList]
GunInjuredAverage = [i[1] for i in GunInjuredList]
GunInjuredTotal = [i[2] for i in GunInjuredList]
GunKilledAverage = [i[1] for i in GunKilledList]
GunKilledTotal = [i[2] for i in GunKilledList]
GunTotalAverage = [i[1] for i in GunTotalList]
GunTotalTotal = [i[2] for i in GunTotalList]
# A simple glance of one of the lists
GunTotalList[:5]
# For Victim
trace1 = go.Bar(
    x=GunType,
    y=GunInjuredAverage,
    marker=dict(
        color='orange'
    ),
    name = 'Average Injured'
)
trace2 = go.Bar(
    x=GunType,
    y=GunKilledAverage,
    marker=dict(
        color='red'
    ),
    name = 'Average Killed'
)
data = [trace1,trace2]
layout = go.Layout(
    title='Number of Average Injured and Killed Caused by Each Gun Type',
    xaxis=dict(
        tickfont=dict(
            size=12,
            color='rgb(107, 107, 107)',
        )
    ),
    yaxis=dict(
        title='Count',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    barmode='group',
    bargap=0.15,
    bargroupgap=0.1,
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
# For Victim
trace1 = go.Bar(
    x=GunType,
    y=GunTotalAverage,
    marker=dict(
        color='maroon'
    ),
    name = 'Average Injured & Killed'
)

data = [trace1]
layout = go.Layout(
    title='Number of Average Injured and Killed Caused by Each Gun Type',
    xaxis=dict(
        tickfont=dict(
            size=12,
            color='rgb(107, 107, 107)',
        )
    ),
    yaxis=dict(
        title='Count',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    barmode='group',
    bargap=0.15,
    bargroupgap=0.1,
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
# For Victim
trace1 = go.Bar(
    x=GunType,
    y=GunInjuredTotal,
    marker=dict(
        color='orange'
    ),
    name = 'Total Injured'
)
trace2 = go.Bar(
    x=GunType,
    y=GunKilledTotal,
    marker=dict(
        color='red'
    ),
    name = 'Total Killed'
)
data = [trace1,trace2]
layout = go.Layout(
    title='Number of Total Injured and Killed Caused by Each Gun Type',
    xaxis=dict(
        tickfont=dict(
            size=10,
            color='rgb(107, 107, 107)',
        )
    ),
    yaxis=dict(
        title='Count',
        range = [0,5000],
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    barmode='group',
    bargap=0.15,
    bargroupgap=0.1,
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
gun_violence_filtered[['incident_characteristics','gun_type']].head(6)
# Convert string into dictionary
def StringToList(S1):
    dic1 = {}
    list1 = str(S1).split('||')
        
    return list1
# Apply the function have defined above to each column, creating a new dictionary column
gun_violence_filtered['incident_dic'] \
= gun_violence_filtered['incident_characteristics'].apply(lambda x: StringToList(x))
gun_violence_filtered[['incident_dic']].head(4)
typeDic = {i:{'Shot - Wounded/Injured':0} for i in GunType}
typeDic['Unknown'] = {'Shot - Wounded/Injured':0}
# This function iterate through two column,
# Count appearances of element of one column(list) in terms of another column(dictionary)
def countIncidentType(df=gun_violence_filtered,incidentCol='incident_dic',typeCol='gun_type_appear',typeDic=typeDic):
    dic = {}
    
    for index,row in df.iterrows():
        for guntype in row[typeCol]:
            for incidentList in row[incidentCol]:
                if incidentList not in typeDic[guntype]:
                    typeDic[guntype][incidentList] = 1
                elif incidentList in typeDic[guntype]:
                    typeDic[guntype][incidentList] += 1
             
    return typeDic
def sortDic(dic):
    sortedDic = sorted(dic.items(),key = lambda item: item[1],reverse=True)
    return sortedDic
typeIncidentDic = countIncidentType()
incidentHandGun = [i[0] for i in sortDic(typeIncidentDic['Handgun'])][:15]
incidentHandGunCount = [i[1] for i in sortDic(typeIncidentDic['Handgun'])][:15]
incidentAR = [i[0] for i in sortDic(typeIncidentDic['223 Rem [AR-15]'])][:15]
incidentARCount = [i[1] for i in sortDic(typeIncidentDic['223 Rem [AR-15]'])][:15]
incidentAK = [i[0] for i in sortDic(typeIncidentDic['7.62 [AK-47]'])][:15]
incidentAKCount = [i[1] for i in sortDic(typeIncidentDic['7.62 [AK-47]'])][:15]
incidentRifle = [i[0] for i in sortDic(typeIncidentDic['Rifle'])][:15]
incidentRifleCount = [i[1] for i in sortDic(typeIncidentDic['Rifle'])][:15]
incidentShotgun = [i[0] for i in sortDic(typeIncidentDic['Shotgun'])][:15]
incidentShotgunCount = [i[1] for i in sortDic(typeIncidentDic['Shotgun'])][:15]
incident9mm = [i[0] for i in sortDic(typeIncidentDic['9mm'])][:15]
incident9mmCount = [i[1] for i in sortDic(typeIncidentDic['9mm'])][:15]
# A galance on two of the lists
print('The most frequent incident for Handgun:',sortDic(typeIncidentDic['Handgun'])[:5])
print()
print('The most frequent incident for Rifle:',sortDic(typeIncidentDic['Rifle'])[:5])
# Distribution of Incident among Different Guns(we take 4 types here)
trace1 = go.Bar(
    x=incidentHandGun,
    y=incidentHandGunCount,
    marker=dict(
        color='orange'
    ),
    name = 'HandGun'
)
trace2 = go.Bar(
    x=incidentAR,
    y=incidentARCount,
    marker=dict(
        color='red'
    ),
    name = '223 Rem [AR-15]'
)
trace3 = go.Bar(
    x=incidentAK,
    y=incidentAKCount,
    marker=dict(
        color='maroon'
    ),
    name = '7.62 [AK-47]'
)
trace4 = go.Bar(
    x=incidentRifle,
    y=incidentRifleCount,
    marker=dict(
        color='purple'
    ),
    name = 'Rifle'
)
trace5 = go.Bar(
    x=incidentShotgun,
    y=incidentShotgunCount,
    marker=dict(
        color='plum'
    ),
    name = 'Shotgun'
)

trace6 = go.Bar(
    x=incident9mm,
    y=incident9mmCount,
    marker=dict(
        color='tan'
    ),
    name = '9mm'
)


fig = tools.make_subplots(rows=3, cols=2,subplot_titles=('Handgun', '223 Rem [AR-15]',
                                                          '7.62 [AK-47]', 'Rifle',
                                                          'Shotgun','9mm'))

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
fig.append_trace(trace3, 2, 1)
fig.append_trace(trace4, 2, 2)
fig.append_trace(trace5, 3, 1)
fig.append_trace(trace6, 3, 2)

fig['layout'].update(height=1000, 
                     width=1000, 
                     title='Distribution of Incidents among Different Guns',
                     xaxis1=dict(
                            tickfont=dict(
                            size=7,
                            color='rgb(107, 107, 107)'
                            )),
                     xaxis2=dict(
                            tickfont=dict(
                            size=7,
                            color='rgb(107, 107, 107)'
                            )),
                     xaxis3=dict(
                            tickfont=dict(
                            size=7,
                            color='rgb(107, 107, 107)'
                             )),
                    xaxis4=dict(
                            tickfont=dict(
                            size=7,
                            color='rgb(107, 107, 107)'
                            )),
                    xaxis5=dict(
                            tickfont=dict(
                            size=7,
                            color='rgb(107, 107, 107)'
                             )),
                    xaxis6=dict(
                            tickfont=dict(
                            size=7,
                            color='rgb(107, 107, 107)'
                             )))
iplot(fig, filename='simple-subplot-with-annotations')