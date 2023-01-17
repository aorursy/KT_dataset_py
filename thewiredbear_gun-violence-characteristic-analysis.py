# Import all required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
from plotly import __version__
import plotly.graph_objs as go 
import plotly.plotly as py
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True) 
%matplotlib inline
import re
# Read in the data
data = pd.read_csv('../input/gun-violence-data_01-2013_03-2018.csv')

# Basic outlook of the data
data.head()
# Information about the dataset and datatypes
data.info()
plt.figure(figsize=(16,5))
sns.heatmap(data.isnull(),cbar=False,yticklabels=False)
plt.show()
from urllib.parse import urlparse

# Even though we have only a few null values, we still remove them
data_source = data.dropna(subset=['sources'],axis=0)

# We rebuild the index as it was broken by removing null values
data_source.reset_index(drop=True,inplace=True)

data_source['clean_source']= [urlparse(data_source.sources[i]).netloc for i in range(len(data_source.sources))]
data_source['clean_source'] = [data_source.clean_source[i].replace('www.','').replace('.com','') for i in range(len(data_source.sources))]
data_short = data_source['clean_source'].value_counts().head(15).to_dict()
x_list = list(data_short.keys())
y_list = list(data_short.values())
data1 = [go.Bar(x=x_list,y=y_list,marker=dict(color=['grey','grey','grey','grey','grey','grey','red','grey','grey','grey','grey','grey','grey','grey','grey']))]
iplot(data1)
#list(data_short.keys())
data_geo = data['state'].value_counts()
x_geo = list(data_geo.keys())
y_geo = list(data_geo.values)
mean_count = data['state'].value_counts().mean()
geo_graph = [go.Bar(x=x_geo,y=y_geo)]
geo_layout = {'shapes':[{'type':'line','x0':0,'y0':mean_count,'x1':50,'y1':mean_count}]}
iplot({'data':geo_graph,'layout':geo_layout})
## Here we change the date column in the date to individually work with years, months and days.
data['date']=pd.to_datetime(data['date'])
data['year'] = data['date'].dt.year
data['month']=data['date'].dt.month
data['day_of_week'] = data['date'].dt.dayofweek
data.head()
year_data = data.groupby('year')['n_killed','n_injured'].sum()

nk_x = list(year_data['n_killed'].keys())
nk_y = list(year_data['n_killed'].values)
nk_graph = [go.Bar(x=nk_x,y=nk_y,marker=dict(color='crimson'))]
nk_layout = go.Layout(title='Year wise Deaths in Shootings')
iplot({'data':nk_graph,'layout':nk_layout})
ni_x = list(year_data['n_injured'].keys())
ni_y = list(year_data['n_injured'].values)
ni_graph = [go.Bar(x=ni_x,y=ni_y,marker=dict(color='lightcoral'))]
ni_layout = go.Layout(title='Year wise Injuries in Shootings')
iplot({'data':ni_graph,'layout':ni_layout})
data['gun_type_parsed'] = data['gun_type'].fillna('0:Unknown')
gt = data.groupby(by=['gun_type_parsed']).agg({'n_killed': 'sum', 'n_injured' : 'sum', 'state' : 'count'}).reset_index().rename(columns={'state':'count'})

results = {}
for i, each in gt.iterrows():
    wrds = each['gun_type_parsed'].split("||")
    for wrd in wrds:
        if "Unknown" in wrd:
            continue
        wrd = wrd.replace("::",":").replace("|1","")
        gtype = wrd.split(":")[1]
        if gtype not in results: 
            results[gtype] = {'killed' : 0, 'injured' : 0, 'used' : 0}
        results[gtype]['killed'] += each['n_killed']
        results[gtype]['injured'] +=  each['n_injured']
        results[gtype]['used'] +=  each['count']
        
resultFrame = pd.DataFrame(results)

resultFrame.transpose().plot.bar(figsize=(12,3),title="Handgun Type Used")
n_guns = data[data["n_guns_involved"].notnull()]
n_guns["n_guns_involved"] = n_guns["n_guns_involved"].astype(int)
n_guns = n_guns[["n_guns_involved"]]

def label(n_guns):
    if n_guns["n_guns_involved"] == 1 :
        return "ONE-GUN"
    elif n_guns["n_guns_involved"] > 1 :
        return "GREATER THAN ONE GUN"

n_guns["x"] = n_guns.apply(lambda n_guns:label(n_guns),axis=1)
n_guns["x"].value_counts().plot.pie(figsize=(7,7),autopct ="%1.0f%%",explode = [0,.2],shadow = True,colors=["indianred","grey"],startangle =25)
plt.title("NO OF GUNS INVOLVED")
plt.ylabel("")
age = data[data["participant_age"].notnull()][["participant_age"]]
age["participant_age"] = age["participant_age"].str.replace("::","-")
age["participant_age"] = age["participant_age"].str.replace(":","-")
age["participant_age"] = age["participant_age"].str.replace("[||]",",")
age = pd.DataFrame(age["participant_age"])
x1 = pd.DataFrame(age["participant_age"].str.split(",").str[0])
x2 = pd.DataFrame(age["participant_age"].str.split(",").str[1])
x3 = pd.DataFrame(age["participant_age"].str.split(",").str[2])
x4 = pd.DataFrame(age["participant_age"].str.split(",").str[3])
x5 = pd.DataFrame(age["participant_age"].str.split(",").str[4])
x6 = pd.DataFrame(age["participant_age"].str.split(",").str[5])
x7 = pd.DataFrame(age["participant_age"].str.split(",").str[6])
x1 = x1[x1["participant_age"].notnull()]
x2 = x2[x2["participant_age"].notnull()]
x3 = x3[x3["participant_age"].notnull()]
x4 = x4[x4["participant_age"].notnull()]
x5 = x5[x5["participant_age"].notnull()]
x6 = x6[x6["participant_age"].notnull()]
x7 = x7[x7["participant_age"].notnull()]

age_dec  = pd.concat([x1,x2,x3,x4,x5,x6,x7],axis = 0)
age_dec["lwr_lmt"] = age_dec["participant_age"].str.split("-").str[0]
age_dec["upr_lmt"] = age_dec["participant_age"].str.split("-").str[1]
age_dec.head()

age_dec= age_dec[age_dec["lwr_lmt"]!='']
age_dec["lwr_lmt"] = age_dec["lwr_lmt"].astype(int)
age_dec["upr_lmt"] = age_dec["upr_lmt"].astype(int)

age_dec["age_bins"] = pd.cut(age_dec["upr_lmt"],bins=[0,20,35,55,130],labels=["TEEN[0-20]","YOUNG[20-35]","MIDDLE-AGED[35-55]","OLD[>55]"])
plt.figure(figsize=(8,8))
age_dec["age_bins"].value_counts().plot.pie(autopct = "%1.0f%%",shadow =True,startangle = 0,colors = sns.color_palette("prism",5),
                                            wedgeprops = {"linewidth" :3,"edgecolor":"k"})
my_circ = plt.Circle((0,0),.7,color = "white")
plt.gca().add_artist(my_circ)
plt.ylabel("")
plt.title("Distribution of age groups of participants",fontsize=20)
from collections import Counter

total_incidents = []
for i, each_inc in enumerate(data['incident_characteristics'].fillna('Not Available')):
    split_vals = [x for x in re.split('\|', each_inc) if len(x)>0]
    total_incidents.append(split_vals)
    if i == 0:
        unique_incidents = Counter(split_vals)
    else:
        for x in split_vals:
            unique_incidents[x] +=1

unique_incidents = pd.DataFrame.from_dict(unique_incidents, orient='index')
colvals = unique_incidents[0].sort_values(ascending=False).index.values
find_val = lambda searchList, elem: [[i for i, x in enumerate(searchList) if (x == e)][0] for e in elem]

a = np.zeros((data.shape[0], len(colvals)))
for i, incident in enumerate(total_incidents):
    aval = find_val(colvals, incident)
    a[i, np.array(aval)] = 1
incident = pd.DataFrame(a, index=data.index, columns=colvals)

prominent_incidents = incident.sum()[[4, 5, 6, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
                                      21, 23,22,24,45,51]]
fig = {
    'data': [
        {
            'labels': prominent_incidents.index,
            'values': prominent_incidents,
            'type': 'pie',
            'hoverinfo':'label+percent+name',
            "domain": {"x": [0, .45]},
        }
    ],
    'layout': {'title': 'Prominent Incidents of Gun Violence',
               'showlegend': False}
}
iplot(fig)
relation = data['participant_relationship']
relation = relation[relation.notnull()]
relation = relation.str.replace("[:|0-9]"," ").str.upper()
relation1 = pd.DataFrame({"count":[len(relation[relation.str.contains("FAMILY")]),
               len(relation[relation.str.contains("ROBBERY")]),
               len(relation[relation.str.contains("FRIENDS")]),
               len(relation[relation.str.contains("AQUAINTANCE")]),
               len(relation[relation.str.contains("NEIGHBOR")]),
               len(relation[relation.str.contains("INVASION")]),
               len(relation[relation.str.contains("CO-WORKER")]),
               len(relation[relation.str.contains("GANG")]),
               len(relation[relation.str.contains("RANDOM")]),
               len(relation[relation.str.contains("MASS SHOOTING")])],
              "category":["FAMILY","ROBBERY","FRIENDS","AQUAINTANCE","NEIGHBOR","INVASION","CO-WORKER","GANG","RANDOM","MASS SHOOTING"]})
relation1
plt.figure(figsize=(14,5))
sns.barplot("category","count",data=relation1,palette="prism")
plt.title("COUNT PLOT FOR PARTICPANT RELATION TYPE IN VIOLENT EVENTS")
data['gun_stolen'] = data['gun_stolen'].fillna('Null')

data['gun_stolen'] = data['gun_stolen'].str.replace('::',',')
data['gun_stolen'] = data['gun_stolen'].str.replace('|',' ')
data['gun_stolen'] = data['gun_stolen'].str.replace(',',' ')
data['gun_stolen']= data['gun_stolen'].str.replace('\d+', '')


data['Stolenguns']=data['gun_stolen'].apply(lambda x: x.count('Stolen'))
data['stolenguns']=data['gun_stolen'].apply(lambda x: x.count('stolen'))
data['Stolengunstotal'] = data['Stolenguns'] + data['stolenguns']

df_year_stolenguns = data[['year','Stolengunstotal']].groupby(['year'], as_index = False).sum()


df_year_stolenguns[['year','Stolengunstotal']].set_index('year').plot(kind='bar')
from collections import Counter
big_text = "||".join(data['incident_characteristics'].dropna()).split("||")
incidents = Counter(big_text).most_common(30)
xx = [x[0] for x in incidents]
yy = [x[1] for x in incidents]

trace1 = go.Bar(
    x=yy[::-1],
    y=xx[::-1],
    name='Incident Characterisitcs',
    marker=dict(color='purple'),
    opacity=0.3,
    orientation="h"
)
data1 = [trace1]
layout = go.Layout(
    barmode='group',
    margin=dict(l=350),
    width=800,
    height=600,
    legend=dict(dict(x=-.1, y=1.2)),
    title = 'Key Incident Characteristics',
)

fig = go.Figure(data=data1, layout=layout)
iplot(fig, filename='grouped-bar')
