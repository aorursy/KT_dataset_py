import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go
import seaborn as sns
sns.set()
py.init_notebook_mode(connected = True)
df = pd.read_csv('../input/log-CTI.csv')
df.head()
print(df.shape)
missing_values_count = df.isnull().sum()
missing_values_count
df = df.drop(['Payload','SourceIpPostalCode','HttpReferrer','HttpUserAgent','HttpUserAgent',
        'HttpMethod','HttpVersion','HttpHost','Custom Field 1','Custom Field 2',
        'Custom Field 3','Custom Field 4','Custom Field 5','SourceIpCountryCode'],axis=1)
df.head()
# legit post request = POST /is-ready HTTP/1.0
# not legit = 33|

requests = []
for i in range(df.shape[0]):
    try:
        if df.HttpRequest.iloc[i].lower().find('post') >= 0:
            requests.append('LEGIT')
        else:
            requests.append('NOT')
    except:
        requests.append('NOT')
        
df['PostLegit'] = requests
df.PostLegit.head()
legit_unique, legit_count = np.unique(df['PostLegit'], return_counts = True)
data = [go.Bar(
            x=legit_unique,
            y=legit_count,
    text=legit_count,
    textposition = 'auto',
            marker=dict(
                color='rgb(179, 224, 255)',
                line=dict(
                    color='rgb(8,48,107)',
                    width=0.5),
            ),
            opacity=0.9
)]
layout = go.Layout(
    title = 'PostLegit count',
     margin = dict(
        t = 50
    )
)
fig = go.Figure(data = data, layout = layout)
py.iplot(fig)
threat = df['Threat Confidence'].unique().tolist()
data_bar = []
for i in threat:
    legit_unique, legit_count = np.unique(df[df['Threat Confidence']==i]['PostLegit'], return_counts = True)
    data_bar.append(go.Bar(x=legit_unique,y=legit_count,name=i + ' threat'))
layout = go.Layout(
    title = 'threat count',
     margin = dict(
        t = 50
    )
)
fig = go.Figure(data = data_bar, layout = layout)
py.iplot(fig)
arrays = []
for i in threat:
    arrays.append(np.unique(df[df['Threat Confidence']==i]['PostLegit'], return_counts = True)[1])
arrays = np.array(arrays)
sum_high = np.sum(arrays[0,:])
sum_low = np.sum(arrays[1,:])

print('ratio high threat for legit:not, %f:%f'%(arrays[0,0]/sum_high,arrays[0,1]/sum_high))
print('ratio low threat for legit:not, %f:%f'%(arrays[1,0]/sum_low,arrays[1,1]/sum_low))
bot_unique, bot_count = np.unique(df['Botnet'], return_counts = True)
data = [go.Bar(
            x=bot_unique,
            y=bot_count,
    text=bot_count,
    textposition = 'auto',
            marker=dict(
                color='rgb(179, 224, 255)',
                line=dict(
                    color='rgb(8,48,107)',
                    width=0.5),
            ),
            opacity=0.9
)]
layout = go.Layout(
    title = 'bot count',
     margin = dict(
        t = 50
    )
)
fig = go.Figure(data = data, layout = layout)
py.iplot(fig)
threat = df['Threat Confidence'].unique().tolist()
data_bar = []
for i in threat:
    bot_unique, bot_count = np.unique(df[df['Threat Confidence']==i]['Botnet'], return_counts = True)
    data_bar.append(go.Bar(x=bot_unique,y=bot_count,name=i + ' threat'))
layout = go.Layout(
    title = 'bot count',
     margin = dict(
        t = 50
    )
)
fig = go.Figure(data = data_bar, layout = layout)
py.iplot(fig)
plt.figure(figsize=(15,7))
colors = ['blue','orange']
for no, i in enumerate(threat):
    plt.scatter(df[df['Threat Confidence']==i]['SourcePort'],df[df['Threat Confidence']==i]['TargetPort'],
                label=i+' threat',color=colors[no])
plt.xlabel('Source port from botnet')
plt.ylabel('targeted port from botnet')
plt.title('scatter study for botnet port')
plt.legend()
plt.show()
queries = df[df['Threat Confidence']=='High']['TargetPort'].values
targetport_below_2k = np.where(queries <= 2000)[0]
print('ratio HIGH threat TARGETED port for botnet <= 2k: > 2k, %f:%f'%(targetport_below_2k.shape[0]/queries.shape[0],
                                                (queries.shape[0]-targetport_below_2k.shape[0])/queries.shape[0]))
queries = df[df['Threat Confidence']=='Low']['TargetPort'].values
targetport_below_2k = np.where(queries <= 2000)[0]
print('ratio LOW threat TARGETED port for botnet <= 2k: > 2k, %f:%f'%(targetport_below_2k.shape[0]/queries.shape[0],
                                                (queries.shape[0]-targetport_below_2k.shape[0])/queries.shape[0]))
queries = df[df['Threat Confidence']=='High']['SourcePort'].values
targetport_below_2k = np.where(queries <= df['SourcePort'].max()/2)[0]
print('ratio HIGH threat SOURCE port for botnet <= 2k: > 2k, %f:%f'%(targetport_below_2k.shape[0]/queries.shape[0],
                                                (queries.shape[0]-targetport_below_2k.shape[0])/queries.shape[0]))
queries = df[df['Threat Confidence']=='Low']['SourcePort'].values
targetport_below_2k = np.where(queries <= df['SourcePort'].max()/2)[0]
print('ratio LOW threat SOURCE port for botnet <= 2k: > 2k, %f:%f'%(targetport_below_2k.shape[0]/queries.shape[0],
                                                (queries.shape[0]-targetport_below_2k.shape[0])/queries.shape[0]))
source_unique, source_count = np.unique(df['SourceIpAsnNr'], return_counts = True)
data = [go.Bar(
            x=source_unique,
            y=source_count,
    text=source_count,
    textposition = 'auto',
            marker=dict(
                color='rgb(179, 224, 255)',
                line=dict(
                    color='rgb(8,48,107)',
                    width=0.5),
            ),
            opacity=0.9
)]
layout = go.Layout(
    title = 'source IP count',
     margin = dict(
        t = 50
    )
)
fig = go.Figure(data = data, layout = layout)
py.iplot(fig)
threat = df['Threat Confidence'].unique().tolist()
data_bar = []
for i in threat:
    isp_unique, isp_count = np.unique(df[df['Threat Confidence']==i]['SourceIpAsnNr'], return_counts = True)
    data_bar.append(go.Bar(x=isp_unique,y=isp_count,name=i + ' threat'))
layout = go.Layout(
    title = 'isp count',
     margin = dict(
        t = 50
    )
)
fig = go.Figure(data = data_bar, layout = layout)
py.iplot(fig)
df['SourceIpCity']=df['SourceIpCity'].fillna('Balingian')
city_unique, city_count = np.unique(df['SourceIpCity'], return_counts = True)
data = [go.Bar(
            x=city_unique,
            y=city_count,
    text=city_count,
    textposition = 'auto',
            marker=dict(
                color='rgb(179, 224, 255)',
                line=dict(
                    color='rgb(8,48,107)',
                    width=0.5),
            ),
            opacity=0.9
)]
layout = go.Layout(
    title = 'City count',
     margin = dict(
        t = 50
    )
)
fig = go.Figure(data = data, layout = layout)
py.iplot(fig)
threat = df['Threat Confidence'].unique().tolist()
data_bar = []
for i in threat:
    city_unique, city_count = np.unique(df[df['Threat Confidence']==i]['SourceIpCity'], return_counts = True)
    data_bar.append(go.Bar(x=city_unique,y=city_count,name=i + ' threat'))
layout = go.Layout(
    title = 'city count',
     margin = dict(
        t = 50
    )
)
fig = go.Figure(data = data_bar, layout = layout)
py.iplot(fig)
data = [ dict(
        type = 'scattergeo',
        lon = df['SourceIpLongitude'],
        lat = df['SourceIpLatitude'],
        text = df['SourceIpCity']+ ': ' + df['Threat Confidence'],
        mode = 'markers',
        marker = dict(
            size = 8,
            opacity = 0.8,
            reversescale = True,
            autocolorscale = False,
            line = dict(
                width=1,
                color='rgba(102, 102, 102)'
            )
        ))]

layout = dict(
        title = 'Source city counts',
        geo = dict(
            scope = 'malaysia',
        showland = True,
        landcolor = "rgb(212, 212, 212)",
        subunitcolor = "rgb(255, 255, 255)",
        countrycolor = "rgb(255, 255, 255)",
        showlakes = True,
        lakecolor = "rgb(255, 255, 255)",
        showsubunits = True,
        showcountries = True,
            lonaxis = dict(
            showgrid = True,
            gridwidth = 0.5,
            range= [ df['SourceIpLongitude'].min()-5, df['SourceIpLongitude'].max()+5],
            dtick = 5
        ),
        lataxis = dict (
            showgrid = True,
            gridwidth = 0.5,
            range= [df['SourceIpLatitude'].min()-5, df['SourceIpLatitude'].max()+5],
            dtick = 5
        )
        ),
    )

fig = dict( data=data, layout=layout)
py.iplot(fig)
first_source, first_num = [], []
for i in range(df.shape[0]):
    first_source.append(df['SourceIp'].iloc[i].split('.')[0]+'.X.X.X')
    first_num.append(int(df['SourceIp'].iloc[i].split('.')[0]))
df['FirstSource'] = first_source
df['FirstNum'] = first_num
df['FirstSource'].head()
firstsource_unique, firstsource_count = np.unique(df['FirstSource'], return_counts = True)
data = [go.Bar(
            x=firstsource_unique,
            y=firstsource_count,
    text=firstsource_count,
    textposition = 'auto',
            marker=dict(
                color='rgb(179, 224, 255)',
                line=dict(
                    color='rgb(8,48,107)',
                    width=0.5),
            ),
            opacity=0.9
)]
layout = go.Layout(
    title = 'firstsource count',
     margin = dict(
        t = 50
    )
)
fig = go.Figure(data = data, layout = layout)
py.iplot(fig)
threat = df['Threat Confidence'].unique().tolist()
data_bar = []
for i in threat:
    FirstSource_unique, FirstSource_count = np.unique(df[df['Threat Confidence']==i]['FirstSource'], return_counts = True)
    data_bar.append(go.Bar(x=FirstSource_unique,y=FirstSource_count,name=i + ' threat'))
layout = go.Layout(
    title = 'FirstSource count',
     margin = dict(
        t = 50
    )
)
fig = go.Figure(data = data_bar, layout = layout)
py.iplot(fig)
targetip_unique, targetip_count = np.unique(df['TargetIp'], return_counts = True)
data = [go.Bar(
            x=targetip_unique,
            y=targetip_count,
    text=targetip_count,
    textposition = 'auto',
            marker=dict(
                color='rgb(179, 224, 255)',
                line=dict(
                    color='rgb(8,48,107)',
                    width=0.5),
            ),
            opacity=0.9
)]
layout = go.Layout(
    title = 'targetip count',
     margin = dict(
        t = 50
    )
)
fig = go.Figure(data = data, layout = layout)
py.iplot(fig)
threat = df['Threat Confidence'].unique().tolist()
data_bar = []
for i in threat:
    TargetIp_unique, TargetIp_count = np.unique(df[df['Threat Confidence']==i]['TargetIp'], return_counts = True)
    data_bar.append(go.Bar(x=TargetIp_unique,y=TargetIp_count,name=i + ' threat'))
layout = go.Layout(
    title = 'TargetIp count',
     margin = dict(
        t = 50
    )
)
fig = go.Figure(data = data_bar, layout = layout)
py.iplot(fig)
from sklearn.preprocessing import LabelEncoder

df['threat_int']=LabelEncoder().fit_transform(df['Threat Confidence'])
plt.figure(figsize=(9,6))
sns.heatmap(df[['SourcePort','TargetPort','FirstNum','threat_int']].corr(), annot=True)
plt.show()
plt.figure(figsize=(20, 15))
sns.pairplot(df[['SourcePort','TargetPort','FirstNum','Threat Confidence']], hue="Threat Confidence")
plt.show()
X = df[['Botnet','FirstSource','PostLegit','SourceIpCity','TargetPort','TargetIp','SourceIpAsnNr','SourcePort']]
X.head()
X[['Botnet','FirstSource','PostLegit','SourceIpCity','TargetIp','SourceIpAsnNr']]=X[['Botnet','FirstSource','PostLegit','SourceIpCity','TargetIp','SourceIpAsnNr']].apply(LabelEncoder().fit_transform)
X.head()
Y = df['threat_int']
Y.head()
from sklearn.cross_validation import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2)
from sklearn.ensemble import *
from sklearn import metrics

gb = GradientBoostingClassifier().fit(X_train, Y_train)
ada = AdaBoostClassifier().fit(X_train, Y_train)
bagging = BaggingClassifier().fit(X_train, Y_train)
rf = RandomForestClassifier().fit(X_train, Y_train)
print(metrics.classification_report(Y_test, gb.predict(X_test), target_names = df['Threat Confidence'].unique()))
print(metrics.classification_report(Y_test, ada.predict(X_test), target_names = df['Threat Confidence'].unique()))
print(metrics.classification_report(Y_test, bagging.predict(X_test), target_names = df['Threat Confidence'].unique()))
print(metrics.classification_report(Y_test, rf.predict(X_test), target_names = df['Threat Confidence'].unique()))
plt.figure(figsize=(15, 5))
plt.bar(np.arange(X.shape[1]),gb.feature_importances_)
plt.xticks(np.arange(X.shape[1]), list(X))
plt.title('Gradient boosting feature importances')
plt.show()
plt.figure(figsize=(15, 5))
plt.bar(np.arange(X.shape[1]),ada.feature_importances_)
plt.xticks(np.arange(X.shape[1]), list(X))
plt.title('Adaboost feature importances')
plt.show()
plt.figure(figsize=(15, 5))
plt.bar(np.arange(X.shape[1]),rf.feature_importances_)
plt.xticks(np.arange(X.shape[1]), list(X))
plt.title('Random Forest feature importances')
plt.show()
