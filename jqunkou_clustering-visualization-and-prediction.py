import warnings 
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import folium

from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from hmmlearn import hmm
df = pd.read_csv('../input/us-weather-events/US_WeatherEvents_2016-2019.csv')
df.tail(3)
datetimeFormat = '%Y-%m-%d %H:%M:%S'
df['End']=pd.to_datetime(df['EndTime(UTC)'], format=datetimeFormat)
df['Start']=pd.to_datetime(df['StartTime(UTC)'], format=datetimeFormat)
df['Duration']=df['End']-df['Start']
df['Duration'] = df['Duration'].dt.total_seconds()
df['Duration'] = df['Duration']/(60*60) #in hours
df = df[(df['Duration']< 30*24) & (df['Duration'] != 0)] #remove obvious wrong data
df.tail(3)
print("Overall Duration Summary")
print("--Count", df['Duration'].size)
print("--%miss.", sum(df['Duration'].isnull()))
print("--card.",df['Duration'].unique().size)
print("--min",df['Duration'].min())
print("--lowerBoundary.",df['Duration'].median()-1.5*((df['Duration'].quantile(0.75))-df['Duration'].quantile(0.25)))
print("--1stQrt",df['Duration'].quantile(0.25))
print("--mean",df['Duration'].mean())
print("--median",df['Duration'].median())
print("--3rdQrt",df['Duration'].quantile(0.75))
print("--upperBoundary.",df['Duration'].median()+1.5*((df['Duration'].quantile(0.75))-df['Duration'].quantile(0.25)))
print("--95%Boundary.",df['Duration'].mean()+1.96*df['Duration'].std())
print("--max",df['Duration'].max())
print("--Std.Dev",df['Duration'].std())
df = df[(df['Duration']< 10)]
df['Duration'].size
df2 = df.groupby(['AirportCode','City','State', 
                  'LocationLat', 'LocationLng','Type']).agg({'Duration':['sum']}).reset_index()
df2.columns=pd.MultiIndex.from_tuples((("AirportCode", " "),("City", " "),
                                       ("State", " "), ("LocationLat", " "),
                                       ("LocationLng", " "), ("Type", " "), ("Duration", " ")))
df2.columns = df2.columns.get_level_values(0)
df2['Duration'] = df2['Duration']/(24*4*3.65) #yearly percentage  
df2 = df2.sort_values(by='Duration')
df2.tail(3)
df_flat = df2.pivot(index='AirportCode', columns='Type', values=['Duration']).reset_index().fillna(0)
df_flat.columns=pd.MultiIndex.from_tuples(((' ', 'AirportCode'),(' ', 'Cold'),(' ', 'Fog'),
            (' ',  'Hail'),(' ', 'Precipitation'),(' ', 'Rain'),(' ', 'Snow'),(' ', 'Storm')))
df_flat.columns = df_flat.columns.get_level_values(1)
#df_flat().tail(3)
uniqueKey = df2[['AirportCode', 'City', 
                 'State', 'LocationLat', 'LocationLng']].sort_values(by='AirportCode').drop_duplicates()
weather = pd.merge(df_flat, uniqueKey, how='inner', on='AirportCode')
weather.tail(3)
fig_sum = px.histogram(df2, x='Type', y= 'Duration',  histfunc = 'avg',
                      title = 'fig 1. National wide weather events duration')
fig_sum.update_xaxes(categoryorder='mean descending')
fig_sum.update_yaxes(title_text='mean of duration% per year')
fig_sum.update_layout(height=750, width=1000)
fig_sum.show()
fig_state=make_subplots(rows=7, cols=1, shared_yaxes=False, vertical_spacing=0.05)

fig_state.add_trace(go.Histogram(x=weather['State'], y=weather['Rain'], name='Rain', histfunc ='avg'),1,1)
fig_state.add_trace(go.Histogram(x=weather['State'], y=weather['Fog'], name='Fog', histfunc ='avg'),2,1)
fig_state.add_trace(go.Histogram(x=weather['State'], y=weather['Snow'], name='Snow', histfunc ='avg'),3,1)
fig_state.add_trace(go.Histogram(x=weather['State'], y=weather['Cold'], name='Cold', histfunc ='avg'),4,1)
fig_state.add_trace(go.Histogram(x=weather['State'], y=weather['Precipitation'], name='Precipitation', histfunc ='avg'),5,1)
fig_state.add_trace(go.Histogram(x=weather['State'], y=weather['Storm'], name='Storm', histfunc ='avg'),6,1)
fig_state.add_trace(go.Histogram(x=weather['State'], y=weather['Hail'], name='Hail', histfunc ='avg'),7,1)

fig_state['layout']['xaxis7'].update(title="State")
fig_state['layout']['yaxis4'].update(title="duration% per year")
fig_state.update_xaxes(categoryorder='mean descending')
fig_state.update_layout( title_text="fig 2. State wide weather events distribution")
fig_state.show()
fig_city = px.scatter_geo(weather, lat='LocationLat', lon='LocationLng',
                      hover_name=weather['City'] + ', ' + weather['State'],
                      scope="usa",
                      title ='fig 3. Cities involved in this dataset')
#fig_city.update_layout(height=750, width=1000)
fig_city.show()
fig_rain = px.scatter_geo(weather, lat='LocationLat', lon='LocationLng',
                      color="Rain",
                      hover_name=weather['City'] + ', ' + weather['State'],
                      color_continuous_scale='dense', 
                      range_color = [0,16],
                      scope="usa",
                      title ='fig 4. City wide rainy days percentage each year from 2016 to 2019')
#fig_rain.update_layout(height=750, width=1000)
fig_rain.show()
fig_fog = px.scatter_geo(weather, lat='LocationLat', lon='LocationLng',
                      color="Fog",
                      hover_name=weather['City'] + ', ' + weather['State'],
                      color_continuous_scale='dense', 
                      #range_color = [0,16],
                      scope="usa",
                      title ='fig 5. City wide foggy days percentage each year from 2016 to 2019')
#fig_fog.update_layout(height=750, width=1000)
fig_fog.show()
fig_snow = px.scatter_geo(weather, lat='LocationLat', lon='LocationLng',
                      color="Snow",
                      #size="Snow",
                      hover_name=weather['City'] + ', ' + weather['State'],
                      color_continuous_scale='dense', 
                      #range_color = [0,16],
                      scope="usa",
                      title ='fig 6. City wide snow days percentage each year from 2016 to 2019')
#fig_snow.update_layout(height=750, width=1000)
fig_snow.show()
fig_cold = px.scatter_geo(weather, lat='LocationLat', lon='LocationLng',
                      color="Cold",
                      #size="Snow",
                      hover_name=weather['City'] + ', ' + weather['State'],
                      color_continuous_scale='dense', 
                      #range_color = [0,16],
                      scope="usa",
                      title ='fig 7. City wide cold days percentage each year from 2016 to 2019')
#fig_cold.update_layout(height=750, width=1000)
fig_cold.show()
X = df_flat.drop(['AirportCode','Cold', 'Hail'], axis=1)
X.tail(3)
from sklearn.cluster import KMeans
distortions = []

K = range(1,20)
for k in K:
    kmean = KMeans(n_clusters=k, random_state=0, n_init = 50, max_iter = 500)
    kmean.fit(X)
    distortions.append(kmean.inertia_)

fig_kmean = px.scatter(x=K, y=distortions, trendline=distortions, title='fig 8. The Elbow Method')
fig_kmean.update_xaxes(title_text='k')
fig_kmean.update_yaxes(title_text='distortion')
#fig_kmean.update_layout(height=650, width=1000)
fig_kmean.show()
kmeans = KMeans(n_clusters=4, random_state=0).fit(X)

df_flat['Cluster'] = (kmeans.labels_).astype(str)
df_cluster = pd.merge(df_flat[['AirportCode','Cluster']], weather.drop(['Cold','Hail'], axis=1), 
                      how='inner', on='AirportCode')
df_cluster.tail(3)
fig_cluster = px.scatter_geo(df_cluster, lat='LocationLat', lon='LocationLng',
                      hover_name=weather['City'] + ', ' + weather['State'],
                      scope="usa",
                      color_discrete_sequence =['#AB63FA', '#EF553B', '#00CC96','#636EFA'],
                      color = 'Cluster',
                      title ='fig 9. City wide weather cluster distribution')
#fig_cluster.update_layout(height=750, width=1000)
fig_cluster.show()
df_cluster2 = df_cluster.groupby(['State','Cluster']).agg({'Cluster':['count']}).reset_index()
df_cluster2.columns=pd.MultiIndex.from_tuples((("State", " "),("Cluster", " "),("Count", " ")))
df_cluster2.columns = df_cluster2.columns.get_level_values(0)
#df_cluster2.tail(3) #state with each cluster counts

df_loc = df_cluster[['State','Cluster','LocationLat', 'LocationLng']]
df_loc1 = df_loc.groupby(['State','Cluster']).agg({'LocationLat':'mean'}).reset_index()
df_loc2 = df_loc.groupby(['State','Cluster']).agg({'LocationLng':'mean'}).reset_index()
df_loc3 = pd.merge(df_loc1,df_loc2, how='inner', on=['State','Cluster'])
#df_loc3.tail(3) #state with cluster and location

df_clusterS = pd.merge(df_loc3,df_cluster2, how='inner', on=['State','Cluster'])
df_clusterS.tail(3) #state with each cluster count location
fig_clusterS = px.scatter_geo(df_clusterS, lat='LocationLat', lon='LocationLng', 
                     color='Cluster',
                     size='Count',
                     color_discrete_sequence=['#636EFA', '#AB63FA', '#EF553B','#00CC96'],
                     hover_name='State',
                     scope="usa",
                     title = 'fig 10. State wide weather cluster distribution')
#fig_clusterS.update_layout(height=750, width=1000)
fig_clusterS.show()
prop = df_cluster[['Cluster', 'Fog',
                   'Precipitation','Rain', 'Snow', 'Storm']].groupby(['Cluster']).mean().reset_index()
prop2 = prop.transpose().reset_index()
prop2 = prop2[(prop2['index'] !='Cluster')].sort_values(by=0)
prop2
#from plotly.subplots import make_subplots
fig_prop=make_subplots(rows=1, cols=4, shared_yaxes=True,horizontal_spacing=0)

fig_prop.add_trace(go.Bar(x=prop2['index'], y=prop2[0], name='Cluster 0'), row=1, col=1)
fig_prop.add_trace(go.Bar(x=prop2['index'], y=prop2[1], name='Cluster 1'), row=1, col=2)
fig_prop.add_trace(go.Bar(x=prop2['index'], y=prop2[2], name='Cluster 2'), row=1, col=3)
fig_prop.add_trace(go.Bar(x=prop2['index'], y=prop2[3], name='Cluster 3'), row=1, col=4)

fig_prop.update_yaxes(title_text="duration%/year", row=1, col=1)
fig_prop.update_layout(title_text="fig 11. Weather distribution in each cluster")
#fig_prop.update_layout(height=550, width=1000)
fig_prop.show()
df3 = weather[(weather['City']=='Seattle')| (weather['City']=='Detroit')|(weather['City']== 'Kansas City')|(weather['City']== 'Denver')]
#df3
df4 = df3[['Storm', 'Precipitation','Snow', 'Fog','Rain', 'City']].groupby(['City']).mean().reset_index()
df4 = df4.transpose().reset_index()
df4.columns = df4.iloc[0]
df4 = df4[(df4['City'] !='City')]
df4
fig_city=make_subplots(rows=1, cols=4, shared_yaxes=True,horizontal_spacing=0)

fig_city.add_trace(go.Bar(x=df4['City'], y=df4['Kansas City'], name='Cluster0'), row=1, col=1)
fig_city.add_trace(go.Bar(x=df4['City'], y=df4['Denver'], name='Cluster1'), row=1, col=2)
fig_city.add_trace(go.Bar(x=df4['City'], y=df4['Detroit'], name='Cluster2'), row=1, col=3)
fig_city.add_trace(go.Bar(x=df4['City'], y=df4['Seattle'], name='Cluster3'), row=1, col=4)

fig_city['layout']['xaxis1'].update(title="Kansas City, MO")
fig_city['layout']['xaxis2'].update(title="Denver, CO")
fig_city['layout']['xaxis3'].update(title="Detroit, MI")
fig_city['layout']['xaxis4'].update(title="Seattle, WA")
fig_city.update_yaxes(title_text="duration%/year", row=1, col=1)
fig_city.update_layout(title_text="fig 12. Representative cities in each cluster")
#fig_city.update_layout(height=550, width=1000)
fig_city.show()
pca = PCA().fit(X)
pcaX = pca.transform(X)
c0 = []
c1 = []
c2 = []
c3 = []

for i in range(len(pcaX)):
    if kmeans.labels_[i] == 0:
        c0.append(pcaX[i])
    if kmeans.labels_[i] == 1:
        c1.append(pcaX[i])
    if kmeans.labels_[i] == 2:
        c2.append(pcaX[i])
    if kmeans.labels_[i] == 3:
        c3.append(pcaX[i])
c0 = np.array(c0)
c1 = np.array(c1)
c2 = np.array(c2)
c3 = np.array(c3)

fig_pca = make_subplots(rows=1, cols=2, column_widths=[0.3, 0.7], specs=[[{'type':'domain'}, {'type': 'mesh3d'}]])

fig_pca.add_trace(go.Pie(values = pca.explained_variance_ratio_), row=1, col=1)

fig_pca.add_trace(go.Scatter3d(x=c0[:,0], y=c0[:,1], z=c0[:,2],
                               mode='markers', name='Cluster0'),  row=1, col=2)
fig_pca.add_trace(go.Scatter3d(x=c1[:,0], y=c1[:,1], z=c1[:,2], 
                               mode='markers', name='Cluster1'),  row=1, col=2)
fig_pca.add_trace(go.Scatter3d(x=c2[:,0], y=c2[:,1], z=c2[:,2], 
                               mode='markers', name='Cluster2'),  row=1, col=2)
fig_pca.add_trace(go.Scatter3d(x=c3[:,0], y=c3[:,1], z=c3[:,2], 
                               mode='markers', name='Cluster3'),  row=1, col=2)

fig_pca.update_layout(height=750, width=1000, title_text=
          "fig 13. PCA: explained variance%(left) and First 3 Component with mapped cluster(right)")

fig_pca.show()
atlanta = df[(df['City']== 'Atlanta')]
print(atlanta['LocationLat'].unique())
print(atlanta['LocationLng'].unique())
import folium
m = folium.Map(location=[33.755238, -84.388115])
folium.Marker(
    location=[33.6301, -84.4418],
).add_to(m)
folium.Marker(
    location=[33.8784, -84.298],
).add_to(m)
folium.Marker(
    location=[33.7776, -84.5246],
).add_to(m)
m
atlanta = atlanta[(atlanta['LocationLng']== -84.4418)]

atlanta_m = atlanta.loc[(atlanta['Start'].dt.month==12) | (atlanta['Start'].dt.month==11)| (atlanta['Start'].dt.month==1)]
atlanta_m = atlanta_m.loc[(atlanta_m['Type']!='Hail')&(atlanta_m['Type']!='Precipitation')]
print(atlanta_m['Type'].unique())
print(atlanta_m['Severity'].unique())
print(atlanta_m['Type'].count())
X2 = atlanta_m[['Type', 'Severity']]
X2_test = X2.tail(5)
X2_train = X2.head(675)
print(X2_train.tail())
print(X2_test)
print(X2['Severity'].describe())
Type = {'Rain':0, 'Fog':1, 'Snow':2} 
X2.Type = [Type[item] for item in X2.Type] 

Severity = {'Light':0, 'Moderate':1, 'Severe':2, 'Heavy':2}
X2.Severity = [Severity[item] for item in X2.Severity] 
X2_test = X2.tail(5)
X2_train = X2.head(675)
print(X2_train.tail())
print(X2_test)
# A function that implements the Markov model to forecast the state. [3]
def transition_matrix(transitions):
    n = 1+ max(transitions) #number of states

    M = [[0]*n for _ in range(n)]

    for (i,j) in zip(transitions,transitions[1:]):
        M[i][j] += 1

    #now convert to probabilities:
    for row in M:
        s = sum(row)
        if s > 0:
            row[:] = [f/s for f in row]
    return M

#test:
t = list(X2_train.Type)
m = transition_matrix(t)
for row in m: print(' '.join('{0:.2f}'.format(x) for x in row))  
# The statespace
states = ["Rain","Fog","Snow"]

# Possible sequences of events
transitionName = [["RR","RF","RS"],["FR","FF","FS"],["SR","SF","SS"]]

# Probabilities matrix (transition matrix)
transitionMatrix = [[0.89,0.10,0.02],[0.59,0.39,0.02],[0.33,0.07,0.60]]

# A function that implements the Markov model to forecast the state. [4]
def activity_forecast(days):
    # Choose the starting state
    activityToday = "Fog"
    print("Start state: " + activityToday)
    # Shall store the sequence of states taken. So, this only has the starting state for now.
    activityList = [activityToday]
    i = 0
    # To calculate the probability of the activityList
    prob = 1
    while i != days:
        if activityToday == "Rain":
            change = np.random.choice(transitionName[0],replace=True,p=transitionMatrix[0])
            if change == "RR":
                prob = prob * 0.89
                activityList.append("Rain")
                pass
            elif change == "RF":
                prob = prob * 0.1
                activityToday = "Fog"
                activityList.append("Fog")
            else:
                prob = prob * 0.02
                activityToday = "Snow"
                activityList.append("Snow")
        elif activityToday == "Fog":
            change = np.random.choice(transitionName[1],replace=True,p=transitionMatrix[1])
            if change == "FR":
                prob = prob * 0.59
                activityList.append("Rain")
                pass
            elif change == "FF":
                prob = prob * 0.39
                activityToday = "Fog"
                activityList.append("Fog")
            else:
                prob = prob * 0.02
                activityToday = "Snow"
                activityList.append("Snow")
        elif activityToday == "Snow":
            change = np.random.choice(transitionName[2],replace=True,p=transitionMatrix[2])
            if change == "SR":
                prob = prob * 0.33
                activityList.append("Rain")
                pass
            elif change == "SF":
                prob = prob * 0.07
                activityToday = "Fog"
                activityList.append("Fog")
            else:
                prob = prob * 0.6
                activityToday = "Snow"
                activityList.append("Snow")
        i += 1  
    print("Possible states: " + str(activityList))
    print("Probability of the possible sequence of states: " + str(prob))
    
# Function that forecasts the possible state for the next 4 events
activity_forecast(4)
print('Acutal sequence of states: Rain, Fog, Rain, Rain, Rain')
activity_forecast(4)
print('Acutal sequence of states: Rain, Fog, Rain, Rain, Rain')
from hmmlearn import hmm
model = hmm.MultinomialHMM(n_components=3, n_iter=50, tol=0.001)
model.startprob_=np.array([0.82, 0.132, 0.044])
model.fit(np.array(X2_train))
#model.fit(X2_train['Type'], X2_train['Severity'])
print('startprob', np.round(model.startprob_,3))
print('transmat', np.round(model.transmat_,3))
print('emissionprob', np.round(model.emissionprob_,3))
print('score', model.score(X2))
print ('Acutal sequence of states: 0, 2, 0, 0 ,0')
seen = np.array(X2_test[['Type']])
model.decode(seen, algorithm='viterbi')