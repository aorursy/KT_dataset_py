import numpy as np
import datetime
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import pandas as pd
df_world= pd.read_csv('../input/colombiasrelevantinformationfromuncovercovid19/1.csv')
df_latam= pd.DataFrame(df_world[df_world['country_region']=='Argentina'])
latamcountries=[ 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Costa Rica', 'Cuba', 'Ecuador', 'El Salvador', 'Guatemala', 'Honduras', 'Mexico', 'Nicaragua', 'Panama', 'Paraguay', 'Peru',  'Uruguay', 'Venezuela']
df_latam.append(df_world[df_world['country_region'] =='Colombia'])
for country in latamcountries:
    df_latam=df_latam.append(df_world[df_world['country_region'] ==country])
#print(df_latam['date'])
number_of_days= len(df_latam.groupby('date')['date'].nunique())
days = pd.date_range(start="2020-01-22",end="2020-03-31")
country_dates= df_latam.pivot_table(values='confirmed', index='country_region', columns='date')


for counry in latamcountries:
    plt.plot(days,df_latam[df_latam['country_region']==counry]['confirmed'], label= counry)
plt.xticks(rotation=90)
plt.legend(bbox_to_anchor=(1.1, 1.05))
plt.title('Latam confirmed cases between "2020-01-22"and "2020-03-31"')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime
df_pol= pd.read_csv('../input/school-and-politics/acaps.csv')
#print(df_pol.head())
df_pollatam= pd.DataFrame(df_pol[df_pol['country']=='Argentina'])
border_closed=[]
for country in latamcountries:
    df_pollatam=df_pollatam.append(df_pol[df_pol['country'] ==country])
df_pollatam.fillna(' ', inplace=True)
closed_boards=df_pollatam[pd.Series(df_pollatam['measure']).str.contains('lockdown')]

closed_boards.drop_duplicates(subset ="country", 
                     keep = 'first', inplace = True)

closed_boards.reindex(np.random.permutation(closed_boards.index))
plt.figure(figsize=(18,5))

names = closed_boards['country'].values.tolist()

dates = closed_boards['entry_date'].values.tolist()

values= [2,5,5,2,5,5,5,2,5,5,2,5,]

# print count
# Convert date strings (e.g. 2014-10-18) to datetime
markerline, stemlines, baseline = plt.stem(dates, values, '-.')
for x,y,val in zip(dates, names,values):
    plt.annotate(y, xy=(x,val), xytext=(0,5), textcoords='offset points',ha='center', rotation=35)
plt.title('Timeline of border closing')
plt.ylim(0,7)
import random
cl_schools=df_pollatam[pd.Series(df_pollatam['measure']).str.contains('Schools')]

plt.figure(figsize=(18,5))
cl_schools.drop_duplicates(subset ="country", 
                     keep = 'first', inplace = True)
names = cl_schools['country'].values.tolist()

dates = cl_schools['entry_date'].values.tolist()
values= [4,8,8,8,8,5,8,2,4,8,2,6,8,4]
# print count
# Convert date strings (e.g. 2014-10-18) to datetime
markerline, stemlines, baseline = plt.stem(dates, values, '-.')
for x,y,val in zip(dates, names,values):
    plt.annotate(y, xy=(x,val), xytext=(0,5), textcoords='offset points',ha='center', rotation=35)
plt.tick_params(bottom='off')
plt.title('Timeline of school closing')
plt.ylim(0,13)
df_emer=df_pollatam[pd.Series(df_pollatam['measure']).str.contains('emergency')]
df_emer= df_emer.append(df_pollatam[pd.Series(df_pollatam['measure']).str.contains('Emergency')])
plt.figure(figsize=(18,5))
df_emer.drop_duplicates(subset ="country", 
                     keep = 'first', inplace = True)
names = df_emer['country'].values.tolist()

dates = df_emer['entry_date'].values.tolist()
values= [8,4,8,8,5,8,8,8,4,8,2,6,2]
# print count
# Convert date strings (e.g. 2014-10-18) to datetime
markerline, stemlines, baseline = plt.stem(dates, values, '-.')
for x,y,val in zip(dates, names,values):
    plt.annotate(y, xy=(x,val), xytext=(0,5), textcoords='offset points',ha='center', rotation=35)
plt.tick_params(bottom='off')
plt.title('Timeline of state of emergency declaration')
plt.ylim(0,13)

df_data=pd.read_csv('../input/colombiasrelevantinformationfromuncovercovid19/11.csv')
df_mov=pd.read_csv('../input/colombiasrelevantinformationfromuncovercovid19/9.csv')
df_mov= df_mov[(df_mov['date']=='2020-04-17') & (df_mov['region']=='Total')]
df_mov=df_mov.set_index('country').join(df_data.set_index('country'))
df_mov=df_mov.dropna(subset=['total_cases'])
df_mov=df_mov.drop(columns=['new_cases','new_deaths'])
df_mov=df_mov.dropna(axis=0)
ac=pd.DataFrame()
for c in names:
    ac=ac.append(df_mov.loc[c])

plt.figure(figsize=(18,5))
plt.subplot(131)
plt.scatter(names, ac['grocery_and_pharmacy'],marker=">")
plt.title('grocery and pharmacy')
plt.xticks( rotation=60)
plt.subplot(132)
plt.title('Parks')
plt.scatter(names, ac['parks'],marker=">")
plt.xticks( rotation=60)
plt.subplot(133)
plt.title('residential')
plt.scatter(names, ac['residential'],marker=">")
plt.xticks( rotation=60)

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

df_data=pd.read_csv('../input/colombiasrelevantinformationfromuncovercovid19/11.csv')
df_mov=pd.read_csv('../input/colombiasrelevantinformationfromuncovercovid19/9.csv')
df_mov= df_mov[(df_mov['date']=='2020-04-17') & (df_mov['region']=='Total')]
df_mov=df_mov.set_index('country').join(df_data.set_index('country'))
df_mov=df_mov.dropna(subset=['total_cases'])
df_mov=df_mov.drop(columns=['new_cases','new_deaths'])
df_mov=df_mov.dropna(axis=0)
X=df_mov[['retail','grocery_and_pharmacy','parks','transit_stations','workplaces','residential']]

Y=df_mov['total_cases']
x_train, x_test, y_train, y_test = train_test_split(X,Y, train_size=0.75)
for i in range (10):
    model = DecisionTreeRegressor(max_depth=100+10*i ,criterion="mae",splitter="random")
    plt.figure(figsize=(20,5))
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    plt.scatter(x_test.index,y_pred)
    plt.scatter(x_test.index,y_test)
    plt.xticks(rotation=45,size=15)
    plt.title('Predicted total death using decission Trees with '+str(10+i)+'0')
plt.show()

import numpy as np
from sklearn.manifold import TSNE
import sklearn.cluster
ac=pd.read_csv('../input/colombiasrelevantinformationfromuncovercovid19/3.csv')
df_rec=ac[ac['date']=='2020-03-31']

fig1, axs2 = plt.subplots()

sicol=ac[ac['country_region']=='Colombia']
#print(sicol)

axs2.plot(np.arange(70),sicol['recovered'],'r--o')
axs2.set_title('Colombia evolution')
axs2.set_xlabel('Days')
axs2.set_ylabel('Recupered')

fig1, axs1 = plt.subplots()
axs1.scatter(df_rec['lat'],df_rec['long'],s=np.log(df_rec['recovered']))
axs1.set_title('Recupered per latitude and longitude')
axs1.set_xlabel('latitude')
axs1.set_ylabel('longitude')

fig1, axs3 = plt.subplots()

ad=pd.read_csv('../input/colombiasrelevantinformationfromuncovercovid19/7.csv')
ad= ad.notna()
X=ad[['lat','long','incident_rate']]
X.fillna(0)
Y=ad['mortality_rate']
X_embedded = TSNE(n_components=2).fit_transform(X)
X_embedded.shape
#print (X_embedded)
n_clusters = 2
k_means = sklearn.cluster.KMeans(n_clusters=n_clusters)
k_means.fit(X_embedded) # training
cluster = k_means.predict(X_embedded) # predice a cual cluster corresponde cada elmento
distance = k_means.transform(X_embedded) 
axs3.scatter(X_embedded[:,0], X_embedded[:,1], c=cluster)
axs3.set_title('Cluster formed')

#print (df_rec[df_rec['date']=='2020-04-27'])
