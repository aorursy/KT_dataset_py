import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob

import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import datetime
import time
from plotly.subplots import make_subplots

video = False
df2 = pd.read_csv("/kaggle/input/uncover/UNCOVER/our_world_in_data/coronavirus-disease-covid-19-statistics-and-research.csv")
df2.keys()
df2[df2['location']=="Colombia"].head()
df = pd.read_csv('/kaggle/input/uncover/UNCOVER/google_mobility/regional-mobility.csv')
df.keys()
countries = np.unique(df['country'])
dates = df[ (df['country']=="Colombia") & (df['region']=="Total")]['date']
data = []
for j in dates:    
    C = []
    for i in countries:
        cases = np.array(df2['total_cases'][(df2['location']==i)&(df2['date']==j)]).astype(int)
        if cases.size == 0:
            cases = 0
        else:
            cases = cases[0]
        C.append(cases)
    data.append([j,np.array(C).flatten()])
with open('/kaggle/input/world-json/countries.geo.json') as response:
    world = json.load(response)

if video == False:
    N = 1
else:
    N = len(dates)

for i in range(N):
    cases = data[i][1]
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "table"},{"type": "mapbox" }]],column_widths=[0.2, 1], horizontal_spacing=0.01)

    for loc in world['features']:
        loc['id'] = loc['properties']['name']
    fig.add_trace(go.Choroplethmapbox(
                        geojson=world,
                        locations=countries,
                        z=cases,
                        colorscale='thermal',
                        colorbar_title="Cases"), row=1, col=2)

    ii = np.argsort(cases)[::-1]
    fig.add_trace(go.Table(header=dict(values=['Country', 'Cases'],height=20, font=dict(size=14)),
                           cells=dict(values=[countries[ii],cases[ii]],height=20,font=dict(size=12)),columnwidth = [40,40]),row=1,col=1)

    fig.update_layout(mapbox_style="open-street-map",
                      mapbox_zoom=1,
                      mapbox_center = {"lat": 0, "lon": 0},
                      width=1500,
                      height=800,
                      margin=dict(l=0,r=0,b=0,t=50,pad=4),
                      title="Covid-19 World / "+str(np.array(dates)[i]),
                      font=dict(family="Courier New, monospace",size=18,color="#7f7f7f"))


    fig.show()
#     fig.write_image("Figures/World_"+str(round(i,2))+".png")
df = pd.read_csv('/kaggle/input/uncover/UNCOVER/google_mobility/regional-mobility.csv')
df.keys()
fig = plt.figure(figsize=(20,10))
scountries = ['Colombia','Germany','Spain','Brazil','France','Italy','China','Ecuador']

for sc in scountries:
    dates = df[ (df['country']==sc) & (df['region']=="Total")]['date']
    plt.plot(dates,df[ (df['country']==sc) & (df['region']=="Total")]['residential'], label=sc)
    plt.xticks(rotation=90,size=15)
    plt.yticks(size=15)
    plt.title("Residential",size=20)
plt.legend(fontsize=15)
plt.show()
fig = plt.figure(figsize=(20,10))
scountries = ['Colombia','Germany','Spain','Brazil','France','Italy','China','Ecuador']

for sc in scountries:
    dates = df[ (df['country']==sc) & (df['region']=="Total")]['date']
    plt.plot(dates,df[ (df['country']==sc) & (df['region']=="Total")]['workplaces'], label=sc)
    plt.xticks(rotation=90,size=15)
    plt.yticks(size=15)
    plt.title("Workplaces",size=20)
plt.legend(fontsize=15)
plt.show()
gro_phar = np.zeros(len(countries))
tra_stat = np.zeros(len(countries))
workpla = np.zeros(len(countries))
resid = np.zeros(len(countries))
parks = np.zeros(len(countries))

for i, c in enumerate(countries):
    gro_phar[i] = np.average(df['grocery_and_pharmacy'][(df['country']==c) & (df['region']=="Total")])
    tra_stat[i] = np.average(df['transit_stations'][(df['country']==c) & (df['region']=="Total")])
    workpla[i] = np.average(df['workplaces'][(df['country']==c) & (df['region']=="Total")])
    resid[i] = np.nan_to_num(np.average(df['residential'][(df['country']==c) & (df['region']=="Total")]))
    parks[i] = np.average(df['parks'][(df['country']==c) & (df['region']=="Total")])
columns=['country','grocery_pharmacy','transit_station','workplaces','residential','parks']
MEANS = pd.DataFrame(np.array([countries,gro_phar,tra_stat,workpla,resid,parks]).T,columns=columns)
print(MEANS[MEANS['country']=="Colombia"]) #show some examples
print(MEANS[MEANS['country']=="Germany"])
print(MEANS[MEANS['country']=="Japan"])
print(MEANS[MEANS['country']=="Italy"])
df2 = pd.read_csv("/kaggle/input/uncover/UNCOVER/our_world_in_data/coronavirus-disease-covid-19-statistics-and-research.csv")
df2.keys()
new_cases = np.zeros(len(countries))
new_test = np.zeros(len(countries))

for i,c in enumerate(countries):
    new_cases[i] = np.nan_to_num(np.mean(df2['new_cases'][(df2['location']==c)]))
    new_test[i] = np.nan_to_num( np.mean(df2['new_tests'][(df2['location']==c)]))
df4 = pd.read_csv("/kaggle/input/uncover/UNCOVER/HDE/global-school-closures-covid-19.csv")
print("/kaggle/input/uncover/UNCOVER/HDE/global-school-closures-covid-19.csv")
print(df4.keys())
df4[df4['country']=="Germany"].head()
school = np.zeros(len(countries))
for i, c in enumerate(countries):
    ds = np.array(df4['scale'][(df4['country']==c)])
    ds[ds=="Localized"]=0
    ds[ds=="National"]=1
    if np.shape(ds)[0] != 0:
        school[i] = np.mean(ds)
    else:
        school[i] = 0
school = np.array(school)
df4 = pd.read_csv("/kaggle/input/uncover/UNCOVER/worldometer/worldometer-confirmed-cases-and-deaths-by-country-territory-or-conveyance.csv")
print("/kaggle/input/uncover/UNCOVER/worldometer/worldometer-confirmed-cases-and-deaths-by-country-territory-or-conveyance.csv")
print(df4.keys())
df4[df4['country']=="Colombia"].head()
total_cases_pm = np.zeros(len(countries))
serious_cases = np.zeros(len(countries))
active_cases = np.zeros(len(countries))
total_recovered = np.zeros(len(countries))

total_deaths = np.zeros(len(countries))
for i,c in enumerate(countries):
    total_cases_pm[i] = np.mean(df4['total_cases_per_1m_pop'][(df4['country']==c)])
    serious_cases[i] = np.nan_to_num(np.mean(df4['serious_critical_cases'][(df4['country']==c)]))
    active_cases[i] = np.nan_to_num(np.mean(df4['active_cases'][(df4['country']==c)]))
    total_recovered[i] = np.nan_to_num(np.mean(df4['total_recovered'][(df4['country']==c)]))
    total_deaths[i] = np.nan_to_num(np.mean(df4['total_deaths_per_1m_pop'][(df4['country']==c)]))
MEANS['new_cases'] = new_cases
MEANS['new_test'] = new_test
MEANS['total_cases_pm'] = total_cases_pm
MEANS['school'] = school
MEANS['serious_cases'] = serious_cases
MEANS['active_cases'] = active_cases
MEANS['total_recovered'] = total_recovered

MEANS['total_deaths'] = total_deaths
MEANS
x_col = ['country','grocery_pharmacy','transit_station','workplaces',
         'residential','parks','new_cases','new_test','total_cases_pm',
         'school','serious_cases','active_cases','total_recovered']
for i in x_col:
    MEANS = MEANS[MEANS[i].notnull()]
MEANS
MEANS[MEANS['country'] == "Colombia"]
Xc = MEANS[MEANS['country'] =="Colombia"][x_col]
Yc = MEANS[MEANS['country'] =="Colombia"]['total_deaths']

X = MEANS[MEANS['country'] !="Colombia"][x_col]
Y = MEANS[MEANS['country'] !="Colombia"]['total_deaths']
Xc
X
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y, train_size=0.6)

# x_train = X[X['country'] !="Colombia"][x_col]
# y_train = Y[X['country'] !="Colombia"]

# x_test = X[X['country'] =="Colombia"][x_col]
# y_test = Y[X['country'] =="Colombia"]


x_test = x_test.append(Xc)
y_test = y_test.append(Yc)

country_train = x_train['country']
country_test = x_test['country']
x_train = x_train.drop(['country'],axis=1)
x_test = x_test.drop(['country'],axis=1)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
import umap
fig = plt.figure(figsize=(5,5))

XUMAP = MEANS[x_col].drop(['country'],axis=1)

reducer = umap.UMAP(n_neighbors=15,min_dist=0.5,metric='cosine')
reducer.fit(XUMAP)
embedding = reducer.transform(XUMAP)
xmin = min(embedding[:,0])-1
xmax = max(embedding[:,0])+1
ymin = min(embedding[:,1])-1
ymax = max(embedding[:,1])+1            

plt.title("Umap",size=15)
plt.scatter(embedding[:,0], embedding[:,1], cmap='Paired', s=4.0)
plt.grid(alpha=0.3)
plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)
plt.show()
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
md = np.arange(2,200,5)
R2 = np.zeros(len(md))
best = 0
r2 = 0

for i, m in enumerate(md):
    model = DecisionTreeRegressor(max_depth=m,criterion="mae",splitter="random",min_samples_split=3)
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    R2[i] = r2_score(y_test,y_pred)

    if R2[i] > r2:
        best = m
        r2 = R2[i]
            
fig = plt.figure(figsize=(5,5))
plt.subplot(1,1,1)
plt.plot(md,R2)
plt.xlabel("Max Depth")
plt.ylabel(r'$r^2$')
plt.show()

model = DecisionTreeRegressor(max_depth=best,criterion="mae",splitter="random",min_samples_split=3)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

fig = plt.figure(figsize=(20,6))
plt.plot(country_test,y_pred,"*",label="Predicted")
plt.plot(country_test,y_test,"*",label="Truth")
plt.xticks(rotation=90,size=15)
plt.title(r'Desicion Tree Regression  - $r^2$ {}'.format(round(r2,2)),size=30)
plt.ylabel("Total Deaths")
plt.yticks(rotation=0,size=15)
plt.legend(fontsize=15)
plt.show()
print("Colombia will have a number of deaths per million of "+str(y_pred[-1]))
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

fig = plt.figure(figsize=(20,6))
plt.plot(country_test,y_pred,"*",label="Predicted")
plt.plot(country_test,y_test,"*",label="Truth")
plt.xticks(rotation=90,size=15)
plt.yticks(rotation=0,size=15)
plt.title(r'Linear Regression -  $r^2$={}'.format(round(r2_score(y_test,y_pred),2)), size=30)
plt.legend(fontsize=15)
plt.show()
print("Colombia will have a number of deaths per million of population of "+str(y_pred[-1]))
country_test = np.array(country_test)
for i in x_test.keys():
    fig = plt.figure(figsize=(10,10))

    plt.scatter(x_test[i][:-1], y_pred[:-1], cmap='Paired', s=10.0)
    plt.scatter(x_test[i][-1:], y_pred[-1:], cmap='Paired', s=15.0, c="red", )
    for j in range(len(x_test)):
        if country_test[j] == "Colombia":
            plt.text(np.array(x_test[i])[j], y_pred[j], country_test[j] , fontsize=15,color="red")
        else:
            plt.text(np.array(x_test[i])[j], y_pred[j], country_test[j] , fontsize=11,color="blue")
    plt.title("Variable: "+str(i),size=15)
    plt.xlabel(str(i),size=15)
    plt.ylabel("total deaths per million",size=15)
    plt.grid(alpha=0.3)
    plt.show()
