import numpy as np
import pandas as pd
from IPython.display import Image
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go 
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.externals.six import StringIO  
#import pydotplus
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.metrics import mean_absolute_error as mae
from sklearn.tree import DecisionTreeRegressor as dtr
input_file = "../input/globalterrorismdb_shorter.csv"
df = pd.read_csv(input_file, header = 0,usecols=['iyear', 'imonth', 'iday', 'extended', 'country', 'country_txt', 'region', 'latitude', 'longitude','success', 'suicide','attacktype1','attacktype1_txt', 'targtype1', 'targtype1_txt', 'natlty1','natlty1_txt','weaptype1', 'weaptype1_txt' ,'nkill','multiple', 'individual', 'claimed','nkill','nkillter', 'nwound', 'nwoundte'])
df.head(10)
df.info()
df_USA = df[df.country == 217]
df_USA.head()
df_USA.info()
df_USA = df_USA.drop([ 'region', 'country', 'country_txt','claimed', 'nkillter', 'nwoundte'],axis=1)
df_USA.tail()
df_USA.describe()
df_USA.plot(kind= 'scatter', x='longitude', y='latitude', alpha=0.4, figsize=(16,7))
plt.show()
df_USA.nsmallest(6,"longitude")
df_USA.nlargest(6,"longitude")
df_USA = df_USA[df_USA.longitude < 0]
df_USA.nlargest(6,"longitude")
df_USA.plot(kind= 'scatter', x='longitude', y='latitude', alpha=0.4, figsize=(16,9))
plt.show()
df_USA.plot(kind= 'scatter', x='longitude', y='latitude', alpha=1.0,  figsize=(18,6),  
               s=df_USA['nkill']*3, label= 'Nr of casualties', fontsize=1, c='nkill', cmap=plt.get_cmap("jet"), colorbar=True)
plt.ylabel("Latitude", fontsize=14)
plt.xlabel("Longitude", fontsize=14)
plt.show()
terror_peryear = np.asarray(df_USA.groupby('iyear').iyear.count())
successes_peryear = np.asarray(df_USA.groupby('iyear').success.sum())

terror_years = np.arange(1970, 2016)

trace1 = go.Bar(x = terror_years, y = terror_peryear, name = 'Nr of terrorist attacks')

trace2 = go.Scatter(x = terror_years, y = successes_peryear, name = 'Nr of succesful terrorist attacks', line = dict(color = ('rgb(205, 12, 24)'),width=5))

layout = go.Layout(title = 'Terrorist Attacks by Year in USA (1970-2016)', legend=dict(orientation="h"),
         barmode = 'group')

figure = dict(data = [trace1,trace2], layout = layout)
iplot(figure)
attacks_per_type = (df_USA.groupby('attacktype1_txt').attacktype1_txt.count())
successes_per_type = (df_USA.groupby('attacktype1_txt').success.sum())
print(attacks_per_type, successes_per_type)
trace2 = go.Bar(
    y=['Unknown','Hijacking','Hostage Taking (Kidnapping)','Unarmed Assault','Hostage Taking (Barricade Incident)','Assassination','Armed Assault','Facility/Infrastructure Attack','Bombing/Explosion'],
    x=[11,17,20,56,59,128,249,836,1377],
    name='Nr of terrorist attacks',
    orientation = 'h',
    marker = dict(color = 'rgb(255,140,0)'))

trace1 = go.Bar(
    y=['Unknown','Hijacking','Hostage Taking (Kidnapping)','Unarmed Assault','Hostage Taking (Barricade Incident)','Assassination','Armed Assault','Facility/Infrastructure Attack','Bombing/Explosion'],
    x=[8,15,20,31,56,80,233,748,1080],
    name='Nr of successful terrorist attacks',
    orientation = 'h',
    marker = dict(color = 'rgb(0,200,200)'))
data = [trace1, trace2]
layout = go.Layout(
    legend=dict(x=0.5, y=0.5), # placing legend in the middle
    title = 'Terrorist attacks in USA 1970-2016 <br>by Type',
    barmode='group',
    bargap=0.1,
    bargroupgap=0,
    autosize=False,
    width=1000,
    height=500,
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
terror_peryear = np.asarray(df_USA.groupby('iyear').iyear.count())
affiliated_attacks_peryear = np.asarray(df_USA.groupby('iyear').individual.sum())
percentage = affiliated_attacks_peryear / terror_peryear * 100

terror_years = np.arange(1970, 2016)

trace1 = go.Bar(x = terror_years, y = terror_peryear,name = 'Terrorist attacks')

trace2 = go.Scatter(x = terror_years, y = affiliated_attacks_peryear,name = 'Terrorist attacks by people affiliated with terrorist organisations',yaxis = "y2")

trace3 = go.Scatter(x = terror_years, y = percentage, name = "Percentage of terrorist attacks carried out by people affiliated with terrorist organisations", yaxis  = "y3")

data = [trace1,trace2,trace3]

layout = go.Layout(
    title='Rise of Terrorist groups in the USA',
    yaxis1=dict(title='Terrorist attacks',showline = False,showgrid=False),
    yaxis2=dict(
        titlefont=dict(
            color='rgb(148, 103, 189)'
        ),
        showgrid=False,
        zeroline= False,
        showline=False,
        ticks="",
        showticklabels=False,
        overlaying='y',
        autorange = True,
        side='right'),
    yaxis3 = dict(
        titlefont = dict(
            color = "rgb(124,252,0)"
        ),
        showgrid=False,
        zeroline= False,
        showline=False,
        ticks = "",
        showticklabels=False,
        overlaying = "y",
        autorange = True,
        side = "right"
    ),
    legend=dict(orientation="h")
)

figure = go.Figure(data = data, layout = layout)
iplot(figure)
df_USA.info()
df_USA.describe()
df_USA['nkill'].fillna(1.361194, inplace=True)
df_USA['nwound'].fillna(6.802632, inplace=True)
df_USA['latitude'].fillna(36.683652, inplace=True)
df_USA['longitude'].fillna(-92.125972, inplace=True)
df_USA["natlty1"].fillna(217, inplace=True)
df_USA["natlty1_txt"].fillna("United States", inplace=True)
df_USA.info()
df_USA.corr()
corrmat = df_USA.corr()
f, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(corrmat, vmax=1, square=True);
plt.show()
df_USA = df_USA.drop(["iyear","attacktype1_txt","targtype1_txt", "weaptype1_txt", "natlty1_txt"], axis=1)
y = df_USA["success"]
features_success = ['imonth', 'iday', 'extended', 'latitude', 'longitude', 'multiple', 'suicide', 'attacktype1', 'targtype1', 'natlty1','individual', 'weaptype1', 'nkill', 'nwound']
X = df_USA[features_success]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=1)
terrorism_success_model_depth_leaves = tree.DecisionTreeClassifier(random_state = 1, max_depth = 3, max_leaf_nodes = 10)
terrorism_success_model_depth_leaves.fit(X_train, y_train)
success_pred_depth_leaves = terrorism_success_model_depth_leaves.predict(X_val)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_val,success_pred_depth_leaves))
print(confusion_matrix(y_val,success_pred_depth_leaves))
terrorism_success_model_depth = tree.DecisionTreeClassifier(random_state = 1, max_depth = 3)
terrorism_success_model_depth.fit(X_train, y_train)
success_pred_depth = terrorism_success_model_depth.predict(X_val)
print(classification_report(y_val,success_pred_depth))
print(confusion_matrix(y_val,success_pred_depth))
terrorism_success_model_leaves = tree.DecisionTreeClassifier(random_state = 1, max_leaf_nodes = 10)
terrorism_success_model_leaves.fit(X_train, y_train)
success_pred_leaves = terrorism_success_model_leaves.predict(X_val)
print(classification_report(y_val,success_pred_leaves))
print(confusion_matrix(y_val,success_pred_leaves))
terrorism_success_model_bare = tree.DecisionTreeClassifier(random_state = 1)
terrorism_success_model_bare.fit(X_train, y_train)
success_pred_bare = terrorism_success_model_bare.predict(X_val)
print(classification_report(y_val,success_pred_bare))
print(confusion_matrix(y_val,success_pred_bare))
rf = RandomForestClassifier(n_estimators=100) 
rf = rf.fit(X_train, y_train)
rf_pred = rf.predict(X_val)
print(classification_report(y_val,rf_pred))
print(confusion_matrix(y_val,rf_pred))
for name, score in zip(X_train, rf.feature_importances_):
    print(name, score)
data = go.Bar(
    y=['extended','suicide', 'individual',  'multiple', 'nkill',"nwound","natlty1",'weaptype1',  'attacktype1','targtype1', 
       'imonth', 'longitude', 'iday', 'latitude'],
    x=[0.0008434176541557457, 0.0014349551050431954,0.01660395793541506,0.021650941814853716,
       0.026608046715114467,0.032573556495003576,0.042784340039457504,0.046320397204197755,
       0.0754779145247244,0.10563968568238052,0.1245681748749377,0.1588732207915285,
       0.16474122932796273,0.18188016183522507],   
    orientation = 'h',
    marker = dict(color = 'rgba(255,0,0, 0.6)', line = dict(width = 0.5)))

data = [data]
layout = go.Layout(title = 'Relative Importance of the Features in the Random Forest',
    barmode='group', bargap=0.1, width=800,height=500,)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
features_nkill = ['imonth', 'iday', 'extended', 'latitude', 'longitude', 'multiple','success', 'suicide', 'attacktype1', 'targtype1', 'natlty1','individual', 'weaptype1', 'nwound']
X_nkill = df_USA[features_nkill]
y_nkill = df_USA["nkill"]
X_nkill_train, X_nkill_val, y_nkill_train, y_nkill_val = train_test_split(X_nkill, y_nkill, test_size=0.20, random_state=1)
nkill_model = dtr(random_state = 1)
nkill_model.fit(X_nkill_train,y_nkill_train)
nkill_pred = nkill_model.predict(X_nkill_val)
print(mae(y_nkill_val, nkill_pred))
nkill_model = dtr(random_state = 1, max_leaf_nodes = 27)
nkill_model.fit(X_nkill_train,y_nkill_train)
nkill_pred = nkill_model.predict(X_nkill_val)
print(mae(y_nkill_val, nkill_pred))
nkill_model = dtr(random_state = 1, max_depth = 15)
nkill_model.fit(X_nkill_train,y_nkill_train)
nkill_pred = nkill_model.predict(X_nkill_val)
print(mae(y_nkill_val, nkill_pred))
nkill_model = dtr(random_state = 1, max_depth = 15, max_leaf_nodes = 27)
nkill_model.fit(X_nkill_train,y_nkill_train)
nkill_pred = nkill_model.predict(X_nkill_val)
print(mae(y_nkill_val, nkill_pred))
nkill_model = rfr(random_state = 1, n_estimators = 10)
nkill_model.fit(X_nkill_train,y_nkill_train)
nkill_pred = nkill_model.predict(X_nkill_val)
print(mae(y_nkill_val, nkill_pred))
for name, score in zip(X_nkill_train, nkill_model.feature_importances_):
    print(name, score)
features_nkill = ['imonth', 'iday', 'extended', 'latitude', 'longitude', 'multiple','success', 'suicide', 'attacktype1', 'targtype1', 'natlty1','individual', 'weaptype1']
X_nkill = df_USA[features_nkill]
y_nkill = df_USA["nkill"]
X_nkill_train, X_nkill_val, y_nkill_train, y_nkill_val = train_test_split(X_nkill, y_nkill, test_size=0.20, random_state=1)
nkill_model = rfr(random_state = 1, n_estimators = 100)
nkill_model.fit(X_nkill_train,y_nkill_train)
nkill_pred = nkill_model.predict(X_nkill_val)
print(mae(y_nkill_val, nkill_pred))
sorted_importance_dict = sorted(zip(X_nkill_train, nkill_model.feature_importances_), key=lambda x: x[1])
print(sorted_importance_dict)

data = go.Bar(
    y=['extended','multiple','success','weaptype1','natlty1','iday','imonth','attacktype1',
       'individual','targtype1','latitude','suicide','longitude'],
    x=[6.229373475964622e-06, 0.00018061437686766628,0.00022199675803425399,0.0007881470118179646,
       0.0017783660577080336,0.00888759926303102,0.012838555852620923,0.018069844219501384,
       0.06589093075573221,0.09836071796324783,0.10878349610841871,0.271867786077219,0.41232571618232533],   
    orientation = 'h',
    marker = dict(color = 'rgba(255,0,0, 0.6)', line = dict(width = 0.5)))

data = [data]
layout = go.Layout(title = 'Relative Importance of the Features in the Random Forest',
    barmode='group', bargap=0.1, width=800,height=500,)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
nkill_model = rfr(random_state = 1, n_estimators = 10)
nkill_model.fit(X_nkill,y_nkill)

month = 9
day = 11
extended = 0
latitude = 40.711675 
longitude = -70.013285
multiple = 1
success = 1
suicide = 1
attackType = 8
targetType = 11
natlty1 = 217
individual = 1
weaponType = 12

kill_count = nkill_model.predict([[month,day,extended,latitude,longitude,multiple,success,suicide,
                                  attackType,targetType,individual,weaponType, natlty1]])
print("Unfortunately, this attack will kill "+str(int(kill_count[0]))+" people...")