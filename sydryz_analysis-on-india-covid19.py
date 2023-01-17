#!conda install -c conda-forge cufflinks-py
## utility libraries
from IPython.core.display import HTML
from datetime import datetime
from datetime import timedelta


# storing and anaysis
import pandas as pd
import geopandas as gpd
import numpy as np

#Visualization Libraries
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style

import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
from plotly.offline import init_notebook_mode,plot,iplot

import folium
import seaborn as sns

import cufflinks as cf

# Warning
import warnings
warnings.filterwarnings('ignore')


print('Pandas Version' , pd.__version__)
print('Matplotlib Version' , matplotlib.__version__)
print('Plotly Version' , plotly.__version__)
print('Seaborn Version' , sns.__version__)
print('Folium Version' , folium.__version__)
# setting up some setting for libraries
%matplotlib inline
plt.rcParams['figure.figsize'] = 17,8
pyo.init_notebook_mode(connected=True)
cf.go_offline()

#style.use('ggplot')
# color pallette
cnf = '#393e46' # confirmed - grey
dth = '#ff2e63' # death - red
rec = '#21bf73' # recovered - cyan
act = '#fe9801' # active case - yellow
!ls ../input/covid19-corona-virus-india-dataset
#importing data
df = pd.read_csv('../input/covid19-corona-virus-india-dataset/complete.csv',
                parse_dates = ['Date'])

df.tail()
df.columns
df.info()
df_clean = df[['Date', 'Name of State / UT', 'Latitude', 'Longitude', 'Total Confirmed cases', 'Death', 'Cured/Discharged/Migrated']]
df_clean.columns = ['Date', 'State/UT', 'Latitude', 'Longitude', 'Confirmed', 'Deaths', 'Cured']

df_clean['Date'] = df_clean['Date'].dt.normalize()

df_clean['Active'] = df_clean['Confirmed']-(df_clean['Deaths']+df_clean['Cured'])
df_clean['Mortality Rate'] = df_clean['Deaths']/df_clean['Confirmed']
df_clean['Recovery Rate'] = df_clean['Cured']/df_clean['Confirmed']
df_clean.info()

latest = df_clean[df_clean['Date']==max(df_clean['Date'])]


total_confirm = latest['Confirmed'].sum()
total_active = latest['Active'].sum()
total_cured = latest['Cured'].sum()
total_death = latest['Deaths'].sum()


now  = datetime.now().strftime("%B %d, %Y")

print(u"\u2022",f'Total Number of Confirmed Covid 2019 Cases across India till date ({now}):', total_confirm)
print(u"\u2022",f'Total Number of Active Cases till date ({now}):', total_active)
print(u"\u2022",f'Total Number of Cured Cases across India till date ({now}):', total_cured)
print(u"\u2022",f'Total Number of Deaths across India till date ({now}):', total_death)
tm = latest.melt(id_vars="Date", value_vars=['Active', 'Deaths', 'Cured'])
tm.head()
fig = px.treemap(tm, path=["variable"], values="value",height=250, width=800,
                 color_discrete_sequence=[act, rec, dth], title='Latest Stats')

fig.data[0].textinfo = 'label+value+text'
fig.show()
latest.head()
temp = latest.groupby(by = ['State/UT']).sum()

temp.tail()
temp = temp[['Confirmed','Deaths','Cured','Active','Mortality Rate','Recovery Rate']]
temp.sort_values('Confirmed',ascending=False,inplace = True)
#temp.head()

temp.style\
    .background_gradient(cmap="Blues", subset=['Active','Confirmed'])\
    .background_gradient(cmap="Greens", subset=['Cured', 'Recovery Rate'])\
    .background_gradient(cmap="Reds", subset=['Deaths', 'Mortality Rate'])
temp.columns
#Visualization
temp_1 = temp[['Confirmed', 'Deaths', 'Cured']]

temp_1.iplot(kind = 'bar',xTitle= 'State/UT' , yTitle='Numbers of Cases',mode = 'markers+lines',
            title = f'Cases State Wise on {now}')
temp_2 = temp[['Mortality Rate','Recovery Rate']]
               
temp_2.iplot(kind ='scatter',xTitle='State/UT',yTitle='Avrage',title = f'Mortality and Recovery Rate on {now}',
             mode = 'markers', size = 5)
# Date wise data visualization whole country

temp = df_clean.groupby(by = ['Date']).sum()
temp.drop(['Latitude','Longitude','Mortality Rate','Recovery Rate'],axis=1,inplace=True)

temp.tail()
temp.iplot(title = 'Covid-19 Growth in India', yTitle='Cases',size=5,mode='markers+lines')
#df_clean.columns
cases_df = df_clean.groupby('Date')['Confirmed', 'Deaths'].sum()

filt_cnf = (cases_df['Confirmed'] >= 50)

temp = cases_df[filt_cnf].diff().dropna()

temp
col_y = ['Confirmed','Deaths']
colr = [cnf,dth]

for i,x in enumerate(col_y):
    temp.iplot(kind = 'scatter',mode = "markers+lines" ,size = 5,y = col_y[i],color=colr[i],yTitle=col_y[i],title=f'New {col_y[i]} Cases after Crossing 50 Confirmed Cases')
top_10 = latest.groupby(by = ['State/UT']).agg({'Confirmed': 'sum', 'Deaths': 'sum', 'Cured' : 'sum', 'Active' : 'sum'})\
                .nlargest(10,['Confirmed','Deaths','Cured','Active'])

top_10
#Creating Figures
plot_c = px.bar(top_10.sort_values('Confirmed') ,x="Confirmed",y = top_10.sort_values('Confirmed').index,
               text='Confirmed', orientation='h', color_discrete_sequence = [cnf])

plot_d = px.bar(top_10.sort_values('Deaths'),x="Deaths",y = top_10.sort_values('Deaths').index,
               text='Deaths', orientation='h', color_discrete_sequence = [dth])

plot_r = px.bar(top_10.sort_values('Cured'),x="Cured",y = top_10.sort_values('Cured').index,
               text='Cured', orientation='h', color_discrete_sequence = [rec])

plot_a = px.bar(top_10.sort_values('Active'),x="Active",y = top_10.sort_values('Active').index,
               text='Active', orientation='h', color_discrete_sequence = [act])


# plot
fig = make_subplots(rows=2, cols=2, shared_xaxes=False, horizontal_spacing=0.14, vertical_spacing=0.08,
                    subplot_titles=('Confirmed cases', 'Deaths reported', 'Recovered', 'Active cases'))

fig.add_trace(plot_c['data'][0],row=1, col=1)
fig.add_trace(plot_d['data'][0],row=1, col=2)
fig.add_trace(plot_r['data'][0],row=2, col=1)
fig.add_trace(plot_a['data'][0],row=2, col=2)

fig.update_layout(height=600 ,title_text="Top 10 States ")

HTML('<div class="flourish-embed flourish-bar-chart-race" data-src="visualisation/1977187" data-url="https://flo.uri.sh/visualisation/1977187/embed"><script src="https://public.flourish.studio/resources/embed.js"></script></div>')
df_clean.columns
#Deaths,Cured ,Active Cases Date Wise

col = ['Deaths','Cured','Active']

for i,val in enumerate(col):
    p_df = pd.pivot_table(df_clean,index  = 'Date', values = val, columns ='State/UT').fillna(0).astype('int').reset_index()
    p_df.iplot(x = 'Date' ,title = col[i], xTitle = 'Date',yTitle = 'Cases')
latest.columns
geo_map = folium.Map([20.5937,78.9629],zoom_start=4,tiles ='cartodbpositron' )

for lat,long,active,deaths,cured,name in zip(latest['Latitude'],latest['Longitude'],\
                                             latest['Active'],latest['Deaths'],\
                                             latest['Cured'],latest['State/UT']):

    folium.CircleMarker([lat,long],radius=active*0.005\
                       ,tooltip = (f'''<strong>name</strong>: {str(name).capitalize()} <br>
                               <strong>Active</strong>: {str(active)}<br>
                               <strong>Deaths</strong>: {str(deaths)}<br>
                               <strong>Cured</strong>: {str(cured)}<br>''')\
                       ,color = 'red',fill_color = 'red',fill_opacity=0.3).add_to(geo_map)

geo_map

grp_states = df_clean.groupby('Date')['State/UT']
affected_states = grp_states.unique().apply(len).values
#affected_states


dates = grp_states.unique().apply(len).index
#dates
fig = go.Figure()

fig.add_trace(go.Scatter(x=dates, y=[36 for i in range(len(affected_states))], 
                         mode='lines', name='Total no. of States+UT', 
                         line = dict(color='#222831', dash='longdashdot')))

fig.add_trace(go.Scatter(x=dates, y=affected_states, hoverinfo='x+y',
                         mode='lines', name='No. of affected States+UT', 
                         line = dict(color='#c70039')))

fig.update_layout(title='No. of affected States/UT over Time', 
                  xaxis_title='Dates', yaxis_title='No. of affected States/UT')
fig.show()
latest_cnf_dth = latest[latest['Confirmed']>10]
latest_cnf_dth = latest_cnf_dth[['State/UT','Confirmed','Deaths']]

px.scatter(latest_cnf_dth,x = 'Confirmed', y = 'Deaths', color ='State/UT', size = "Confirmed",log_x=True, title = 'Confirmed vs Deaths')


# Reading Popolation Data

pop2018 = pd.read_csv('../input/covid19-corona-virus-india-dataset/pop2018.csv')

pop2018.rename(columns = {'State': 'State/UT'},inplace = True)
pop2018.dtypes

pop2018['State/UT'].replace('Telangana', 'Telengana', inplace=True)
pop2018['State/UT'].replace('Jammu & Kashmir', 'Jammu and Kashmir', inplace=True)
pop2018['State/UT'].replace('A.& N.Islands', 'Andaman and Nicobar Islands', inplace=True)
pop2018['State/UT'].replace('D.& N.Haveli', 'Dadar Nagar Haveli', inplace=True)


pop2018.head(36)
# after 2018 D & N Haveli and Daman And Diu Became one so Combinig these with one value
pop2018.loc[33,'2018'] =  pop2018.loc[33,'2018']+ pop2018.loc[34,'2018']   

# droping Daman & Diu
pop2018.drop(34, inplace = True)
state = list(np.setdiff1d(latest['State/UT'],pop2018['State/UT']))
print(state, 'is not in Population DataFrame')
# adding Ladakh poluation Manually
pop2018.loc[36] = ['Ladakh' , 290492]
pop2018.tail()
latest.shape
pop_vs_cnf = pd.merge(latest,pop2018, on = 'State/UT')
pop_vs_cnf.shape
pop_vs_cnf
grp = pop_vs_cnf[['State/UT','2018', 'Confirmed']].set_index('State/UT')
grp['2018'] = np.log10(grp['2018'])
grp['Confirmed'] = np.log10(grp['Confirmed'])

grp = grp.rename(columns = {'2018':'Estimated Population 2018' })
grp.head()
grp.iplot(kind = 'bar' ,barmode  = 'overlay', title = 'Estimated Population 2018 Vs Confirmed', yTitle = 'Log Base 10')
pop_vs_cnf['Percentage'] = (pop_vs_cnf.loc[:,'Confirmed'].values*100)/ pop_vs_cnf.loc[:,'2018'].values
pop_vs_cnf.head(3)
grf = pop_vs_cnf[['State/UT','Percentage']].set_index('State/UT')

grf.iplot(title='Percentage Population Effected',yTitle = 'Percentage')
pt_df = pd.read_csv('../input/covid19-corona-virus-india-dataset/patients_data.csv')

pt_df.head()
pt_df['date_announced'] = pd.to_datetime(pt_df['date_announced'], format='%d/%m/%Y')
pt_df['status_change_date'] = pd.to_datetime(pt_df['status_change_date'], format='%d/%m/%Y')
pt_df.info()
print(pt_df.shape)
dist = pt_df.groupby(['detected_state', 'detected_district'])['patient_number'].count().reset_index()
dist.head()
fig = px.treemap(dist, path=["detected_state", "detected_district"], values="patient_number", height=700,
           title='Number of Confirmed Cases', color_discrete_sequence = px.colors.qualitative.Prism)
fig.data[0].textinfo = 'label+text+value'
fig.show()
map_data = gpd.read_file('../input/india-district-wise-shape-files/output.shp')
map_data.head(2)
#State Wise Grouping Data
states = map_data.dissolve(by='statename').reset_index()
states.head()
states['statename'].unique()
pt_st = pt_df.groupby('detected_state')['patient_number'].count().reset_index()
pt_st['detected_state'].unique()
np.setdiff1d(pt_st['detected_state'],states['statename'])
states['statename'] = states['statename'].str.replace('&', 'and')
states['statename'] = states['statename'].str.replace('NCT of ', '')
states['statename'] = states['statename'].str.replace('Chhatisgarh', 'Chhattisgarh')
states['statename'] = states['statename'].str.replace('Orissa', 'Odisha')
states['statename'] = states['statename'].str.replace('Pondicherry', 'Puducherry')
states['statename'] = states['statename'].str.replace('Dadra and Nagar Haveli', 'Dadra and Nagar Haveli and Daman and Diu')

print(states.shape)
np.setdiff1d(pt_st['detected_state'],states['statename'])
pt_st.head(3)
pt_st.columns = ['state', 'count']
print(pt_st.shape)
pt_st.head(2)
state_map = pd.merge(states, pt_st, left_on='statename', right_on='state', how='right')
print(state_map.shape)
state_map.tail(2)
state_map['distarea'] = state_map['distarea'].fillna(0)
state_map.isnull().sum()
#folium.Choropleth?
s = folium.Map(location=[23, 78.9629], tiles='cartodbpositron',
               min_zoom=4, max_zoom=6, zoom_start=4)

choropleth = folium.Choropleth(state_map,data = state_map, columns=['statename','count'],
                  key_on ='feature.properties.statename',
                 fill_color='YlOrRd',
                 line_weight=0.1,
                 line_opacity=0.5,
                 legend_name='No. of reported cases').add_to(s)

choropleth.geojson.add_child(
    folium.features.GeoJsonTooltip(fields=['statename','count'],aliases=['State Name', 'Cases'])
).add_to(s)

folium.LayerControl().add_to(s)
s
map_data.columns
pt_df.columns
## District Wise Patient Data

pt_dst = pt_df.groupby('detected_district')['patient_number'].count().reset_index()
pt_dst.columns = ['district', 'count']

print(pt_dst.shape)
pt_dst.head(2)
ind_dist = pd.merge(map_data,pt_dst, right_on='district', left_on='distname', how='left')
ind_dist.isnull().sum()
#droping missing values from patient count

ind_dist = ind_dist[ind_dist['count'].notnull()]
ind_dist.info()
#data type convertion

ind_dist['count'] = ind_dist['count'].astype('int32')
ind_dist.info()
#ind_dist['count'].max()
bins = [0, 50, 200, 500, 1250, 3500]

d = folium.Map(location=[23, 78.9629], tiles='cartodbpositron',
               min_zoom=4, max_zoom=6, zoom_start=4)

choropleth = folium.Choropleth(ind_dist,data = ind_dist, columns=['distname','count'],
                  key_on ='feature.properties.distname',
                 fill_color='YlOrRd',
                 line_weight=0.1,
                 line_opacity=0.5,
                 bins=bins,
                 legend_name='No. of reported cases').add_to(d)

choropleth.geojson.add_child(
    folium.features.GeoJsonTooltip(fields=['distname','count'],aliases=['District Name', 'Cases'])
).add_to(d)

folium.LayerControl().add_to(d)
d
pt_df.columns
#missing values in date announced column
#pt_df.shape[0]-pt_df['date_announced'].dropna().shape[0]
total_values = pt_df.shape[0]
missing_values_in_announced_dates = (pt_df.shape[0]) - (pt_df['date_announced'].dropna().shape[0])
availabe_values = total_values - missing_values_in_announced_dates

print('\u2022 Total no. of values :', total_values)
print('\u2022 No. of missing values :', missing_values_in_announced_dates)
print('\u2022 No. of available values :', availabe_values)
#cf.colors.scales()
new_cases = pt_df.groupby('date_announced')[['date_announced']].count()
new_cases.iplot(kind= 'bar', xTitle = 'Dates',yTitle = 'Count',title='No. of cases reported each day', colorscale= 'set1')
total_values = pt_df.shape[0]
missing_values_in_age_bracket = (pt_df.shape[0]) - (pt_df['age_bracket'].dropna().shape[0])
availabe_values = total_values - missing_values_in_announced_dates

print('\u2022 Total no. of values :', total_values)
print('\u2022 No. of missing values :', missing_values_in_age_bracket)
print('\u2022 No. of available values :', availabe_values)
#cf.getThemes() 
pt_df['age_bracket'].iplot(kind = 'histogram', histfunc ='count', bins=70,
                           title='Distribution of ages of confirmed patients', yTitle = 'Count', xTitle = 'Age Bracket',
                           theme = 'ggplot',colors= '#555555', linecolor = '#555555' )
def data_distribution_pie_chart(values,colors,names,hole=0.4,x=0.5,y=0.1):
    
    '''function To Visualize Missing data and Available Data'''
    
    fig = px.pie(values=values,color_discrete_sequence = colors, names = names,hole= hole)
    fig.update_layout(title={'text': 'Data Distribution',
                         "font":{'size':14},
                         'y':y,'x':x,
                        'xanchor':'center'},
                  autosize=False,
                  width=900,
                  height=400)
    fig.update_traces(textposition='outside', textinfo='label+value+percent')

    fig.show()
c_vs_g = pt_df[['p_id','gender']]
c_vs_g['gender'].unique()
c_vs_g['gender'] = c_vs_g['gender'].replace('Non-Binary' , np.nan)
c_vs_g['gender'].unique()
temp = c_vs_g.copy().dropna()
total = c_vs_g.shape[0]
missing = c_vs_g.shape[0] - temp.shape[0]
available = temp.shape[0]

print('\u2022 Total no. of values :', total)
print('\u2022 No. of missing values :', missing)
print('\u2022 No. of available values :', available)

# Visualtization

value = [missing,available]
colors = ['#74264d','#b678ad']
names = ['Missing','Available']
hole= 0.5

data_distribution_pie_chart(value,colors,names,hole)
temp.sample(5)
temp['gender'] = temp['gender'].replace('M' , 'Male')
temp['gender'] = temp['gender'].replace('F' , 'Female')
temp.head()
gen_value = temp.groupby('gender').count()['p_id'].tolist()

fig = px.pie(values=gen_value,color_discrete_sequence = [ '#edbf4a','#2a4158'], names = ['Female','Male'],hole= 0.4)

fig.update_layout(title={'text': 'Gender Distribution in Confirmed Cases <br>(Sample Size 7153)',
                         "font":{'size':10},
                         'y':0.1,'x':0.5,
                         'xanchor': 'center','yanchor': 'bottom'},
                  autosize=False,
                  width=900,
                  height=400)

fig.update_traces(textposition='outside', textinfo='label+value+percent')

fig.show()

a_vs_g = pt_df[['age_bracket','gender']]
temp = a_vs_g.copy().dropna()
total = a_vs_g.shape[0]
missing = a_vs_g.shape[0] - temp.shape[0]
available = temp.shape[0]

print('\u2022 Total no. of values :', total)
print('\u2022 No. of missing values :', missing)
print('\u2022 No. of available values :', available)

# Visualtization

value = [missing,available]
colors = ['#8c9ea3','#2a4158']
names = ['Missing','Available']
hole= 0.5

data_distribution_pie_chart(value,colors,names,hole)
temp.sample(5)
gen_distrb = temp.groupby('gender').count()['age_bracket'].tolist()
#gen_distrb

fig = make_subplots(rows=1, cols = 2,
                    column_widths=[0.8, 0.2],
                    subplot_titles= ['Gender vs Age'],
                    specs=[[{"type": "histogram"}, {"type": "pie"}]])

fig.add_trace(go.Histogram(x=temp[temp['gender']=='F']['age_bracket'], nbinsx=50, name='Female', marker_color='#9171b0'), 1, 1)
fig.add_trace(go.Histogram(x=temp[temp['gender']=='M']['age_bracket'], nbinsx=50, name='Male', marker_color='#5a3f70'), 1, 1)

fig.add_trace(go.Pie(values=gen_distrb,hole= 0.3, labels=['Female','Male'], marker_colors = [ '#9171b0','#5a3f70']),1, 2)

fig.update_layout(showlegend=False)
fig.update_layout(barmode='stack', xaxis_title_text = 'Age Bins', yaxis_title_text = 'Count')
fig.data[2].textinfo = 'label+text+value+percent'

fig.show()

c_vs_g = pt_df[['age_bracket','current_status']]
temp1 = c_vs_g.copy().dropna()
total = c_vs_g.shape[0]
missing = c_vs_g.shape[0] - temp1.shape[0]
available = temp1.shape[0]

print('\u2022 Total no. of values :', total)
print('\u2022 No. of missing values :', missing)
print('\u2022 No. of available values :', available)

# Visualtization

values=[missing,available]
color = ['#618691','#fecb01']
names = ['Missing','Available']
hole= 0.4

data_distribution_pie_chart(values,color,names,hole)
temp1.sample(5)
cur_stats_dist = temp1.groupby('current_status').count()['age_bracket'].tolist()
#cur_stats_dist
labels =  temp1.groupby('current_status').count().index.tolist()
#labels
fig = make_subplots(rows=1, cols = 2,
                    column_widths=[0.8, 0.2],
                    subplot_titles= ['Case Status vs Age'],
                    specs=[[{"type": "histogram"}, {"type": "pie"}]])

fig.add_trace(go.Histogram(x=temp1[temp1['current_status']=='Deceased']['age_bracket'], nbinsx=50, name='Deceased', marker_color='#fd0054'), 1, 1)
fig.add_trace(go.Histogram(x=temp1[temp1['current_status']=='Recovered']['age_bracket'], nbinsx=50, name='Recovered', marker_color='#40a798'), 1, 1)
fig.add_trace(go.Histogram(x=temp1[temp1['current_status']=='Hospitalized']['age_bracket'], nbinsx=50, name='Hospitalized', marker_color='#5a3f70'), 1, 1)

fig.add_trace(go.Pie(values=cur_stats_dist,hole= 0.3, labels=labels, marker_colors = ['#fd0054','#5a3f70','#40a798']),1, 2)

fig.update_layout(showlegend=False)
fig.update_layout(barmode='stack', xaxis_title_text = 'Age Bins', yaxis_title_text = 'Count')
fig.data[3].textinfo = 'label+text+value+percent'

fig.show()
pt_df.columns
pt_df['type_of_transmission'].unique()
pt_df['type_of_transmission'] = pt_df['type_of_transmission'].replace('Imported ', 'Imported')
pt_df['type_of_transmission'] = pt_df['type_of_transmission'].replace('Unknown', 'TBD')

tot = pt_df[['type_of_transmission']]
temp = pt_df[['type_of_transmission']].dropna()
total = tot.shape[0]
missing = tot.shape[0] - temp.shape[0]
available = temp.shape[0]

print('\u2022 Total no. of values :', total)
print('\u2022 No. of missing values :', missing)
print('\u2022 No. of available values :', available)

values = [ missing,available]
color = ['#1d2b3f','#688cc3']
names = ['Missing','Available']
hole= 0.4
x = 0.5
y = 0.05

data_distribution_pie_chart(values,color,names,hole,x,y)
temp1 = temp.copy().groupby('type_of_transmission')[['type_of_transmission']].count()
temp1.columns = ['Count']
values = temp1['Count'].tolist()
label = temp1.index.tolist()
fig = px.pie(values= values, names = label, labels = label, color_discrete_sequence = ['#243448'], hole = 0.5)

fig.update_layout(autosize=False,width=900,height=400)
fig.update_traces(title='Type of Transmission',textposition='inside', textinfo='label+value+percent')


fig.show()
temp = pt_df.groupby('nationality')['patient_number'].count().reset_index()
temp = temp.sort_values('patient_number')
temp = temp[temp['nationality']!='India']
fig = px.bar(temp, x='patient_number', y='nationality', orientation='h', text='patient_number', width=600,
       color_discrete_sequence = ['#243448'], title='No. of Foreign Citizens')
fig.update_xaxes(title='')
fig.update_yaxes(title='')
fig.show()
pt_df.columns
zn = pd.read_csv('../input/covid19-corona-virus-india-dataset/zones.csv')
zn.isnull().sum()
zn.nunique()
zn= zn.dropna()
dst= zn.groupby('zone')["district"].count().reset_index()
dst
dst.columns = ['Zone', 'Count']
dst.sort_values("Count",inplace = True)
dst.head()
orng = '#ff6500'
grn = '#3CB371'
red = '#FF0000'
fig_1 = px.treemap(dst, path=['Zone'], values="Count", height=250, width=800,
                 color_discrete_sequence=[orng, grn, red], title='Zone Distribution on District Level')
fig_1.data[0].textinfo = 'label+text+value'
fig_1.show()
st = zn.groupby(['state','zone'])[['district']].count().reset_index()
st.head()
fig_1 = px.treemap(st,path = ['state','zone'],values="district",
                 color_discrete_sequence = px.colors.qualitative.G10, title='Zone Distribution on State Level')
fig_1.data[0].textinfo = 'label+text+value'
fig_1.show()
HTML('''<iframe title="Geographical Map  District Wise Zone" aria-label="Map" id="datawrapper-chart-jzl6U" src="https://datawrapper.dwcdn.net/jzl6U/6/" scrolling="no" frameborder="0" style="width: 0; min-width: 100% !important; border: none;" height="400"></iframe><script type="text/javascript">!function(){"use strict";window.addEventListener("message",(function(a){if(void 0!==a.data["datawrapper-height"])for(var e in a.data["datawrapper-height"]){var t=document.getElementById("datawrapper-chart-"+e)||document.querySelector("iframe[src*='"+e+"']");t&&(t.style.height=a.data["datawrapper-height"][e]+"px")}}))}();
</script>''')




