import numpy as np 

import pandas as pd 

import plotly.express as px



import warnings

warnings.filterwarnings('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df = pd.read_csv('../input/covid19-hist-data/COVID-19-geographic-disbtribution-worldwide-2020-06-07.csv', delimiter = ';', engine='python')

df['Date'] = pd.to_datetime(df[['day','month', 'year']])

df = df[df.continentExp != 'Oceania']

df.dropna(inplace = True)

display(df.head())

display(df.info())
Data_to_plot = df[(df['Date'] == df['Date'].max()) & (df['deaths'] > 0) & (df['cases']> 0)]



fig = px.scatter(Data_to_plot, x="cases", y="deaths",)

fig.update_layout(title="Scatter 2",xaxis_title="Cases (log)",yaxis_title="Deaths (log)")

fig2 = px.scatter(Data_to_plot, x="cases", y="deaths",log_x=True, log_y=True)

fig2.update_layout(title="Scatter 1",xaxis_title="Cases (log)",yaxis_title="Deaths (log)")



fig.show()

fig2.show()
fig = px.scatter(Data_to_plot, x="cases", y="deaths", color="continentExp",log_x=True, log_y=True, labels={'continentExp' : 'Continent', 'cases': 'Cases', 'deaths': 'Deaths', 'popData2018' : 'Population'},marginal_y="rug", marginal_x="rug")

fig.update_layout(title="Scatter 3",xaxis_title="Cases (log)",yaxis_title="Deaths (log)")

fig.show()



fig2 = px.scatter(Data_to_plot, x="cases", y="deaths", color="continentExp",log_x=True, log_y=True,size='popData2018', labels={'continentExp' : 'Continent', 'cases': 'Cases', 'deaths': 'Deaths', 'popData2018' : 'Population'},marginal_y="violin", marginal_x="box", trendline="ols")

fig2.update_layout(title="Scatter 4",xaxis_title="Cases (log)",yaxis_title="Deaths (log)")

fig2.show()
Data_to_plot = df[(df['cases'] > 0) & (df['deaths']> 0) & (df['year'] >= 2020)]

fig = px.scatter(Data_to_plot, x="cases", y="deaths", facet_row="year", facet_col="month", color="continentExp", trendline="ols",labels={'continentExp' : 'Continent', 'cases': 'Cases', 'deaths': 'Deaths', 'popData2018' : 'Population'})

fig.update_layout(title="Scatter 5",xaxis_title="Cases",yaxis_title="Deaths")

fig.show()



Data_to_plot = df[(df['cases'] > 0) & (df['deaths']> 0) & (df['year'] >= 2020)]

Data_to_plot['Date_str']  = Data_to_plot['Date'].astype(str)

fig2 = px.scatter(Data_to_plot, x="cases", y="deaths", animation_frame="Date_str", size="cases", color="continentExp", hover_name="continentExp", facet_col="continentExp", facet_row="year",log_x=True, log_y=True

                ,labels={'continentExp' : 'Continent', 'cases': 'Cases', 'deaths': 'Deaths', 'popData2018' : 'Population', 'Date_str' : 'Date'})

fig2.show()
Data_to_plot = df.groupby('Date')[['deaths', 'cases']].sum().cumsum().reset_index()

Data_to_plot = pd.melt(Data_to_plot, id_vars=['Date'], value_vars=['cases', 'deaths'])

fig = px.line(Data_to_plot, x="Date", y="value",color= 'variable', labels={'continentExp' : 'Continent', 'cases': 'Cases', 'deaths': 'Deaths', 'popData2018' : 'Population', 'value' : 'Value', 'variable' : 'Variable'},line_shape="spline", render_mode="svg")

fig.update_layout(title="Line 1",xaxis_title="Date",yaxis_title="Cases")

fig.show()
Data_to_plot = df.groupby(['Date', 'continentExp'])[['deaths', 'cases']].sum().reset_index()

fig = px.line(Data_to_plot, x="Date", y="cases",labels={'continentExp' : 'Continent', 'cases': 'Cases', 'deaths': 'Deaths', 'popData2018' : 'Population'},line_shape="spline", render_mode="svg", hover_name="continentExp", color = "continentExp")

fig.update_layout(title="Line 2 (Cases)",xaxis_title="Date",yaxis_title="Cases")

fig.show()



fig2 = px.line(Data_to_plot, x="Date", y="deaths",labels={'continentExp' : 'Continent', 'cases': 'Cases', 'deaths': 'Deaths', 'popData2018' : 'Population'},      line_shape="spline", render_mode="svg", hover_name="continentExp", color = "continentExp")

fig2.update_layout(title="Line 3 (Deaths)",xaxis_title="Date",yaxis_title="Deaths")

fig2.show()
Data_to_plot = df.groupby(['Date', 'continentExp'])[['deaths', 'cases']].sum().reset_index()

Data_to_plot[['cumsum_deaths', 'cumsum_cases']] = Data_to_plot.groupby('continentExp').cumsum()

fig = px.area(Data_to_plot, x="Date", y="cumsum_deaths",labels={'continentExp' : 'Continent', 'cases': 'Cases', 'deaths': 'Deaths', 'popData2018' : 'Population', 'cumsum_cases' : 'Cumulative Cases', 'cumsum_deaths' : 'Cumulative Deaths'},line_shape="spline", hover_name="continentExp", color = "continentExp")

fig.update_layout(title="Area",xaxis_title="Date",yaxis_title="Cumulative Deaths")

fig.show()
Data_to_plot = df.groupby(['Date'])[['deaths', 'cases']].sum().reset_index()

fig = px.bar(Data_to_plot, x="Date", y="cases",labels={'continentExp' : 'Continent', 'cases': 'Cases', 'deaths': 'Deaths', 'popData2018' : 'Population', 'count':'Count'},)

fig.update_layout(title="Bar 1",xaxis_title="Date",yaxis_title="Cases")

fig.show()
Data_to_plot = df.groupby(['Date'])[['deaths', 'cases']].sum().reset_index()

Data_to_plot = pd.melt(Data_to_plot, id_vars=['Date'], value_vars=['cases', 'deaths'])

Data_to_plot

fig = px.bar(Data_to_plot, x="Date", y="value", color="variable",labels={'continentExp' : 'Continent', 'cases': 'Cases', 'deaths': 'Deaths', 'popData2018' : 'Population', 'variable':'Variables'},)

fig.update_layout(title="Bar 2",xaxis_title="Date",yaxis_title="Values")

fig.show()

fig2 = px.bar(Data_to_plot, x="Date", y="value", color="variable", barmode="group",labels={'continentExp' : 'Continent', 'cases': 'Cases', 'deaths': 'Deaths', 'popData2018' : 'Population', 'variable':'Variables'},)

fig2.update_layout(title="Bar 3",xaxis_title="Date",yaxis_title="Values")

fig2.show()
Data_to_plot = df[df['Date'] == df['Date'].max()].groupby(['countriesAndTerritories'])[['deaths', 'cases']].sum().reset_index().sort_values(by = 'cases', ascending = True).tail(20)

Data_to_plot = pd.melt(Data_to_plot, id_vars=['countriesAndTerritories'], value_vars=['cases', 'deaths'])

fig = px.bar(Data_to_plot,x='value', y="countriesAndTerritories", color='variable',  text='value', orientation='h', width=700, labels={'continentExp' : 'Continent', 'cases': 'Cases', 'deaths': 'Deaths', 'countriesAndTerritories' : 'Countries', 'variable':'Variables'},color_discrete_sequence = px.colors.qualitative.Dark2)

fig.update_layout(title="Bar 4",xaxis_title="Values",yaxis_title="Countries")

fig.show()
Data_to_plot = df.copy()

Data_to_plot['deaths_log']  = np.log10(Data_to_plot['deaths'])

Data_to_plot['cases_log']  = np.log10(Data_to_plot['cases'])

Data_to_plot.head()

fig = px.density_contour(Data_to_plot, x="deaths_log", y="cases_log",labels={'continentExp' : 'Continent', 'cases': 'Cases', 'deaths': 'Deaths', 'popData2018' : 'Population', 'cases_log' : 'Cases Log10', 'deaths_log' : 'Deaths Log10', 'count':'Count'},)

fig.update_layout(title="Density Contour 1",xaxis_title="Cases (log10)",yaxis_title="Deaths (log10)")

fig.show()

fig2 = px.density_contour(Data_to_plot, x="deaths_log", y="cases_log",color="continentExp", marginal_x="rug", marginal_y="histogram",labels={'continentExp' : 'Continent', 'cases': 'Cases', 'deaths': 'Deaths', 'popData2018' : 'Population', 'cases_log' : 'Cases Log10', 'deaths_log' : 'Deaths Log10', 'count':'Count'},)

fig2.update_layout(title="Density Contour 2",xaxis_title="Cases (log10)",yaxis_title="Deaths (log10)")

fig2.show()
fig = px.density_heatmap(Data_to_plot, x="deaths_log", y="cases_log",marginal_x="rug", marginal_y="histogram",labels={'continentExp' : 'Continent', 'cases': 'Cases', 'deaths': 'Deaths', 'popData2018' : 'Population', 'cases_log' : 'Cases Log10', 'deaths_log' : 'Deaths Log10', 'count':'Count'},)

fig.update_layout(title="Density Heatmap",xaxis_title="Cases (log10)",yaxis_title="Deaths (log10)")

fig.show()
Data_to_plot = df[df['Date'] == df['Date'].max()].groupby(['continentExp'])[['deaths', 'cases']].sum().reset_index()

fig = px.pie(Data_to_plot, values='cases', names='continentExp',labels={'continentExp' : 'Continent', 'cases': 'Cases', 'deaths': 'Deaths', 'countriesAndTerritories' : 'Countries', 'variable':'Variables'},)

fig.update_layout(title="Pie 1")

fig.show()

Data_to_plot = df[df['Date'] == df['Date'].max()].groupby(['countriesAndTerritories'])[['deaths', 'cases']].sum().reset_index()

fig2 = px.pie(Data_to_plot, values='cases', names='countriesAndTerritories', labels={'continentExp' : 'Continent', 'cases': 'Cases', 'deaths': 'Deaths', 'countriesAndTerritories' : 'Countries', 'variable':'Variables'},)

fig2.update_layout(title="Pie 2",)

fig2.show()
Data_to_plot = df[df.month.isin(['1','2','3','4','5'])].groupby(['month', 'countriesAndTerritories'])[['deaths', 'cases']].sum().reset_index()

Data_to_plot['deaths_log']  = np.log10(Data_to_plot['deaths']+1)

Data_to_plot['cases_log']  = np.log10(Data_to_plot['cases']+1)

fig = px.strip(Data_to_plot, x="month", y="cases_log", labels={'continentExp' : 'Continent', 'cases_log': 'Cases Log10', 'deaths_log': 'Deaths Log10', 'month' : 'Month', 'variable':'Variables'},)

fig.update_layout(title="Strip")

fig.show()
Data_to_plot = df[df.month.isin(['1','2','3','4','5'])].groupby(['month', 'countriesAndTerritories'])[['deaths', 'cases']].sum().reset_index()

Data_to_plot = pd.melt(Data_to_plot, id_vars=['month', 'countriesAndTerritories'], value_vars=['cases', 'deaths'])

Data_to_plot['value_log10']  = np.log10(Data_to_plot['value']+1)

fig = px.box(Data_to_plot, x="month", y="value_log10", color="variable", notched=True, labels={'variable' : 'Variables', 'cases_log': 'Cases Log10', 'deaths_log': 'Deaths Log10', 'month' : 'Month', 'value_log10':'Values Log10'},)

fig.update_layout(   title="Box")

fig.show()
fig = px.violin(Data_to_plot, x="month", y="value_log10", color="variable", box=True, points="all", labels={'variable' : 'Variables', 'cases_log': 'Cases Log10', 'deaths_log': 'Deaths Log10', 'month' : 'Month', 'value_log10':'Values Log10'},)

fig.update_layout(title="Violin")

fig.show()
df_to_plot =  df[df.month.isin(['2'])][['cases','deaths','day','month', 'continentExp', 'popData2018']]

dict_ = {'Asia' : 1, 'Europe' : 2, 'Africa' : 3, 'America' : 4}

df_to_plot['continentEmap'] = df_to_plot['continentExp'].map(dict_)

df_to_plot['cases_log10'] =  np.log10(df_to_plot['cases']+1)

df_to_plot['deaths_log10'] =  np.log10(df_to_plot['deaths']+1)

fig = px.parallel_coordinates(df_to_plot[['cases_log10','deaths_log10','continentEmap']], color = 'continentEmap',labels={'cases_log10' : 'Cases Log10', 'deaths_log10': 'Deaths Log10', 'continentEmap': 'Continent Id', 'month' : 'Month', 'value_log10':'Values Log10'},color_continuous_scale=px.colors.diverging.Tealrose, color_continuous_midpoint=2)

fig.update_layout(title="Parallel Coordinates")

fig.show()
df_to_plot['cut_cases_log10'] = pd.cut(df_to_plot['cases_log10'], 4).astype(str)

df_to_plot['cut_deaths_log10'] = pd.cut(df_to_plot['deaths_log10'], 4).astype(str)

fig = px.parallel_categories(df_to_plot[['cut_cases_log10','cut_deaths_log10','continentEmap']], color = 'continentEmap',labels={'cut_cases_log10' : 'Interval Cases Log10', 'cut_deaths_log10': 'Interval Deaths Log10', 'continentEmap': 'Continent Id', 'month' : 'Month', 'value_log10':'Values Log10'},color_continuous_scale=px.colors.diverging.Tealrose, color_continuous_midpoint=2)

fig.update_layout(title="Parallel Categories")

fig.show()
df2 =pd.DataFrame(np.random.rand(40,4))

df2['label'] = np.round((3*np.random.rand(40,1) - 3)*-1)

df2[df2['label'] == 0] = 1

df2['label'] = df2['label'].astype(int).map({1:'Label_1', 2:'Label_2', 3:'Label_3'})

df2.columns = ['Value_1','Value_2','Value_3','Value_4', 'Labels']

fig = px.scatter_ternary(df2, a="Value_1", b="Value_2", c="Value_3", color="Labels", size="Value_4", hover_name="Labels",size_max=15, color_discrete_map = {"Label_1": "blue", "Label_2": "green", "Label_3":"red"},labels={'Value_1' : 'Value 1', 'Value_2': 'Value 2', 'Value_3': 'Value 3', 'Value_4' : 'Value 4'})

fig.update_layout(title="Scatter Ternary")

fig.show()
fig = px.line_ternary(df2, a="Value_1", b="Value_2", c="Value_3", color="Labels", hover_name="Labels",line_dash="Labels",color_discrete_map = {"Label_1": "blue", "Label_2": "green", "Label_3":"red"},labels={'Value_1' : 'Value 1', 'Value_2': 'Value 2', 'Value_3': 'Value 3', 'Value_4' : 'Value 4'})

fig.update_layout(title="Line Ternary")

fig.show()
fig = px.scatter_3d(df2, x="Value_1", y="Value_2", z="Value_3", color="Labels", size="Value_4", hover_name="Labels",symbol="Labels", color_discrete_map = {"Label_1": "blue", "Label_2": "green", "Label_3":"red"},labels={'Value_1' : 'Value 1', 'Value_2': 'Value 2', 'Value_3': 'Value 3', 'Value_4' : 'Value 4'})

fig.update_layout(title="Scatter 3D")

fig.show()
df2 = pd.DataFrame(np.round((2.5*np.random.rand(128,1) - 2.5)*-1, 1))

df2['label'] = np.round((5*np.random.rand(128,1) - 5)*-1)

df2['direction'] = np.round((12*np.random.rand(128,1) - 12)*-1)

df2[df2['label'] == 0] = 1

df2[df2['direction'] == 0] = 1

df2['label'] = df2['label'].astype(int).map({1:'Label_1', 2:'Label_2', 3:'Label_3', 4:'Label_4' , 5:'Label_5'})

df2['direction'] = df2['direction'].astype(int).map({1 : 'D1', 2 : 'D2', 3 : 'D3', 4: 'D4', 5 : 'D5', 6 : 'D6', 7 : 'D7', 8 : 'D8', 9 : 'D9', 10 : 'D10', 11 : 'D11', 12 : 'D12'})

df2.columns = ['Values','Label','Direction']

fig = px.scatter_polar(df2, r="Values", theta="Direction", color="Label", symbol="Label",color_discrete_sequence=px.colors.sequential.Rainbow)

fig.update_layout(title="Scatter Polar")

fig.show()
fig = px.line_polar(df2,r="Values", theta="Direction", color="Label", line_close="Label",line_dash="Label",color_discrete_sequence=px.colors.sequential.Rainbow)

fig.update_layout(title="Line Polar")

fig.show()
fig = px.bar_polar(df2,r="Values", theta="Direction", color="Label", template="plotly_dark",color_discrete_sequence= px.colors.sequential.Plasma_r)

fig.update_layout(title="Bar Polar")

fig.show()
df3 = pd.read_csv('../input/covid19-in-usa/us_counties_covid19_daily.csv')

df3['Date'] = pd.to_datetime(df3['date'])

df3 = df3[df3['Date'] == df3['Date'].max()]

df3 = df3.groupby('state').sum().reset_index()



import pandas as pd

import requests

from bs4 import BeautifulSoup



url = 'https://www.latlong.net/category/states-236-14.html'



r = requests.get(url)

html = r.text



soup = BeautifulSoup(html)

table = soup.find('table')

rows = table.find_all('tr')

data = []

for row in rows[1:]:

    cols = row.find_all('td')

    cols = [ele.text.strip() for ele in cols]

    data.append([ele for ele in cols if ele])



result = pd.DataFrame(data, columns=['Place_Name', 'Latitude', 'Longitude'])



result['Place_Name'] = result['Place_Name'].str.replace(', the USA', '')

result['Place_Name'] = result['Place_Name'].str.replace(', the US', '')

result['Place_Name'] = result['Place_Name'].str.replace(', USA', '')

result['Place_Name'] = result['Place_Name'].str.replace(' State', '')

result['Latitude'] = result['Latitude'].astype(float)

result['Longitude'] = result['Longitude'].astype(float)



df3 = df3.merge(result, left_on='state', right_on='Place_Name')

px.set_mapbox_access_token('pk.eyJ1IjoibWVoZGlnYXNtaSIsImEiOiJjazkwcXplbGowNDNwM25saDBldzY0NmQwIn0.gYQr41tH3KKMOHnml_REeQ')

fig = px.scatter_mapbox(df3, lat="Latitude", lon="Longitude",  color="cases", size = "cases",  zoom=3, labels={'cases' : 'Cases'},  hover_name="state",color_continuous_scale=px.colors.sequential.Plotly3)

fig.update_layout(title="Map 1")

fig.show()



px.data.gapminder()

df_to_plot = df.groupby(['Date', 'countryterritoryCode', 'countriesAndTerritories', 'month'])['cases','deaths'].sum().reset_index()

df_to_plot.loc[df_to_plot.cases < 0,'cases'] = 0

fig2 = px.scatter_geo(df_to_plot, locations="countryterritoryCode", color="cases", hover_name="countriesAndTerritories", size="cases",labels={'cases' : 'Cases', 'countryterritoryCode' : 'Country', 'month' : 'Month'},projection="natural earth", animation_frame="month", color_continuous_scale=px.colors.sequential.Plotly3)

fig2.update_layout(title="Map 2")

fig2.show()



df_to_plot2 = df_to_plot[df_to_plot.Date == df_to_plot.Date.max()]

fig3 = px.scatter_geo(df_to_plot2, locations="countryterritoryCode", color="deaths",size="deaths", projection="orthographic", labels={'deaths' : 'Deaths', 'countryterritoryCode' : 'Country', 'month' : 'Month'},color_continuous_scale=px.colors.sequential.Plotly3, hover_name="countriesAndTerritories")

fig3.update_layout(title="Map 3")

fig3.show()
fig = px.choropleth(df_to_plot, locations="countryterritoryCode", color="cases", hover_name="countriesAndTerritories", animation_frame="month", range_color=[0,30000], labels={'cases' : 'Cases', 'countryterritoryCode' : 'Country', 'month' : 'Month'}, color_continuous_scale=px.colors.sequential.OrRd)

fig.update_layout(title="Map 4")

fig.show()