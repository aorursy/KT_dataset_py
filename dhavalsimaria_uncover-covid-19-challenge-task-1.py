import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib.font_manager
from sklearn.impute import SimpleImputer
import plotly.express as px
import plotly.offline as offline
from plotly.offline import init_notebook_mode,iplot
import plotly.graph_objects as go
import cufflinks as cf
init_notebook_mode(connected=True)

import pandas as pd
import numpy as np
gov_measures_data = pd.read_csv('../input/uncover/UNCOVER_v4/UNCOVER/HDE/HDE/acaps-covid-19-government-measures-dataset.csv')
gov_measures_data.shape
gov_measures_data.describe()
gov_measures_data.info()
covid19_stats_data = pd.read_csv('../input/uncover/UNCOVER_v4/UNCOVER/our_world_in_data/coronavirus-disease-covid-19-statistics-and-research.csv')
covid19_stats_data.shape
covid19_stats_data.info()
covid19_stats_with_gov_measures_data = pd.merge(covid19_stats_data, gov_measures_data, how='outer', left_on=['iso_code', 'date'], right_on=['iso', 'date_implemented'])
covid19_stats_with_gov_measures_data.shape
covid19_stats_with_gov_measures_data = covid19_stats_with_gov_measures_data.drop(['id','pcode','comments','source', 'source_type', 'link', 'alternative_source'], axis=1)
covid19_stats_with_gov_measures_data.country.fillna(covid19_stats_with_gov_measures_data.location, inplace=True)
covid19_stats_with_gov_measures_data.iso_code.fillna(covid19_stats_with_gov_measures_data.iso, inplace=True)
covid19_stats_with_gov_measures_data.date_implemented.fillna(covid19_stats_with_gov_measures_data.date, inplace=True)
covid19_stats_with_gov_measures_data = covid19_stats_with_gov_measures_data.drop(['location', 'iso'], axis=1)
covid19_stats_with_gov_measures_data[covid19_stats_with_gov_measures_data['country'].isna()]
covid19_stats_with_gov_measures_data = covid19_stats_with_gov_measures_data.dropna(subset=['country', 'date'])
covid19_stats_with_gov_measures_data = covid19_stats_with_gov_measures_data[(covid19_stats_with_gov_measures_data['country'] != 'Moldova Republic of') & (covid19_stats_with_gov_measures_data['country'] != 'Czech republic') & (covid19_stats_with_gov_measures_data['country'] != 'Afghanistan')]
countries = covid19_stats_with_gov_measures_data.country.unique()
numeric_cols = list(covid19_stats_with_gov_measures_data.select_dtypes(include=['number']).columns)
non_numeric_cols = list(covid19_stats_with_gov_measures_data.select_dtypes(exclude=['number']).columns)
for col in numeric_cols:
    for country in countries:
        mask = (covid19_stats_with_gov_measures_data['country']==country)
        sum_notna = covid19_stats_with_gov_measures_data.loc[mask, col].notna().sum()
        col_mean = 0
        if sum_notna > 0:
            col_mean = covid19_stats_with_gov_measures_data[mask][col].mean(skipna = True)
            
        covid19_stats_with_gov_measures_data[col].fillna(col_mean, inplace=True)
for col in non_numeric_cols:
    for country in countries:
        mask = (covid19_stats_with_gov_measures_data['country']==country)
        mask2 = (covid19_stats_with_gov_measures_data['country']==country) & covid19_stats_with_gov_measures_data[col].notna()
        sum_notna = covid19_stats_with_gov_measures_data.loc[mask, col].notna().sum()
        if(sum_notna > 0):
            imputer = SimpleImputer(strategy='most_frequent')
            covid19_stats_with_gov_measures_data[mask2][col] = imputer.fit_transform(covid19_stats_with_gov_measures_data[mask2])
for country in countries:
    mask = (covid19_stats_with_gov_measures_data['country']==country)
    
covid19_stats_with_gov_measures_data = covid19_stats_with_gov_measures_data[covid19_stats_with_gov_measures_data['measure'].notna()]
covid19_stats_with_gov_measures_data['date_implemented'] = pd.to_datetime(covid19_stats_with_gov_measures_data['date_implemented'])
covid19_stats_with_gov_measures_data['entry_date'] = pd.to_datetime(covid19_stats_with_gov_measures_data['entry_date'])
covid19_stats_with_gov_measures_data['date'] = pd.to_datetime(covid19_stats_with_gov_measures_data['date'])
covid19_stats_with_gov_measures_data[non_numeric_cols]
covid19_stats_measures_df1 = covid19_stats_with_gov_measures_data[covid19_stats_with_gov_measures_data.columns[~covid19_stats_with_gov_measures_data.columns.isin(['tests_units', 'admin_level_name', 'non_compliance'])]]
non_numeric_cols_df1 = list(covid19_stats_measures_df1.select_dtypes(include=['object']).columns)
covid19_stats_measures_df1[non_numeric_cols_df1]
# Correlation Matrix Heatmap
f, ax = plt.subplots(figsize=(20, 12))
corr = covid19_stats_measures_df1.corr()
hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm",fmt='.2f',
                 linewidths=.05)
f.subplots_adjust(top=0.93)
t= f.suptitle('DS1+DS2 Covid-19 statistics Correlation Heatmap', fontsize=14)
countries = covid19_stats_measures_df1.country.unique()
fig2 = go.Figure()

col_to_plot=["total_cases", "total_deaths", "new_cases", "new_deaths"]

updatemenu= []
buttons = []

data = []

for n, country in enumerate(countries):
    visible = [False] * len(countries)
    visible[n] = True
    temp_dict = dict(label = str(country),
                    method = 'update',
                    args = [{'visible': visible},
                           {'title': 'Country %s' % country}])
    updatemenu.append(temp_dict)
    

for n, col in enumerate(col_to_plot):
    for country in countries:
        mask = (covid19_stats_measures_df1.country.values == country)
        trace = (dict(
            visible = False,
            name = col,
            x = covid19_stats_measures_df1.loc[mask, 'date'],
            y = covid19_stats_measures_df1.loc[mask, col],
            mode='lines+markers'
        ))
        
        data.append(trace)
    

layout = dict(updatemenus = list([dict(buttons = updatemenu)]), title = "COVID-19 stats")

fig2 = dict(data = data, layout = layout)
offline.iplot(fig2, filename = 'update_dropdown')
covid19_stats_measures_df1_india = covid19_stats_measures_df1[covid19_stats_measures_df1['country']=='India']
covid19_stats_measures_df1_india['percent_change_total_cases'] = covid19_stats_measures_df1_india['total_cases'].pct_change();
covid19_stats_measures_df1_india['percent_change_total_deaths'] = covid19_stats_measures_df1_india['total_deaths'].pct_change();
covid19_stats_measures_df1_india['percent_change_new_cases'] = covid19_stats_measures_df1_india['new_cases'].pct_change();
covid19_stats_measures_df1_india['percent_change_new_deaths'] = covid19_stats_measures_df1_india['new_deaths'].pct_change();
covid19_stats_measures_df1_india.replace([np.inf, -np.inf], np.nan,inplace=True)
covid19_stats_measures_df1_india = covid19_stats_measures_df1_india.fillna(0)
covid19_stats_measures_df1_india[['date', 'total_cases', 'percent_change_total_cases', 'total_deaths', 'percent_change_total_deaths', 'new_cases', 'percent_change_new_cases', 'new_deaths', 'percent_change_new_deaths']]
full_lockdown_date = covid19_stats_measures_df1_india[covid19_stats_measures_df1_india['measure'] == "Full lockdown"].date_implemented
full_lockdown_date= np.datetime_as_string(full_lockdown_date, unit='D')


fig_new_case_count = px.line(covid19_stats_measures_df1_india, x="date", y=["new_cases"], log_y=False, title="Effect of Full Lockdown on Count of New Covid-19 cases in India")
fig_new_case_count.update_layout(shapes=[
    dict(
      type= 'line',
      yref= 'paper', y0= 0, y1= 1,
      xref= 'x', x0= full_lockdown_date[0], x1= full_lockdown_date[0]
    )
])
fig_new_case_count.show()

fig_new_case_percent = px.line(covid19_stats_measures_df1_india, x="date", y=["percent_change_new_cases"], log_y=False, title="Effect of Full Lockdown on Percent change in New Covid-19 cases in India")
fig_new_case_percent.update_layout(shapes=[
    dict(
      type= 'line',
      yref= 'paper', y0= 0, y1= 1,
      xref= 'x', x0= full_lockdown_date[0], x1= full_lockdown_date[0]
    )
])
fig_new_case_percent.show()