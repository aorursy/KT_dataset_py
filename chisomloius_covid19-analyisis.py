import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os
import datetime
import plotly.express as px
#huge_data_path='./corona/2019_nCoV_data.csv'
main_data_path ='/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv'
#recovered_data_path='./corona/time_series_2019_ncov_recovered.csv'
#deaths_data_path='./corona/time_series_2019_ncov_deaths.csv'
#confirmed_data_path='./corona/time_series_2019_ncov_confirmed.csv'
work_data = pd.read_csv(main_data_path)

print("dataset is ready")
work_data.info()
work_data.shape
work_data.head(10)
work_data.columns
work_data.drop(['Last Update', 'SNo'], axis = 1) 
covid_data.info()
#pre-processing data for the next step
work_data["ObservationDate"] = work_data['ObservationDate'].astype('datetime64')
work_data["Confirmed"] = work_data['Confirmed'].astype('int64')
work_data["Deaths"] = work_data['Deaths'].astype('int64')
work_data["Recovered"] = work_data['Recovered'].astype('int64')
work_data_date=pd.DataFrame(work_data.groupby(by='ObservationDate').sum())
work_data_date['Date']=work_data_date.index
work_data_date.Date=work_data_date.Date.apply(lambda x:x.date())
work_data_melted=pd.melt(work_data_date,id_vars=['Date'])
def plot_builder(col_name,data_name,**kwargs):
    title = ''
    try:
        title = kwargs['title']
    except:
        title = 'Corona Virus 2019'
    plt.figure(figsize=(10,15))
    plt.xticks(rotation=90)
    plt.xlabel('Date', fontsize=18)
    plt.suptitle(title)
    plot_1=sns.barplot(x='Date',y=col_name,data=data_name)
    plot_1
plot_builder('Confirmed',work_data_date)
'''each
for row in covid_data_date['Confirmed']:
    if(!row_prev==None):
        '''
work_data_date_diff=pd.DataFrame(work_data_date.diff())
work_data_date_diff['Date']=work_data_date_diff.index
work_data_date_diff.Date=work_data_date_diff.Date.apply(lambda x:x.date())
work_data_diff_melted = pd.melt(work_data_date_diff,id_vars=['Date'])
plot_builder('Confirmed',work_data_date_diff, title='Confirmed COVID19 Daily')
plot_builder('Deaths',work_data_date,title='Accumalted amount of Deaths')
plot_builder('Deaths',work_data_date_diff,title='Quantity of Deaths Daily')
covid_deaths_sorted = work_data_date_diff.sort_values('Deaths')
plot_builder('Deaths',covid_deaths_sorted,title='Ascending amount of Deaths')
cmp_plot=sns.catplot(x='Date', y='value', hue='variable', data=work_data_diff_melted, kind='bar',height=10,aspect =1.6,legend=True)
cmp_plot.set_xticklabels( rotation=90)

cmp_plot=sns.catplot(x='Date', y='value', hue='variable', data=work_data_diff_melted, kind='bar',height=10,aspect =1.6,legend=True)
cmp_plot.set_xticklabels( rotation=90)
fig = px.line(covid_date_melted, x="Date",y='value', color='variable')
fig.show()
work_data_country=pd.DataFrame(work_data.groupby(by='Country/Region').sum())
work_data_country['country']=work_data_country.index
work_data_country.index
work_data_country.sort_values(['Confirmed','Deaths','Recovered'],ascending=[False, False, False])
work_data_country["country"].replace({"Ivory Coast": "Cote d'Ivoire", 
                                        "Mainland China": "China",
                                        "Hong Kong":"Hong Kong, China",
                                       "South Korea":"Korea, Rep.",
                                        "UK":"United Kingdom",
                                        "US":"United States",
                                        "Macau" :"China"
                                       }, inplace=True)

df = px.data.gapminder().query("year == 2007")
work_data_country_geo_cd=pd.merge(work_data_country,df, how='left',on='country')
work_data_country_geo_cd=work_data_country_geo_cd.dropna(how='any')
work_data_country_geo_cd=work_data_country_geo_cd.drop(['year','lifeExp','pop','gdpPercap'],axis=1)
fig = px.scatter_geo(work_data_country_geo_cd[work_data_country_geo_cd['country']!= 'China'], locations="iso_alpha",
                     size="Confirmed", # size of markers, "pop" is one of the columns of gapminder)
                    )
fig.show()
fig = px.scatter_geo(work_data_country_geo_cd[work_data_country_geo_cd['country']!= 'China'], locations="iso_alpha",
                     size="Deaths", # size of markers, "pop" is one of the columns of gapminder)
                    )
fig.show()
fig = px.scatter_geo(work_data_country_geo_cd[work_data_country_geo_cd['country']!= 'China'], locations="iso_alpha",
                     size="Recovered", #size of the column
                    )
fig.show()
fig = px.choropleth(work_data_country_geo_cd[work_data_country_geo_cd['country']!= None], locations="iso_alpha",
                    color="Confirmed", # Column name we want
                    hover_name="country", # If u hover your mouse over the zone, you will see the country name
                    color_continuous_scale=px.colors.sequential.Burg)
fig.show()
fig = px.choropleth(work_data_country_geo_cd[work_data_country_geo_cd['country']!= 'China'], locations="iso_alpha",
                    color="Deaths", # Col name
                    hover_name="country", # If u hover your mouse over the zone, you will see the country name
                    color_continuous_scale=px.colors.sequential.OrRd)
fig.show()
def cross_entropy(y,a):
    output = pd.Series(np.subtract(np.exp(a,2), np.exp(y,2)))
    return output
work_data_country['Death_ratio']= np.log(work_data_country['Deaths'])/np.log(work_data_country['Confirmed'])
work_data_country['Heal_ratio']=np.log(work_data_country['Recovered'])/np.log(work_data_country['Confirmed'])
work_data_country_efficency_death= work_data_country.sort_values(['Death_ratio','Heal_ratio'],ascending=[False,True], na_position='last')
work_data_country_efficency_death.head(50)
def plot_builder_new(col_name,data_name,**kwargs):
    title = ''
    try:
        title = kwargs['title']
    except:
        title = 'Corona Virus 2019'
    plt.figure(figsize=(10,15))
    plt.xticks(rotation=90)
    plt.xlabel('Date', fontsize=18)
    plt.suptitle(title)
    plot_1=sns.barplot(x='country',y=col_name,data=data_name)
    plot_1
plot_builder_new('Death_ratio',work_data_country_efficency_death,title='efficency')
work_data_country_geo_cd = pd.merge(work_data_country_geo_cd,work_data_country_efficency_death,how='left',on='country')
fig = px.choropleth(work_data_country_geo_cd[work_data_country_geo_cd['country']!= 'China'], locations="iso_alpha",
                    color="Death_ratio", # Col name
                    hover_name="country", # If u hover your mouse over the zone, you will see the country name
                    color_continuous_scale=px.colors.sequential.OrRd)
fig.show()
work_data_country_geo_cd = work_data_country_geo_cd.sort_values(['Heal_ratio','Death_ratio'],ascending=[True,False])
fig = px.choropleth(work_data_country_geo_cd[work_data_country_geo_cd['country']!= 'China'], locations="iso_alpha",
                    color="Heal_ratio", # Col name
                    hover_name="country", # If u hover your mouse over the zone, you will see the country name
                    color_continuous_scale=px.colors.sequential.algae)
fig.show()