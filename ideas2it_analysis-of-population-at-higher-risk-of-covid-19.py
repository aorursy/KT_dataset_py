# # This Python 3 environment comes with many helpful analytics libraries installed
# # It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# # For example, here's several helpful packages to load in 

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# # Input data files are available in the "../input/" directory.
# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# # Any results you write to the current directory are saved as output.
# Importing necessary library
import pandas as pd
import numpy as np
import glob  
import matplotlib.pyplot as plt
import plotly as py
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as pyo
pyo.init_notebook_mode()
import seaborn as sns
covid = pd.read_csv('../input/johns-hopkins/johns-hopkins-covid-19-daily-dashboard-cases-over-time.csv').rename(columns={'country_region':'Country'})
#info = pd.read_csv('countryinfo/covid19countryinfo.csv').rename(columns={'country':'Country'})
info = pd.read_csv('../input/countryinfo/covid19countryinfo.csv').rename(columns={'country':'Country'})
info = info[info.region.isnull()]
info['pop'] = info[~info['pop'].isnull()]['pop'].str.replace(',','').astype('int64')
info['totalcases'] = info[~info['totalcases'].isnull()]['totalcases'].str.replace(',','').astype('int64')
info['casediv1m'] = info[~info['casediv1m'].isnull()]['casediv1m'].str.replace(',','').astype('float')
# info['healthexp'] = info[~info['healthexp'].isnull()]['healthexp'].str.replace(',','').astype('float')
# info['gdp2019'] = info[~info['gdp2019'].isnull()]['gdp2019'].str.replace(',','').astype('float')
covid = covid.merge(info[['Country', 'pop']], how='left', on='Country')

covid.Country.nunique()
covid['confirmed_per_1000']= covid.confirmed*1000/covid['pop']
covid[covid.Country.isin(['Iran','France','Italy', 'Spain'])][['last_update', 'confirmed','Country']].pivot(index='last_update', columns='Country', values='confirmed').plot(figsize=(20,5))
plt.axhline(y=300, color='r', linestyle='dashed')
axes = plt.gca()
axes.set_ylim([0,5000])
plt.ylabel('Confirmed cases')
plt.xlabel('Date')
pandemic_country = []
rate_at_exp = []

for cntry in covid.Country.unique():
    country = covid[covid["Country"]==cntry]
    country = country.sort_values("confirmed",ascending=True)
    #By plotting the confirmed cases over time,
    #the confirmed cases takes exponential shape after critical mass of 300 confirmed cases
    country = country[country.confirmed>300]
    country.reset_index(drop=True, inplace=True)
    spread_rate=country.confirmed.pct_change(7).values
    spread_double_counter=0
    tmplst=[]
    #Check if the exponential happened after a week
    for i in range(7,len(spread_rate),7):
        if spread_rate[i] > 0.8:
            spread_double_counter+=1
            tmplst.append(country.confirmed_per_1000[i])
            
    #Term a country pandemic if doubling effect continued for more than a week        
    if spread_double_counter >1:
        pandemic_country.append(cntry)
        rate_at_exp.extend(tmplst)
print("Pandemic Countries:")
pandemic_country
pandemic_country = pd.DataFrame(pandemic_country, columns={'Country'})
pandemic_country['risk_level']='Pandemic'
median_rate= np.quantile(rate_at_exp,0.5)
risk_country = covid[(covid.confirmed_per_1000>=median_rate) & (covid.confirmed>300) ].Country.unique()
risk_country = [country for country in risk_country if country not in pandemic_country.Country.values]
print('High Risk Countries :\n', risk_country)

risk_country = pd.DataFrame(risk_country, columns={'Country'})
risk_country['risk_level'] = 'High Risk'
risk = risk_country.append(pandemic_country, ignore_index=True)
risk = risk.merge(info, how='left', on='Country')
#Sort from low to high intensity
risk = risk.sort_values(by=['risk_level', 'totalcases'], ascending=True).reset_index(drop=True).reset_index()
risk.Country.nunique()
fig = go.Figure(data=go.Choropleth(
    locations = risk['alpha3code'],
    z = risk['index'],
    text = risk['Country'],
    customdata=risk[['risk_level', 'totalcases']],
    colorscale = [[0, 'rgb(255,255,0)'], [1, 'rgb(255,0,0)']],
    autocolorscale=False,
    reversescale=False,
    marker_line_color='darkgray',
    marker_line_width=0.5,
    hovertemplate="%{text}<br>Cases : %{customdata[1]}<br>Level : %{customdata[0]}<extra></extra>",
    colorbar_title='Intensity'
    #color=risk['risk_level']
))

fig.update_layout( title='Pandemic and High risk countries')

fig.show()
## Keeping a copy of the same to be used later
info_copy = info.copy()
# Taking Relevant fields and changhing the data types of few columns

info = info[['Country','medianage','lung','hospibed','avgtemp','healthexp','avghumidity','casediv1m','gdp2019','smokers','totalcases']]
info["casediv1m"] = info["casediv1m"].astype("float")
info["healthexp"] = info["healthexp"].str.replace(",","").astype("float")
info["gdp2019"] = info["gdp2019"].str.replace(",","").astype("float")
#Correlation Study based on the pandemic countries
info_pandemic = info[info["Country"].isin(risk.Country)]
print("Correlation of different variables with the cases")
info_pandemic.corr()[["casediv1m","totalcases"]]
import seaborn as sns
pairplot =sns.pairplot(info_pandemic)
pairplot.fig.set_size_inches(15,15)
# Box plot showing the median age distribution
sns.boxplot(risk.medianage)
sns.kdeplot(risk.medianage)
#Calculating the potential range for median age
age_lower_limit = round(risk.medianage.round().quantile(0.25))
age_upper_limit = round(risk.medianage.round().quantile(1))

print("The potential range for countries, of median age which have higher risk of contract "
     ,age_lower_limit,"--",age_upper_limit)
print("List of countries within the potential range:")
age_countries = info_copy[(info_copy["medianage"]>=age_lower_limit)][["Country","totalcases","medianage","alpha3code"]]
age_countries = age_countries[~age_countries["Country"].isin(risk.Country)]
age_countries = age_countries.sort_values(by=['medianage', 'totalcases'], ascending=True).reset_index(drop=True).reset_index()
age_countries["Country"].unique()
fig = go.Figure(data=go.Choropleth(
    locations = age_countries['alpha3code'],
    z = age_countries['index'],
    text = age_countries['Country'],
    customdata=age_countries[['medianage', 'totalcases']],
    colorscale = [[0, 'rgb(255,255,0)'], [1, 'rgb(255,0,0)']],
    autocolorscale=False,
    reversescale=False,
    marker_line_color='darkgray',
    marker_line_width=0.5,
    hovertemplate="%{text}<br>Cases : %{customdata[1]}<br>Median Age : %{customdata[0]}<extra></extra>",
    colorbar_title='Intensity'
))

fig.update_layout( title='Countries which might be of risk from higher median age.')

fig.show()
