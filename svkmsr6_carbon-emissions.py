import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style("darkgrid")
#Importing the Raw data
raw_data = pd.read_csv('../input/co2-ghg-emissionsdata/co2_emission.csv')
raw_data.head()
raw_data.info()
null_ent = raw_data[pd.isnull(raw_data['Code'])]['Entity'].unique()
null_ent
#Getting countries and regions that have a valid country code
data_pp = raw_data[pd.notnull(raw_data['Code'])]
data_pp.head()
data_pp.info()
countries = data_pp['Entity'].unique()
country_codes = data_pp['Code'].unique()

countries.sort(axis=0)
countries
country_codes.sort(axis=0)
country_codes
import matplotlib.pyplot as plt
#Generic Function to load a chart for yearly emissions of a particular Country/Region
def gen_chart_for_country(df,country_code='IND'):
  data = df.copy()
  emissions = data[data['Code'] == country_code]
  country = emissions['Entity'].iloc[0]
  emissions = emissions.sort_values('Year')
  emissions['Annual CO₂ emissions (tonnes )'] = emissions['Annual CO₂ emissions (tonnes )'].astype(int)
  sns.lineplot(x='Year',y='Annual CO₂ emissions (tonnes )', data=emissions)
  plt.title(country)
#Tuvalu - Smallest
gen_chart_for_country(data_pp,"TUV")
#Kiribati
gen_chart_for_country(data_pp,"KIR")
#Nauru
gen_chart_for_country(data_pp,"NRU")
#The default country is India
gen_chart_for_country(data_pp)
#Russia
gen_chart_for_country(data_pp,"RUS")
#USA
gen_chart_for_country(data_pp,"USA")
#China - Largest
gen_chart_for_country(data_pp,"CHN")
