from pandas import Series, DataFrame
import pandas as pd
confirmed_df = pd.read_csv('https://data.humdata.org/hxlproxy/api/data-preview.csv?url=https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv&filename=time_series_covid19_confirmed_global.csv')
print(confirmed_df.shape)

deaths_df = pd.read_csv('https://data.humdata.org/hxlproxy/api/data-preview.csv?url=https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv&filename=time_series_covid19_deaths_global.csv')
print(deaths_df.shape)

recovered_df = pd.read_csv('https://data.humdata.org/hxlproxy/api/data-preview.csv?url=https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv&filename=time_series_covid19_recovered_global.csv')
print(recovered_df.shape)

cases_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases_country.csv')
print(cases_df.shape)
confirmed_df.columns
cases_df.columns
confirmed_Botswana = cases_df[cases_df['Country_Region'] == "Botswana"]
confirmed_Botswana = confirmed_Botswana.drop(['Lat', 'Long_', 'Country_Region', 'Last_Update'], axis = 1)
Botswana_Summary = pd.DataFrame(confirmed_Botswana.sum()).transpose()
Botswana_Summary.style.format("{:,.0f}")
confirmed_SouthAfrica = cases_df[cases_df['Country_Region'] == "South Africa"]
confirmed_SouthAfrica = confirmed_SouthAfrica.drop(['Lat', 'Long_', 'Country_Region', 'Last_Update'], axis = 1)
SA_Summary = pd.DataFrame(confirmed_SouthAfrica.sum()).transpose()
SA_Summary.style.format("{:,.0f}")
global_data = cases_df.copy().drop(['Lat', 'Long_', 'Country_Region', 'Last_Update'], axis = 1)
global_summary = pd.DataFrame(global_data.sum()).transpose()
global_summary.style.format("{:,.0f}")
