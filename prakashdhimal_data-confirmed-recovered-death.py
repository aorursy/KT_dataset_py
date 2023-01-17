import pandas as pd
def get_raw_date():
    confirmed_df = pd.read_csv(
        'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
    recovered_df = pd.read_csv(
        'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
    death_df = pd.read_csv(
        'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
    return confirmed_df, recovered_df, death_df
def get_country_data(data, columnName, countryName):
    country_data = data.loc[data['Country/Region'] == countryName]
    country_time_series_only = country_data.drop(["Province/State", "Country/Region", "Lat", "Long"], axis=1)
    transposed = country_time_series_only.transpose()
    transposed = pd.DataFrame({
        columnName: transposed.sum(axis=1)
    })
    return transposed
def get_country_confirmed_recovered_death_data(confirmed_df, recovered_df, death_df, countryName):
    country_confirmed = get_country_data(confirmed_df, "Confirmed", countryName)
    country_recovered = get_country_data(recovered_df, "Recovered", countryName)
    country_death = get_country_data(death_df, "Death", countryName)
    country_all = country_confirmed
    country_all["Recovered"] = country_recovered.Recovered
    country_all["Death"] = country_death.Death
    return country_all
confirmed_df, recovered_df, death_df = get_raw_date()
confirmed_df[confirmed_df[confirmed_df.columns[-1]] >= 10000].loc[:, ['Country/Region',confirmed_df.columns[-1]]]
countries = list()
countries.append("US")
countries.append("Italy")
countries.append("China")
countries.append("France")
countries.append("Spain")
countries
combined_data = pd.DataFrame()
for country in countries:
    country_data = get_country_confirmed_recovered_death_data(confirmed_df, recovered_df, death_df, country)
    combined_data[country + " Confirmed"] = country_data.Confirmed
    combined_data[country + " Recovered"] = country_data.Recovered
    combined_data[country + " Death"] = country_data.Death
    combined_data[country + " Active"] = country_data.Confirmed - (country_data.Recovered + country_data.Death)
# last 10 days
combined_data.iloc[-10:]
combined_data.plot().legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
pd.DataFrame({   
    'US Confirmed': combined_data["US Confirmed"],
    'China Confirmed': combined_data["China Confirmed"],
    'Italy Confirmed': combined_data["Italy Confirmed"],
    'France Confirmed': combined_data["France Confirmed"],
    'Spain Confirmed': combined_data["Spain Confirmed"]
}).plot()
pd.DataFrame({   
    'US Death': combined_data["US Death"],
    'China Death': combined_data["China Death"],
    'Italy Death': combined_data["Italy Death"],
    'France Death': combined_data["France Death"],
    'Spain Death': combined_data["Spain Death"]
}).plot()
pd.DataFrame({   
    'US Recovered': combined_data["US Recovered"],
    'China Recovered': combined_data["China Recovered"],
    'Italy Recovered': combined_data["Italy Recovered"],
    'France Recovered': combined_data["France Recovered"],
    'Spain Recovered': combined_data["Spain Recovered"]
}).plot()
pd.DataFrame({   
    'US Active': combined_data["US Active"],
    'China Active': combined_data["China Active"],
    'Italy Active': combined_data["Italy Active"],
    'France Active': combined_data["France Active"],
    'Spain Active': combined_data["Spain Active"]
}).plot()
pd.DataFrame({   
    'US Recovered': combined_data["US Recovered"]
}).plot()


