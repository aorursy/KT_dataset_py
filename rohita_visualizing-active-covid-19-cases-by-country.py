import pandas as pd
import seaborn as sns

# Dictionary to store file links
files_url = {
                "Confirmed":"https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv",
                "Recovered":"https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv",
                "Deaths":"https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"
            }
def get_data(case_type, file_path):
    # Reading file from file path and storing in data frame
    file_df = pd.read_csv(file_path)
    
    # Renaming and removing columns
    file_df = file_df.rename(columns={"Country/Region":"Country"}).drop(["Lat","Long"], axis="columns")

    # Aggregating data at Country Level (dropping "Province/State" column)
    file_df = file_df.groupby("Country").sum()

    # Pivoting down data
    file_df = file_df.reset_index().melt(id_vars='Country', var_name="Date", value_name=case_type)

    # Changing Date from String to Date type
    file_df['Date'] = pd.to_datetime(file_df['Date'])
    
    return file_df
# Creating combined data frame with appended Confirmed, Recovered and Deaths data
covid19_data = pd.concat([get_data(file_type, path) for file_type, path in files_url.items()])
covid19_data = covid19_data.groupby(['Country','Date']).sum().sort_values(['Country','Date'])

# Calculating Active cases and % Cases Active
covid19_data['Active'] = covid19_data['Confirmed'] - (covid19_data['Deaths'] + covid19_data['Recovered'])
covid19_data['% Cases Active'] = covid19_data['Active']/covid19_data['Confirmed']*100

covid19_data.head()
covid19_data.tail()
# Creating Data frame to segment countries based on % Current Cases Active and % Active Cases from peak
# Calculate max_active, current_active and current_active_pct 
country_map = covid19_data.groupby(['Country']).agg(max_active = ('Active', 'max'),
                                                    current_active = ('Active', 'last'),
                                                    current_active_pct = ('% Cases Active', 'last'))
# Calculate % Active cases from peak
country_map['active_from_peak_pct'] = country_map['current_active']/country_map['max_active']*100

# Defining bin limits and segmenting Countries 
bin_active_from_peak_pct = [-1, 25, 50, 75, 90, 99, 101]
bin_active_from_peak_pct_labels = ['< 25%', '25% - 50%', '50% - 70%', '75% - 90%', '90% - 99%', '100% (Active Cases Still Rising)']
country_map['Group % Cases Active From Peak'] = pd.cut(country_map['active_from_peak_pct'], bins=bin_active_from_peak_pct , labels=bin_active_from_peak_pct_labels)

bin_active_pct = [-1, 25, 50, 75, 90, 101]
bin_active_pct_labels = ['< 25%', '25% - 50%', '50% - 75%', '75% - 90%', '> 90%']
country_map['Group % Cases Active'] = pd.cut(country_map['current_active_pct'], bins=bin_active_pct, labels=bin_active_pct_labels)

country_map.head()
country_map.tail()
max_countries_to_plot = 10
sns.set_style('ticks')
sns.set_context('talk')


# Plotting Active Cases
for country_grp in reversed(bin_active_from_peak_pct_labels):
    list_country = country_map[country_map['Group % Cases Active From Peak']==country_grp].sort_values('current_active', ascending=False).index.unique()[:max_countries_to_plot]
    plot_df = covid19_data.loc[list_country].reset_index()
    g = sns.relplot(x='Date', y='Active', data=plot_df, col='Country', col_wrap=5, col_order=list_country, hue='Country', kind='line', facet_kws={'sharey':None})
    g.set_xticklabels(rotation=45)
    g.fig.suptitle("Current Active Cases % from Peak: "+country_grp, y=1.02)
max_countries_to_plot = 10
sns.set_style('ticks')
sns.set_context('talk')


# Plotting % Active Cases of Total Confirmed Cases
for country_grp in reversed(bin_active_pct_labels):
    list_country = country_map[country_map['Group % Cases Active']==country_grp].sort_values('current_active', ascending=False).index.unique()[:max_countries_to_plot]
    plot_df = covid19_data.loc[list_country].reset_index()
    g = sns.relplot(x='Date', y='% Cases Active', data=plot_df, col='Country', col_wrap=5, col_order=list_country, hue='Country', kind='line')
    g.set_xticklabels(rotation=45)
    g.fig.suptitle("Current Active Cases % from Total Confirmed Cases: "+country_grp, y=1.02)
