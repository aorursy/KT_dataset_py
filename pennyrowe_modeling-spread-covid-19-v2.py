# .. Built-in modules
import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd

# .. My modules
from covid_resources import Covid
from covid_resources import slider_stats
from covid_resources import slider_daily
from covid_resources import slider_beds
from covid_resources import slider_beds_with_closures
from covid_resources import slider_comparison
from get_reported_covid import Reported

# %matplotlib notebook
# .. Get the data:
confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
deaths_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
recoveries_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
us_daily_df = pd.read_csv('https://covidtracking.com/api/v1/us/daily.csv')
%matplotlib notebook

# .. Parameters: feel free to change these
total_people = 1e6                     # Total number of people
number_of_days_infectious = 5.         # length of time a person is infectious (days)
days_to_resolve = 10.                  # time for case to resolve after infectious (days)
delta = 0.008                          # death rate
beta0 = 0.44                           # infections/day/person

# .. Derived terms: don't change the following lines
gamma = 1/number_of_days_infectious   
theta = 1/days_to_resolve
R0 = beta0 / gamma                     # Basic reproduction number
total_days = 600

covid1 = Covid(total_days, total_people, gamma, theta, delta, beta0)
covid1.propagate()

# .. Plot cumulative statistics with slider for beta:
#    S, I, R, D, and C
fig1, ax1 = plt.subplots(num = 1)
ax2 = ax1.twinx()  
cumulative = slider_stats(fig1, ax1, ax2, covid1)

# .. Plot daily statistics with slider for beta:
#    infected/day, recovered/day, and deaths/day
fig2, ax3 = plt.subplots(num = 2)
ax4 = ax3.twinx()
daily = slider_daily(fig2, ax3, ax4, covid1)

# .. Hospitalization rate: fee free to change these
beds_per_person = 34.7/100000        # Critical care beds in the U.S.
hospitalization_rate = 0.02*0.693    # Fraction of infectious+resolving 
                                     # hospitalized on any given day
total_days = 365*2
beta0 = 0.44

# .. Don't change the following lines
# .. Plot COVID-19 hospital beds data with a constant beta
covid2 = Covid(total_days, total_people, gamma, theta, delta, beta0, 
               beds_per_person, hospitalization_rate)
covid2.propagate()

fig3, ax5 = plt.subplots(num = 3)
beds = slider_beds(fig3, ax5, covid2)

# .. Plot COVID-19 hospital beds data with a changing beta
#    sd = social distancing
#    Feel free to change these parameters
beds_per_person = 150/1e6            # 34.7/100000        # Critical care beds in the U.S.
hospitalization_rate = 0.02*0.693    # Fraction of infectious+resolving 
total_days = 600                     # length of simulation (days)
beta0 = 0.4                          # infections/day/person before social distancing
beta_esd = 0.16                      # infections/day/person with extreeme sd
beta_rsd = 0.3                       # infections/day/person with relaxed sd
fraction_to_open = 0.5               # .05 fraction of beds occupied to trigger opening
fraction_to_distance = 0.9           # fraction of beds occupied to trigger distancing

# .. Do not change below this line
# Add the date, starting with day 0 = March 1, 2020
date_vec = [dt.datetime(2020,3,7) + dt.timedelta(i) for i in range(total_days)]

covid3 = Covid(total_days, total_people, gamma, theta, delta,
               beta0, beta_esd, beta_rsd, beds_per_person, 
               hospitalization_rate, fraction_to_distance, fraction_to_open)
covid3.propagate_changing_beta()


# .. Plot available and necessary critical care beds
fig, axs = plt.subplots(2, num = 4)  
beds_w_closures = slider_beds_with_closures(fig, axs, covid3, date_vec)

us_population = 329.       # millions

us_date = []
for i in range(len(us_daily_df['date'])):
    us_date.append(dt.datetime.strptime(str(us_daily_df['date'][i]), '%Y%m%d'))

axs[0].plot(us_date, us_daily_df['hospitalizedCurrently']/us_population, 'r')
# .. Check if the data includes your country of interest:
#    change "country_first_letter" below to the first letter of your country of interest and run the cell
country_first_letter = 'U'
deaths_df['Country/Region'][deaths_df['Country/Region'].str.contains(country_first_letter)]
# .. Set your country and population of interest
country = 'Uruguay'
population = 3.473939       # millions
# .. Plot COVID-19 hospital beds data with a changing beta
#    sd = social distancing
total_days = 200              # length of simulation (days)
beta0 = 0.4                   # infections/day/person before social distancing
beta_esd = 0.4                # infections/day/person with extreeme sd
beta_rsd = 0.4                # infections/day/person with relaxed sd
fraction_to_open = 0.05       # fraction of beds occupied to trigger opening
fraction_to_distance = 0.9    # fraction of beds occupied to trigger distancing

# .. Get reported data for your selected country
reported = Reported(confirmed_df, deaths_df, recoveries_df, country, population)

# .. Comparison to real data
covid4 = Covid(total_days, total_people, gamma, theta, delta,
               beta0, beta_esd, beta_rsd, beds_per_person, 
               hospitalization_rate, fraction_to_distance, fraction_to_open)
covid4.propagate_changing_beta()

fig, axs = plt.subplots(2, num = 5)  
stats_comp = slider_comparison(fig, axs[0], axs[1], covid4, date_vec, reported)

