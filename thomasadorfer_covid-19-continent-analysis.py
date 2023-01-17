import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None  # disable warning
train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv')

test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/test.csv')

ss = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/submission.csv')
print('Train: {0}\nTest: {1}\nSubmission: {2}'.format(train.shape, test.shape, ss.shape))
train.head()
test.head(5)
# get all country/region names

countries = set(train['Country/Region'])

print('The dataset contains {} unique countries/regions.'.format(len(countries))) 
# generate empty column 'Continent'

train.insert(loc=3, column='Continent', value="")

train.head(5)
asia = ['Bangladesh', 'Georgia', 'Uzbekistan', 'Singapore', 'Malaysia', 'Sri Lanka', 'Iraq',

        'Thailand', 'Turkey', 'Kuwait', 'Cyprus', 'Taiwan*', 'Brunei', 'Kazakhstan', 'Vietnam',

        'Israel', 'Bahrain', 'Armenia', 'Saudi Arabia', 'Iran', 'Bhutan', 'Jordan', 

        'Philippines', 'Afghanistan', 'Indonesia', 'Cambodia', 'Oman', 'Azerbaijan', 

        'Maldives', 'Nepal', 'Qatar', 'Pakistan', 'Korea, South', 'India', 'Kyrgyzstan', 

        'United Arab Emirates', 'Mongolia', 'China', 'Lebanon', 'Russia', 'Japan']



europe = ['Latvia', 'Switzerland', 'Liechtenstein', 'Italy', 'Norway', 'Austria', 'Albania',

          'United Kingdom', 'Iceland', 'Finland', 'Luxembourg', 'Belarus', 'Bulgaria', 

          'Guernsey', 'Poland', 'Moldova', 'Spain', 'Bosnia and Herzegovina', 'Portugal', 

          'Germany', 'Monaco', 'San Marino', 'Andorra', 'Slovenia', 'Montenegro', 'Ukraine',

          'Lithuania', 'Netherlands', 'Slovakia', 'Czechia', 'Malta', 'Hungary', 'Jersey', 

          'Serbia', 'Kosovo', 'France', 'Croatia', 'Sweden', 'Estonia', 'Denmark', 

          'North Macedonia', 'Greece', 'Ireland', 'Romania', 'Belgium']



southamerica = ['Peru', 'Bolivia', 'Venezuela', 'Ecuador', 'Brazil', 'Chile', 'French Guiana',

                'Paraguay', 'Uruguay', 'Colombia', 'Suriname', 'Guyana', 'Argentina']



northamerica = ['Barbados', 'Puerto Rico', 'Saint Lucia', 'Greenland', 'Antigua and Barbuda',

                'Panama', 'Honduras', 'Mexico', 'Canada', 'Costa Rica', 

                'Saint Vincent and the Grenadines', 'US', 'Dominican Republic', 'Aruba', 

                'Guadeloupe', 'Cuba', 'The Bahamas', 'Martinique', 'Trinidad and Tobago', 

                'Jamaica', 'Guatemala']



africa = ['South Africa', 'Benin', 'Congo (Brazzaville)', 'Djibouti', 'Reunion', 'Rwanda',

          'Gambia, The', 'The Gambia', 'Mayotte', 'Equatorial Guinea', 'Nigeria', 

          "Cote d'Ivoire", 'Guinea', 'Morocco', 'Somalia', 'Algeria', 'Tanzania', 'Ghana',

          'Mauritius', 'Egypt',  'Liberia', 'Congo (Kinshasa)', 'Republic of the Congo', 

          'Eswatini', 'Zambia', 'Ethiopia', 'Seychelles', 'Namibia', 'Sudan', 'Togo', 

          'Burkina Faso', 'Tunisia', 'Central African Republic', 'Mauritania', 'Cameroon',

          'Senegal', 'Kenya', 'Gabon']



oceania = ['Guam', 'New Zealand', 'Australia']



other = ['Cruise Ship','Holy See']
# double check

assert len(countries) == len(asia)+len(europe)+len(southamerica)+len(northamerica)+len(africa)+len(oceania)+len(other)
# generate dictionary

continents = {'Asia': asia,

              'Europe': europe,

              'South America': southamerica,

              'North America': northamerica,

              'Africa': africa,

              'Oceania': oceania,

              'Other': other}
for i in range(train.shape[0]):

    country = train['Country/Region'][i]

    continent = [k for k, v in continents.items() if country in v][0]

    train['Continent'][i] = continent
# inspect results

train
# get unique dates

dates = list(train['Date'])

dates_unique = set(dates)



# get continent data

def get_continent(continent):

    return train[train['Continent']==continent][['Date', 'ConfirmedCases', 'Fatalities']]



asia = get_continent('Asia')

europe = get_continent('Europe')

sa = get_continent('South America')

na = get_continent('North America')

africa = get_continent('Africa')

oceania = get_continent('Oceania')

other = get_continent('Other')
def get_continent_data(continent):

    

    "Outputs a dataframe with the sum of confirmed and fatal cases per continent for each date"

    

    confirmed_total, fatal_total = [], []



    for date in dates_unique:

        date_bool = (continent['Date']==date)

        confirmed = continent[date_bool]['ConfirmedCases'].sum()

        fatal = continent[date_bool]['Fatalities'].sum()

        confirmed_total.append(confirmed)

        fatal_total.append(fatal)



    df = pd.DataFrame([dates_unique, confirmed_total, fatal_total], index=['Date', 'Confirmed', 'Fatalities']).T

    df['Date'] = pd.to_datetime(df.Date)

    df = df.sort_values('Date').reset_index(drop=True)

    df = df.set_index('Date')



    return df
# get continent-wise data of total confirmed cases and fatalities sorted by date

asia_final = get_continent_data(asia)

europe_final = get_continent_data(europe)

sa_final = get_continent_data(sa)

na_final = get_continent_data(na)

africa_final = get_continent_data(africa)

oceania_final = get_continent_data(oceania)

other_final = get_continent_data(other)
# plot confirmed cases (does not include 'other')

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18,8))



asia_final['Confirmed'].plot(ax=axes[0,0], title='Asia')

europe_final['Confirmed'].plot(ax=axes[0,1], title='Europe')

sa_final['Confirmed'].plot(ax=axes[0,2], title='South America')

na_final['Confirmed'].plot(ax=axes[1,0], title='North America')

africa_final['Confirmed'].plot(ax=axes[1,1], title='Africa')

oceania_final['Confirmed'].plot(ax=axes[1,2], title='Oceania')

plt.subplots_adjust(hspace=0.6)

plt.show()
# plot fatalities (does not include 'other')

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18,8))



asia_final['Fatalities'].plot(ax=axes[0,0], title='Asia')

europe_final['Fatalities'].plot(ax=axes[0,1], title='Europe')

sa_final['Fatalities'].plot(ax=axes[0,2], title='South America')

na_final['Fatalities'].plot(ax=axes[1,0], title='North America')

africa_final['Fatalities'].plot(ax=axes[1,1], title='Africa')

oceania_final['Fatalities'].plot(ax=axes[1,2], title='Oceania')

plt.subplots_adjust(hspace=0.6)

plt.show()