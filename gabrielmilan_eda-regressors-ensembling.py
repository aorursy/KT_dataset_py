import math

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')



FULL_RUN = True



# Listing files

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Loading data

if FULL_RUN:

    df_train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/train.csv')

    df_test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/test.csv')

else:

    df_train = pd.read_csv('/kaggle/input/covid19-week-3-data/df_train.csv')

    df_test = pd.read_csv('/kaggle/input/covid19-week-3-data/df_test.csv')

submission_example = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/submission.csv')

df_population = pd.read_csv('/kaggle/input/population-by-country-2020/population_by_country_2020.csv')

#df_containment = pd.read_csv('/kaggle/input/covid19-containment-and-mitigation-measures/COVID 19 Containment measures data.csv')

df_health_systems = pd.read_csv('/kaggle/input/world-bank-wdi-212-health-systems/2.12_Health_systems.csv')

df_lat_lon = pd.read_csv('/kaggle/input/coronavirus-latlon-dataset/CV_LatLon_21Jan_12Mar.csv')
if FULL_RUN:

    # Getting countries with states

    import math

    has_states = set(df_train[~(df_train['Province_State'].isna())]['Country_Region'])

    common_states = set(df_train['Province_State']) & set(df_lat_lon[df_lat_lon['country'].isin(has_states)]['state'])

    common_countries = set(df_train['Country_Region']) & set(df_lat_lon['country'])

    remaining_countries = {

        'Angola' : [-11.1799566,13.2833794],

        'Antigua and Barbuda' : [17.3257025,-62.2903859],

        'Bahamas' : [24.417926,-78.2102651],

        'Barbados' : [13.1881671,-59.6052954],

        'Belize' : [17.187683,-89.4413417],

        'Benin' : [9.3003431,0.0654059],

        'Botswana' : [-22.3273954,22.4434759],

        'Burma' : [18.7811838,87.6460721],

        'Burundi' : [-3.3893677,29.3648016],

        'Cabo Verde' : [16.0202145,-25.1098509],

        'Central African Republic' : [6.6154729,18.6926929],

        'Chad' : [15.4008548,14.2402013],

        'Congo (Brazzaville)' : [-4.2471919,15.1571824,12],

        'Diamond Princess' : [35.4526321,139.4550321], # Diamond Princess is a ship, seen in Yokohama, Japan (Feb 27th)

        'Djibouti' : [11.8127758,42.0669243],

        'Dominica' : [15.4263293,-61.4975892],

        'El Salvador' : [13.7483455,-89.4906972],

        'Equatorial Guinea' : [1.1431229,6.1935546],

        'Eritrea' : [15.1764605,37.5884248],

        'Eswatini' : [-26.516566,30.9023408],

        'Ethiopia' : [9.1215001,36.00375],

        'Fiji' : [-16.5421848,177.2178571],

        'Gabon' : [-0.9230372,9.2299158],

        'Gambia' : [13.4168603,-15.9293406],

        'Ghana' : [7.8984804,-3.2743994],

        'Grenada' : [12.259767,-61.7303844],

        'Guatemala' : [15.719987,-91.3560049],

        'Guinea' : [9.92542,-13.7038879],

        'Guinea-Bissau' : [11.7002291,-15.8496604],

        'Haiti' : [19.0343549,-73.6754192],

        'Kazakhstan' : [47.6548578,57.9392984],

        'Kenya' : [0.1544419,35.6643364],

        'Kosovo' : [42.5612976,20.3416721],

        'Kyrgyzstan' : [41.2010445,72.4968368],

        'Laos' : [18.1963416,101.615389],

        'Liberia' : [6.4096257,-10.573663],

        'Libya' : [26.2900748,12.8375989],

        'MS Zaandam' : [26.1410956,-80.2156069], # It's a ship located at Fort Lauderdale (April 3rd)

        'Madagascar' : [-18.771976,42.373469],

        'Mali' : [17.5237177,-8.4809037],

        'Mauritania' : [20.959589,-15.444754],

        'Mauritius' : [-20.2030942,56.5543186],

        'Montenegro' : [42.6928556,18.832956],

        'Mozambique' : [-18.5836828,31.3118067],

        'Namibia' : [-22.9037659,13.8724459],

        'Nicaragua' : [12.866514,-86.1389169],

        'Niger' : [17.5460918,3.5859574],

        'Papua New Guinea' : [-6.3567909,145.9055506],

        'Rwanda' : [-1.9435638,29.3199833],

        'Saint Kitts and Nevis' : [16.249782,-62.284578],

        'Saint Lucia' : [13.9128128,-61.1106006],

        'Saint Vincent and the Grenadines' : [12.9714329,-61.5635867],

        'Seychelles' : [-7.0850076,48.9440464],

        'Sierra Leone' : [8.4206974,-12.9587225],

        'Somalia' : [5.2310437,41.808129],

        'Sudan' : [15.7399293,25.7594752],

        'Suriname' : [3.9826927,-57.1279423],

        'Syria' : [34.7943312,36.7594245],

        'Tanzania' : [-6.3533765,30.4940155],

        'Timor-Leste' : [-8.7889361,125.1685995],

        'Trinidad and Tobago' : [10.6962001,-61.7721494],

        'Uganda' : [1.3671063,30.059196],

        'Uruguay' : [-32.600568,-58.0278336],

        'Uzbekistan' : [41.2939152,60.0857832],

        'Venezuela' : [6.6368125,-71.1105344],

        'West Bank and Gaza' : [31.9461203,34.6667392],

        'Zambia' : [-13.101327,23.3590343],

        'Zimbabwe' : [-19.0020825,26.9090619]

    }



    def fillLat (state, country):

        if state in common_states:

            return df_lat_lon[df_lat_lon['state'] == state]['lat'].unique()[0]

        elif country in common_countries:

            return df_lat_lon[df_lat_lon['country'] == country]['lat'].unique()[0]

        elif country in remaining_countries:

            return remaining_countries[country][0]

        else:

            return float('NaN')



    def fillLon (state, country):

        if state in common_states:

            return df_lat_lon[df_lat_lon['state'] == state]['lon'].unique()[0]

        elif country in common_countries:

            return df_lat_lon[df_lat_lon['country'] == country]['lon'].unique()[0]

        elif country in remaining_countries:

            return remaining_countries[country][1]

        else:

            return float('NaN')



    df_train['Lat'] = df_train.loc[:, ['Country_Region', 'Province_State']].apply(lambda x : fillLat(x['Province_State'], x['Country_Region']), axis=1)

    df_train['Lon'] = df_train.loc[:, ['Country_Region', 'Province_State']].apply(lambda x : fillLon(x['Province_State'], x['Country_Region']), axis=1)

    df_test['Lat'] = df_test.loc[:, ['Country_Region', 'Province_State']].apply(lambda x : fillLat(x['Province_State'], x['Country_Region']), axis=1)

    df_test['Lon'] = df_test.loc[:, ['Country_Region', 'Province_State']].apply(lambda x : fillLon(x['Province_State'], x['Country_Region']), axis=1)



    # Filling Province_State column

    # Following the idea at

    # https://www.kaggle.com/ranjithks/25-lines-of-code-results-better-score#Fill-NaN-from-State-feature

    # Filling NaN states with the Country

    EMPTY_VAL = "EMPTY_VAL"

    def fillState(state, country):

        if state == EMPTY_VAL: return country

        return state

    def replaceGeorgiaState (state, country):

        if (state == 'Georgia') and (country == 'US'):

            return 'Georgia_State'

        else:

            return state

    df_train['Province_State'].fillna(EMPTY_VAL, inplace=True)

    df_train['Province_State'] = df_train.loc[:, ['Province_State', 'Country_Region']].apply(lambda x : fillState(x['Province_State'], x['Country_Region']), axis=1)

    df_train['Province_State'] = df_train.loc[:, ['Province_State', 'Country_Region']].apply(lambda x : replaceGeorgiaState(x['Province_State'], x['Country_Region']), axis=1)

    df_test['Province_State'].fillna(EMPTY_VAL, inplace=True)

    df_test['Province_State'] = df_test.loc[:, ['Province_State', 'Country_Region']].apply(lambda x : fillState(x['Province_State'], x['Country_Region']), axis=1)

    df_test['Province_State'] = df_test.loc[:, ['Province_State', 'Country_Region']].apply(lambda x : replaceGeorgiaState(x['Province_State'], x['Country_Region']), axis=1)



    # Checking for missing values

    missing_values = df_train.isnull().sum()

    print("Train missing values:\n{}".format(missing_values[missing_values>0].sort_values(ascending = False)))

    missing_values = df_test.isnull().sum()

    print("Test missing values:\n{}".format(missing_values[missing_values>0].sort_values(ascending = False)))
if FULL_RUN:

    import folium

    from folium.plugins import HeatMap

    geo_df = df_train.groupby(["Province_State"]).max()[['ConfirmedCases', 'Fatalities', 'Lat', 'Lon']]

    max_amount = float(geo_df['ConfirmedCases'].max())

    hmap = folium.Map(location = [0, 0], zoom_start = 2.4, tiles="CartoDB positron")

    title_html = '<h3 align="center" style="font-size:20px"><b>COVID-19 ConfirmedCases Heatmap ({})</b></h3>'.format(df_train['Date'].max())

    hmap.get_root().html.add_child(folium.Element(title_html))

    hm_wide = HeatMap (list (zip (geo_df.Lat.values, geo_df.Lon.values, geo_df.ConfirmedCases.values)),

                       min_opacity = 0.4,

                       max_val = max_amount,

                           radius = 17,

                       blur = 15,

                       max_zoom = 1

                      )

    hmap.add_child(hm_wide)
if FULL_RUN:

    common_countries = set (df_train['Country_Region']) & set(df_population['Country (or dependency)'])

    remaining_countries = {

        'Czechia': {

            'population' : 10650000,

            'density' : 134,

            'area' : 78865,

            'medage' : 41,

        },

        'Saint Vincent and the Grenadines': {

            'population' : 109897,

            'density' : 284,

            'area' : 389,

            'medage' : 33,

        },

        'West Bank and Gaza': {

            'population' : 3340143,

            'density' : 13,

            'area' : 5655,

            'medage' : 17,

        },

        'Kosovo': {

            'population' : 1831000,

            'density' : 159,

            'area' : 168,

            'medage' : 29,

        },

        'MS Zaandam': {

            'population' : 1400,

            'density' : 18178,

            'area' : 0.0770148,    # Approximation

            'medage' : 40,         # Couldn't find this

        },

        'Burma': {

            'population' : 54410000,

            'density' : 79,

            'area' : 676578,

            'medage' : 29,

        },

        'Diamond Princess': {

            'population' : 3711,

            'density' : 20073,     # It's large 'cause it's a boat

            'area' : 0.184875,     # Approximation

            'medage' : 30,         # Couldn't find this

        },

        'Saint Kitts and Nevis': {

            'population' : 53199,

            'density' : 204,

            'area' : 261,

            'medage' : 36,

        },

        "Cote d'Ivoire": {

            'population' : 24290000,

            'density' : 83,

            'area' : 322463,

            'medage' : 20,

        },

    }

    def fillPopulation (country):

        if country == "US":

            return df_population[df_population['Country (or dependency)'] == "United States"]['Population (2020)'].unique()[0]

        elif country.startswith("Congo"):

            return df_population[df_population['Country (or dependency)'] == "Congo"]['Population (2020)'].unique()[0]

        elif country == "Taiwan*":

            return df_population[df_population['Country (or dependency)'] == "Taiwan"]['Population (2020)'].unique()[0]

        elif country == "Korea, South":

            return df_population[df_population['Country (or dependency)'] == "South Korea"]['Population (2020)'].unique()[0]

        elif country in common_countries:

            return df_population[df_population['Country (or dependency)'] == country]['Population (2020)'].unique()[0]

        elif country in remaining_countries:

            return remaining_countries[country]['population']

        else:

            return float('NaN')



    def fillDensity (country):

        if country == "US":

            return df_population[df_population['Country (or dependency)'] == "United States"]['Density (P/Km²)'].unique()[0]

        elif country.startswith("Congo"):

            return df_population[df_population['Country (or dependency)'] == "Congo"]['Density (P/Km²)'].unique()[0]

        elif country == "Taiwan*":

            return df_population[df_population['Country (or dependency)'] == "Taiwan"]['Density (P/Km²)'].unique()[0]

        elif country == "Korea, South":

            return df_population[df_population['Country (or dependency)'] == "South Korea"]['Density (P/Km²)'].unique()[0]

        elif country in common_countries:

            return df_population[df_population['Country (or dependency)'] == country]['Density (P/Km²)'].unique()[0]

        elif country in remaining_countries:

            return remaining_countries[country]['density']

        else:

            return float('NaN')



    def fillArea (country):

        if country == "US":

            return df_population[df_population['Country (or dependency)'] == "United States"]['Land Area (Km²)'].unique()[0]

        elif country.startswith("Congo"):

            return df_population[df_population['Country (or dependency)'] == "Congo"]['Land Area (Km²)'].unique()[0]

        elif country == "Taiwan*":

            return df_population[df_population['Country (or dependency)'] == "Taiwan"]['Land Area (Km²)'].unique()[0]

        elif country == "Korea, South":

            return df_population[df_population['Country (or dependency)'] == "South Korea"]['Land Area (Km²)'].unique()[0]

        elif country in common_countries:

            return df_population[df_population['Country (or dependency)'] == country]['Land Area (Km²)'].unique()[0]

        elif country in remaining_countries:

            return remaining_countries[country]['area']

        else:

            return float('NaN')



    def fillMedAge (country):

        if country == "Andorra":

            return 45

        elif country == "Dominica":

            return 34

        elif country == "Holy See":

            return 60

        elif country == "Liechtenstein":

            return 41

        elif country == "Monaco":

            return 52

        elif country == "San Marino":

            return 45

        elif country == "US":

            return df_population[df_population['Country (or dependency)'] == "United States"]['Med. Age'].unique()[0]

        elif country.startswith("Congo"):

            return df_population[df_population['Country (or dependency)'] == "Congo"]['Med. Age'].unique()[0]

        elif country == "Taiwan*":

            return df_population[df_population['Country (or dependency)'] == "Taiwan"]['Med. Age'].unique()[0]

        elif country == "Korea, South":

            return df_population[df_population['Country (or dependency)'] == "South Korea"]['Med. Age'].unique()[0]

        elif country in common_countries:

            return df_population[df_population['Country (or dependency)'] == country]['Med. Age'].unique()[0]

        elif country in remaining_countries:

            return remaining_countries[country]['medage']

        else:

            return float('NaN')



    df_train['Population'] = df_train.loc[:, ['Country_Region']].apply(lambda x : fillPopulation(x['Country_Region']), axis=1)

    df_test['Population'] = df_test.loc[:, ['Country_Region']].apply(lambda x : fillPopulation(x['Country_Region']), axis=1)

    df_train['Density'] = df_train.loc[:, ['Country_Region']].apply(lambda x : fillDensity(x['Country_Region']), axis=1)

    df_test['Density'] = df_test.loc[:, ['Country_Region']].apply(lambda x : fillDensity(x['Country_Region']), axis=1)

    df_train['Area'] = df_train.loc[:, ['Country_Region']].apply(lambda x : fillArea(x['Country_Region']), axis=1)

    df_test['Area'] = df_test.loc[:, ['Country_Region']].apply(lambda x : fillArea(x['Country_Region']), axis=1)

    df_train['MedAge'] = df_train.loc[:, ['Country_Region']].apply(lambda x : fillMedAge(x['Country_Region']), axis=1)

    df_test['MedAge'] = df_test.loc[:, ['Country_Region']].apply(lambda x : fillMedAge(x['Country_Region']), axis=1)



    # Checking for missing values

    missing_values = df_train.isnull().sum()

    print("Train missing values:\n{}".format(missing_values[missing_values>0].sort_values(ascending = False)))

    missing_values = df_test.isnull().sum()

    print("Test missing values:\n{}".format(missing_values[missing_values>0].sort_values(ascending = False)))
if FULL_RUN:

    # Creating Cases/Density column

    cases_density_df = df_train.groupby(["Province_State"]).max()[['ConfirmedCases', 'Density']]

    cases_density_df['Cases_Density'] = cases_density_df.loc[:, ['ConfirmedCases', 'Density']].apply(lambda x : x['ConfirmedCases'] / x['Density'], axis=1)



    # Making plot

    plt.figure(figsize=(16,8))

    plt.title("Number of COVID-19 confirmed cases")

    plt.xlabel("Confirmed Cases")

    plt.barh(cases_density_df.sort_values(by='ConfirmedCases', ascending = False).iloc[:10].sort_values(by='ConfirmedCases', ascending = True).index, cases_density_df.sort_values(by='ConfirmedCases', ascending = False).iloc[:10].sort_values(by='ConfirmedCases', ascending = True)['ConfirmedCases'], color='#ffb3b3')

    for i, v in enumerate(cases_density_df.sort_values(by='ConfirmedCases', ascending = False).iloc[:10].sort_values(by='ConfirmedCases', ascending = True)['ConfirmedCases']):

        plt.text(v + 100, i-0.1, str(int(v)), color='#880000')

    plt.show()
if FULL_RUN:

    # Making plot

    plt.figure(figsize=(16,8))

    plt.title("Number of COVID-19 confirmed cases per country density")

    plt.xlabel("Confirmed Cases / Density")

    plt.barh(cases_density_df.sort_values(by='Cases_Density', ascending = False).iloc[:10].sort_values(by='Cases_Density', ascending = True).index, cases_density_df.sort_values(by='Cases_Density', ascending = False).iloc[:10].sort_values(by='Cases_Density', ascending = True)['Cases_Density'], color='#ffb3b3')

    for i, v in enumerate(cases_density_df.sort_values(by='Cases_Density', ascending = False).iloc[:10].sort_values(by='Cases_Density', ascending = True)['Cases_Density']):

        plt.text(v + 50, i-0.1, str(np.round(v,2)), color='#880000')

    plt.show()



    # Filling df_train and df_test

    df_train['Cases_Density'] = df_train.loc[:, ['Province_State']].apply(lambda x : cases_density_df.loc[x['Province_State']]['Cases_Density'], axis=1)

    df_test['Cases_Density'] = df_test.loc[:, ['Province_State']].apply(lambda x : cases_density_df.loc[x['Province_State']]['Cases_Density'], axis=1)



    # Checking for missing values

    missing_values = df_train.isnull().sum()

    print("Train missing values:\n{}".format(missing_values[missing_values>0].sort_values(ascending = False)))

    missing_values = df_test.isnull().sum()

    print("Test missing values:\n{}".format(missing_values[missing_values>0].sort_values(ascending = False)))
if FULL_RUN:

    def getFatalRatio (country):

        df = df_train[df_train['Country_Region'] == country]

        cases = df['ConfirmedCases'].max()

        fatal = df['Fatalities'].max()

        if cases <= 0:

            return 0

        else:

            return fatal/cases



    df_train['FatalRatio'] = df_train.loc[:, ['Country_Region']].apply(lambda x : getFatalRatio(x['Country_Region']), axis=1)

    df_test['FatalRatio'] = df_test.loc[:, ['Country_Region']].apply(lambda x : getFatalRatio(x['Country_Region']), axis=1)



    plot_df = df_train.groupby(["Province_State", 'Country_Region']).max()[['FatalRatio', 'MedAge']]

    plot_df = plot_df[plot_df['FatalRatio'] > 0]

    plt.figure(figsize=(16,8))

    plt.title("Fatality ratio vs. Median age")

    plt.xlabel("Median age")

    plt.ylabel("Fatality ratio")

    plt.scatter(plot_df['MedAge'].astype(int), plot_df['FatalRatio'].values)

    plt.show()
if FULL_RUN:

    # Generating new features

    df_train['ConfirmedCases_Hat'] = df_train['ConfirmedCases'] - df_train['ConfirmedCases'].shift(1)

    df_train['ConfirmedCases_Hat'] = df_train['ConfirmedCases_Hat'].apply(lambda x: 0 if x < 0 else x)

    df_train['ConfirmedCases_Hat'] = df_train['ConfirmedCases_Hat'].fillna(0)

    df_train['ConfirmedCases_Hat_Hat'] = df_train['ConfirmedCases_Hat'] - df_train['ConfirmedCases_Hat'].shift(1)

    df_train['ConfirmedCases_Hat_Hat'] = df_train['ConfirmedCases_Hat_Hat'].fillna(0)

    df_train['Fatalities_Hat'] = df_train['Fatalities'] - df_train['Fatalities'].shift(1)

    df_train['Fatalities_Hat'] = df_train['Fatalities_Hat'].apply(lambda x: 0 if x < 0 else x)

    df_train['Fatalities_Hat'] = df_train['Fatalities_Hat'].fillna(0)

    df_train['Fatalities_Hat_Hat'] = df_train['Fatalities_Hat'] - df_train['Fatalities_Hat'].shift(1)

    df_train['Fatalities_Hat_Hat'] = df_train['Fatalities_Hat_Hat'].fillna(0)



    # Getting the most critical

    most_critical = set(df_train.groupby(["Province_State"]).max().sort_values(by='Cases_Density', ascending = False).iloc[:8].index)

    xticks = df_train['Date'].unique()

    i = 0

    for tick in xticks:

        if i == 1:

            xticks[xticks==tick] = ''

            i = 0

        else:

            i = 1



    plot_df = df_train[df_train['Province_State'].isin(most_critical)]

    plt.figure(figsize=(16,10))

    plt.title("COVID-19 confirmed cases over time")

    plt.xlabel("Date")

    plt.ylabel("Confirmed cases")

    for state in most_critical:

        plt.plot(plot_df[plot_df['Province_State'] == state]['Date'], plot_df[plot_df['Province_State'] == state]['ConfirmedCases'])

    plt.legend(most_critical)

    plt.xticks(xticks, rotation=45)

    plt.show()
if FULL_RUN:

    plot_df = df_train[df_train['Province_State'].isin(most_critical)]

    plt.figure(figsize=(16,10))

    plt.title("COVID-19 fatalities over time")

    plt.xlabel("Date")

    plt.ylabel("Fatalities")

    for state in most_critical:

        plt.plot(plot_df[plot_df['Province_State'] == state]['Date'], plot_df[plot_df['Province_State'] == state]['Fatalities'])

    plt.legend(most_critical)

    plt.xticks(xticks, rotation=45)

    plt.show()
if FULL_RUN:

    from scipy.interpolate import make_interp_spline, BSpline

    from datetime import datetime



    plot_df = df_train.groupby(["Date"]).sum()

    plt.figure(figsize=(16,10))

    plt.title("Worldwide COVID-19 Confirmed Cases")

    plt.xlabel("Date")

    x = plot_df.index

    y = plot_df['ConfirmedCases']

    plt.plot(x, y)

    x = pd.to_datetime(plot_df.index)

    y = plot_df['ConfirmedCases_Hat']

    xnew = np.linspace(0, len(x), 300)

    spl = make_interp_spline(range(len(x)), y, k=2)  # type: BSpline

    power_smooth = spl(xnew)

    plt.plot(xnew, power_smooth)

    plt.yscale('log')

    plt.xticks(xticks, rotation=45)

    plt.legend(['Confirmed cases', 'First derivative (smoothed)'])

    plt.show()
if FULL_RUN:

    plt.figure(figsize=(16,10))

    plt.title("Worldwide COVID-19 Fatalities")

    plt.xlabel("Date")

    x = plot_df.index

    y = plot_df['Fatalities']

    plt.plot(x, y)

    x = pd.to_datetime(plot_df.index)

    y = plot_df['Fatalities_Hat']

    xnew = np.linspace(0, len(x), 300)

    spl = make_interp_spline(range(len(x)), y, k=2)  # type: BSpline

    power_smooth = spl(xnew)

    plt.plot(xnew, power_smooth)

    plt.yscale('log')

    plt.xticks(xticks, rotation=45)

    plt.legend(['Fatalities', 'First derivative (smoothed)'])

    plt.show()
if FULL_RUN:

    # == North:

    #  - Spring runs from March 1 to May 31;

    #  - Summer runs from June 1 to August 31;

    #  - Fall (autumn) runs from September 1 to November 30; and

    #  - Winter runs from December 1 to February 28 (February 29 in a leap year).

    #

    # == South:

    #  - Spring starts September 1 and ends November 30;

    #  - Summer starts December 1 and ends February 28 (February 29 in a Leap Year);

    #  - Fall (autumn) starts March 1 and ends May 31; and

    #  - Winter starts June 1 and ends August 31;

    def getSeason (latitude, date):

        month = pd.to_datetime(date).month

        # North

        if latitude >= 0:

            # Spring

            if ((month >= 3) and (month <= 5)):

                return "spring"

            # Summer

            elif ((month >= 6) and (month <= 8)):

                return "summer"

            # Fall

            elif ((month >= 9) and (month <= 11)):

                return "fall"

            # Winter

            else:

                return "winter"

        # South

        else:

            # Fall

            if ((month >= 3) and (month <= 5)):

                return "fall"

            # Winter

            elif ((month >= 6) and (month <= 8)):

                return "winter"

            # Spring

            elif ((month >= 9) and (month <= 11)):

                return "spring"

            # Summer

            else:

                return "summer"



    #getSeason(1, df_train['Date'].unique()[0])

    df_train['Season'] = df_train.loc[:, ['Lat', 'Date']].apply(lambda x : getSeason(x['Lat'], x['Date']), axis=1)

    df_test['Season'] = df_test.loc[:, ['Lat', 'Date']].apply(lambda x : getSeason(x['Lat'], x['Date']), axis=1)



    # Checking for missing values

    missing_values = df_train.isnull().sum()

    print("Train missing values:\n{}".format(missing_values[missing_values>0].sort_values(ascending = False)))

    missing_values = df_test.isnull().sum()

    print("Test missing values:\n{}".format(missing_values[missing_values>0].sort_values(ascending = False)))
if FULL_RUN:

    # Getting missing countries

    missing_countries = set(df_train['Country_Region']) - set(df_health_systems['Country_Region'])

    print ("Missing countries are: {}".format(missing_countries))
if FULL_RUN:

    desired_columns = [

        'Health_exp_pct_GDP_2016',              # Gives the idea on how much is spent on health/GDP

        'Health_exp_public_pct_2016',           # Tells how much comes from public sources

        'Health_exp_per_capita_USD_2016',       # Expenditures on health per capita

        'Physicians_per_1000_2009-18',          # Number of physicians/10K

        'Nurse_midwife_per_1000_2009-18',       # Number of nurses/midwives/10K

        'Specialist_surgical_per_1000_2008-18', # Number of surgeons/10K

        'Completeness_of_death_reg_2008-16',    # Percentage of death correctly registered

    ]

    desired_names = [

        'Health_GDP',

        'Health_Public',

        'Health_USD',

        'Physicians',

        'Nurses',

        'Surgeons',

        'DeathCompleteness'

    ]

    df_health_systems = pd.read_csv('/kaggle/input/world-bank-wdi-212-health-systems/2.12_Health_systems.csv')

    df_health_systems = df_health_systems[['Country_Region', 'Province_State'] + desired_columns]

    missing_countries = set(missing_countries)

    health_states = set(df_health_systems['Province_State'])

    health_countries = set(df_health_systems['Country_Region'])



    def getFeature (country, state, feature_name):

        if country in set(missing_countries):

            return df_health_systems[feature_name].mean()

        elif state in health_states:

            return df_health_systems[df_health_systems['Province_State'] == state][feature_name].mean()

        elif country in health_countries:

            return df_health_systems[df_health_systems['Country_Region'] == country][feature_name].mean()

        else:

            return float('NaN')



    for i in range(len(desired_columns)):

        feature_name = desired_columns[i]

        desired_name = desired_names[i]

        print ("Getting feature {}".format(desired_name))

        df_train[desired_name] = df_train.loc[:, ['Country_Region', 'Province_State']].apply(lambda x : getFeature(x['Country_Region'], x['Province_State'], feature_name), axis=1)

        df_test[desired_name] = df_test.loc[:, ['Country_Region', 'Province_State']].apply(lambda x : getFeature(x['Country_Region'], x['Province_State'], feature_name), axis=1)



    # Checking for missing values

    missing_values = df_train.isnull().sum()

    print("Train missing values:\n{}".format(missing_values[missing_values>0].sort_values(ascending = False)))

    missing_values = df_test.isnull().sum()

    print("Test missing values:\n{}".format(missing_values[missing_values>0].sort_values(ascending = False)))
if FULL_RUN:

    # Filling missing with column mean

    for i in range(len(desired_columns)):

        df_train[desired_names[i]] = df_train[desired_names[i]].fillna(df_health_systems[desired_columns[i]].mean())

        df_test[desired_names[i]] = df_test[desired_names[i]].fillna(df_health_systems[desired_columns[i]].mean())



    # Checking for missing values

    missing_values = df_train.isnull().sum()

    print("Train missing values:\n{}".format(missing_values[missing_values>0].sort_values(ascending = False)))

    missing_values = df_test.isnull().sum()

    print("Test missing values:\n{}".format(missing_values[missing_values>0].sort_values(ascending = False)))
if FULL_RUN:

    import seaborn as sns

    sns.pairplot(df_train,

                 y_vars = ['ConfirmedCases', 'Fatalities', 'FatalRatio'],

                 x_vars = desired_names,

                 diag_kind="kde",

                 palette="husl"

                )

    plt.show()
if FULL_RUN:

    # Making Nurse plot

    plot_df = df_train.groupby(["Province_State"]).max()[['Nurses', 'Health_GDP']].sort_values(by='Nurses', ascending = False).iloc[:10].sort_values(by='Nurses', ascending = True)

    fig, ax = plt.subplots(figsize=(16,8))

    plt.title("Nurses per 1000 people")

    plt.xlabel("Nurses/1K")

    ax.barh(plot_df.index, plot_df['Nurses'], color='#99e699')

    for i, v in enumerate(plot_df['Nurses']):

        plt.text(v + 0.1, i-0.1, str(np.round(v, 1)), color='#1f7a1f')

    plt.show()
if FULL_RUN:

    df = df_train.groupby(["Province_State"]).max()

    brazil = df.loc['Brazil']

    brazil
if FULL_RUN:

    # Making Date become number

    import time

    from datetime import datetime

    df_train['Date'] = pd.to_datetime(df_train['Date'])

    df_test['Date'] = pd.to_datetime(df_test['Date'])

    df_train['Date'] = df_train['Date'].apply(lambda s: time.mktime(s.timetuple()))

    df_test['Date'] = df_test['Date'].apply(lambda s: time.mktime(s.timetuple()))

    min_timestamp = np.min(df_train['Date'])

    df_train['Date'] = df_train['Date'].apply(lambda s: (s - min_timestamp) / 86400.0)

    df_test['Date'] = df_test['Date'].apply(lambda s: (s - min_timestamp) / 86400.0)
if FULL_RUN:

    import time

    from tqdm import tqdm



    start_time = time.time()

    lag_range = np.arange(1,15,1)

    states = set(df_train['Province_State'])



    with tqdm(total = len(list(states))) as pbar:

        for state in states:

            for d in df_train['Date'].drop_duplicates().astype('int'):

                mask = (df_train['Date'] == d) & (df_train['Province_State'] == state)

                for lag in lag_range:

                    mask_org = (df_train['Date'] == (d - lag)) & (df_train['Province_State'] == state)

                    try:

                        df_train.loc[mask, 'ConfirmedCases_Lag_' + str(lag)] = df_train.loc[mask_org, 'ConfirmedCases'].values

                    except:

                        df_train.loc[mask, 'ConfirmedCases_Lag_' + str(lag)] = 0

                    try:

                        df_train.loc[mask, 'Fatalities_Lag_' + str(lag)] = df_train.loc[mask_org, 'Fatalities'].values

                    except:

                        df_train.loc[mask, 'Fatalities_Lag_' + str(lag)] = 0

            pbar.update(1)

    print('Time spent for building features is {} minutes'.format(round((time.time()-start_time)/60,1)))

if FULL_RUN:

    # Never forget to add'em into your test dataset

    missing_cols = set(df_train.columns) - set(df_test.columns) - set(['Id'])

    print ("Do not forget to add these columns into your test dataset: {}".format(missing_cols))
if FULL_RUN:

    for col in missing_cols:

        df_test[col] = -1

    missing_cols = set(df_train.columns) - set(df_test.columns) - set(['Id'])

    if (missing_cols == set()):

        print ("No remaining missing columns on test dataset!")

    else:

        print ("Something's gone wrong, these are missing: {}".format(missing_cols))
if FULL_RUN:

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64', 'uint8']



    all_columns = set(df_train.columns)

    numeric_columns = set(df_train.select_dtypes(include=numerics).columns)

    remaining_columns = all_columns - numeric_columns

    print ("Non-numerical columns: {}".format(remaining_columns))
if FULL_RUN:

    df_train['MedAge'] = df_train['MedAge'].astype(int)

    df_test['MedAge'] = df_test['MedAge'].astype(int)

    numeric_columns = set(df_train.select_dtypes(include=numerics).columns)

    remaining_columns = all_columns - numeric_columns

    print ("Non-numerical columns: {}".format(remaining_columns))
if FULL_RUN:

    df_train['Location'] = ['_'.join(x) for x in zip(df_train['Country_Region'], df_train['Province_State'])]

    df_test['Location'] = ['_'.join(x) for x in zip(df_test['Country_Region'], df_test['Province_State'])]

    df_train.drop(columns=['Country_Region', 'Province_State'], inplace=True)

    df_test.drop(columns=['Country_Region', 'Province_State'], inplace=True)
if FULL_RUN:

    # One-hot encoding

    df_train = pd.concat([df_train,pd.get_dummies(df_train['Location'], prefix='Location',dummy_na=False)],axis=1).drop(['Location'],axis=1)

    df_test = pd.concat([df_test,pd.get_dummies(df_test['Location'], prefix='Location',dummy_na=False)],axis=1).drop(['Location'],axis=1)

    df_train.shape, df_test.shape
if FULL_RUN:

    all_columns = set(df_train.columns)

    numeric_columns = set(df_train.select_dtypes(include=numerics).columns)

    remaining_columns = all_columns - numeric_columns

    print ("Non-numerical columns: {}".format(remaining_columns))
if FULL_RUN:

    df_train = pd.concat([df_train,pd.get_dummies(df_train['Season'], prefix='Season',dummy_na=False)],axis=1).drop(['Season'],axis=1)

    df_test = pd.concat([df_test,pd.get_dummies(df_test['Season'], prefix='Season',dummy_na=False)],axis=1).drop(['Season'],axis=1)

    df_train.shape, df_test.shape
if FULL_RUN:

    missing_seasons = (set(df_train.columns) - set(df_test.columns))

    for col in missing_seasons:

        if col.startswith('Season'):

            df_test[col] = 0

    df_train.shape, df_test.shape
if FULL_RUN:

    # Double checking for test dataset columns

    all_columns = set(df_test.columns)

    numeric_columns = set(df_test.select_dtypes(include=numerics).columns)

    remaining_columns = all_columns - numeric_columns

    print ("Non-numerical columns: {}".format(remaining_columns))
if FULL_RUN:

    # Saving DFs

    df_train.to_csv('df_train.csv')

    df_test.to_csv('df_test.csv')
from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import LassoCV



all_columns = list(df_train.columns)

for c in ['Id', 'ConfirmedCases', 'Fatalities']:

    all_columns.remove(c)

for c in [x for x in df_train.columns if 'Hat' in x]:

    all_columns.remove(c)    



X = df_train[all_columns]

y_cases = df_train['ConfirmedCases']

y_fatal = df_train['Fatalities']



X_scaler = StandardScaler()

X = X_scaler.fit_transform(X)



print (" * Fitting ConfirmedCases")

threshold = 0.25

clf = LassoCV()

sfm = SelectFromModel(clf, threshold=threshold)

sfm.fit(X, y_cases)

n_features = sfm.transform(X).shape[1]

print ("   - Got {} features from threshold {}".format(n_features, threshold))

while ((threshold < .95) and (n_features > 10)):

    threshold += .05

    sfm = SelectFromModel(clf, threshold=threshold)

    sfm.fit(X, y_cases)

    X_cases = sfm.transform(X)

    n_features = sfm.transform(X).shape[1]

    print ("   - Got {} features from threshold {}".format(n_features, threshold))

cases_cols = []

mask = sfm.get_support()

for i in range(len(all_columns)):

    if mask[i]:

        cases_cols.append(all_columns[i])

print ("   - For ConfirmedCases, you'll want {}".format(cases_cols))

    

print (" * Fitting Fatalities")

threshold = 0.25

clf = LassoCV()

sfm = SelectFromModel(clf, threshold=threshold)

sfm.fit(X, y_fatal)

n_features = sfm.transform(X).shape[1]

print ("   - Got {} features from threshold {}".format(n_features, threshold))

while ((threshold < .95) and (n_features > 10)):

    threshold += .05

    sfm = SelectFromModel(clf, threshold=threshold)

    sfm.fit(X, y_fatal)

    X_fatal = sfm.transform(X)

    n_features = sfm.transform(X).shape[1]

    print ("   - Got {} features from threshold {}".format(n_features, threshold))

fatal_cols = []

mask = sfm.get_support()

for i in range(len(all_columns)):

    if mask[i]:

        fatal_cols.append(all_columns[i])

print ("   - For Fatalities, you'll want {}".format(fatal_cols))
from sklearn.feature_selection import RFECV

from sklearn.linear_model import BayesianRidge



X_cases = df_train[all_columns]

X_fatal = df_train[all_columns]



X_cases_scaler = StandardScaler()

X_fatal_scaler = StandardScaler()

X_cases = X_cases_scaler.fit_transform(X_cases)

X_fatal = X_fatal_scaler.fit_transform(X_fatal)



model = BayesianRidge()



# print (" * Fitting ConfirmedCases...")

# selector_cases = RFECV (model)

# selector_cases.fit(X_cases, y_cases)

# cases_mask = selector_cases.support_

# new_cases_cols = []

# for i in range(len(cases_mask)):

#     if cases_mask[i]:

#         new_cases_cols.append(all_columns[i])

# print ("   - Wanted features are: {}".format(new_cases_cols))

new_cases_cols = ['ConfirmedCases_Lag_1', 'ConfirmedCases_Lag_2', 'Fatalities_Lag_2', 'ConfirmedCases_Lag_3', 'Fatalities_Lag_3', 'ConfirmedCases_Lag_4', 'Fatalities_Lag_4']

        

# print (" * Fitting Fatalities...")

# selector_fatal = RFECV (model)

# selector_fatal.fit(X_fatal, y_fatal)

# fatal_mask = selector_fatal.support_

# new_fatal_cols = []

# for i in range(len(fatal_mask)):

#     if fatal_mask[i]:

#         new_fatal_cols.append(all_columns[i])

# print ("   - Wanted features are: {}".format(new_fatal_cols))

new_fatal_cols = ['ConfirmedCases_Lag_1', 'Fatalities_Lag_1', 'ConfirmedCases_Lag_2', 'Fatalities_Lag_2', 'Fatalities_Lag_3', 'ConfirmedCases_Lag_4', 'Fatalities_Lag_4', 'Fatalities_Lag_5', 'Fatalities_Lag_7', 'Fatalities_Lag_8', 'Fatalities_Lag_10', 'Fatalities_Lag_11', 'ConfirmedCases_Lag_13', 'Fatalities_Lag_13', 'ConfirmedCases_Lag_14', 'Fatalities_Lag_14']
# Heatmap of positive correlation features

import seaborn as sns

correlation = df_train.corr()

k = len([i for i in correlation['ConfirmedCases'] if abs(i) >= 0.75])

cols = correlation.nlargest(k,'ConfirmedCases')['ConfirmedCases'].index

cm = np.corrcoef(df_train[cols].values.T)

f , ax = plt.subplots(figsize = (18,16))

sns.heatmap(cm, vmax=.8, linewidths=0.01,square=True,annot=True,cmap='viridis',

            linecolor="white",xticklabels = cols.values ,annot_kws = {'size':12},yticklabels = cols.values)

ax.set_title('ConfirmedCases correlation heatmap')

plt.show()

my_cases_cols = list(cols)

for c in ['Id', 'ConfirmedCases', 'Fatalities']:

    try:

        my_cases_cols.remove(c)

    except ValueError:

        pass

for c in [x for x in df_train.columns if 'Hat' in x]:

    try:

        my_cases_cols.remove(c)

    except ValueError:

        pass
k = len([i for i in correlation['Fatalities'] if abs(i) >= 0.75])

cols = correlation.nlargest(k,'Fatalities')['Fatalities'].index

cm = np.corrcoef(df_train[cols].values.T)

f , ax = plt.subplots(figsize = (18,16))

sns.heatmap(cm, vmax=.8, linewidths=0.01,square=True,annot=True,cmap='viridis',

            linecolor="white",xticklabels = cols.values ,annot_kws = {'size':12},yticklabels = cols.values)

ax.set_title('Fatalities correlation heatmap')

plt.show()

my_fatal_cols = list(cols)

for c in ['Id', 'ConfirmedCases', 'Fatalities']:

    try:

        my_fatal_cols.remove(c)

    except ValueError:

        pass

for c in [x for x in df_train.columns if 'Hat' in x]:

    try:

        my_fatal_cols.remove(c)

    except ValueError:

        pass
index = df_test[df_test['Date'] <= np.max(df_train['Date'])].index

df_intersection = df_train[df_train['Date'] >= np.min(df_test['Date'])]

df_intersection.set_index(index)

df_intersection['ForecastId'] = df_test[df_test['Date'] <= np.max(df_train['Date'])]['ForecastId'].values

df_intersection.drop(columns=['Id'], inplace=True)

df_intersection.head()
df_train = df_train[df_train['Date'] < np.min(df_test['Date'])]

df_train.tail()
# Create scalers dict

scaler = {}

# Set cases train, valid

X_cases = df_train[cases_cols]

# Set fatal train, valid

X_fatal = df_train[fatal_cols]

# Set scalers

scaler['cases'] = []

scaler['cases'].append(StandardScaler())

scaler['fatal'] = []

scaler['fatal'].append(StandardScaler())

# Scaling cases

X_cases_one = scaler['cases'][0].fit_transform(X_cases)

# Scaling fatal

X_fatal_one = scaler['fatal'][0].fit_transform(X_fatal)

# Set cases train, valid

X_cases = df_train[new_cases_cols]

# Set fatal train, valid

X_fatal = df_train[new_fatal_cols]

# Scaling cases

scaler['cases'].append(StandardScaler())

X_cases_two = scaler['cases'][1].fit_transform(X_cases)

# Scaling fatal

scaler['fatal'].append(StandardScaler())

X_fatal_two = scaler['fatal'][1].fit_transform(X_fatal)

# Set cases train, valid

X_cases = df_train[my_cases_cols]

# Set fatal train, valid

X_fatal = df_train[my_fatal_cols]

# Scaling cases

scaler['cases'].append(StandardScaler())

X_cases_three = scaler['cases'][2].fit_transform(X_cases)

# Scaling fatal

scaler['fatal'].append(StandardScaler())

X_fatal_three = scaler['fatal'][2].fit_transform(X_fatal)

# Getting y

y_cases = df_train['ConfirmedCases']

y_fatal = df_train['Fatalities']



X_train = {

    'cases' : [X_cases_one, X_cases_two, X_cases_three],

    'fatal' : [X_cases_two, X_fatal_two, X_fatal_three]

}



y_train = {

    'cases' : y_cases,

    'fatal' : y_fatal

}
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor, BayesianRidge, Lasso, LassoLars, ElasticNet, TheilSenRegressor

from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_log_error

import keras.backend as K



def root_mean_squared_log_error(y_true, y_pred):

    return K.sqrt(K.mean(K.square(K.log(y_pred + 1) - K.log(y_true + 1)))) 



def rmsle(estimator, X, y0):

    y = estimator.predict(X)

    if len(y[y<=-1]) != 0:

        y[y<=-1] = 0.0

    assert len(y) == len(y0)

    r = np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))

    if math.isnan(r):

        print("this is a nan")

        print(scipy.stats.describe(y))

        plt.hist(y, bins=10, color='blue')

        plt.show()

    return r



models = [

    LinearRegression(),

    Ridge(),

    SGDRegressor(),

    BayesianRidge(),

    Lasso(),

    LassoLars(),

    ElasticNet(),

    TheilSenRegressor()

]



for pred_type in X_train:

    print (" * Predicting {}...".format(pred_type))

    for model in models:

        print ("   - {}".format(model))

        print ("      . Dataset 1: ", end='')

        scores = cross_val_score(model, X_train[pred_type][0], y_train[pred_type], cv=5, scoring=rmsle)

        print (scores.mean())

        print ("      . Dataset 2: ", end='')

        scores = cross_val_score(model, X_train[pred_type][1], y_train[pred_type], cv=5, scoring=rmsle)

        print (scores.mean())

        print ("      . Dataset 3: ", end='')

        scores = cross_val_score(model, X_train[pred_type][2], y_train[pred_type], cv=5, scoring=rmsle)

        print (scores.mean())
from xgboost import XGBRegressor

from lightgbm import LGBMRegressor



models = [

    XGBRegressor(),

    LGBMRegressor()

]



for pred_type in X_train:

    print (" * Predicting {}...".format(pred_type))

    for model in models:

        print ("   - {}".format(model))

        print ("      . Dataset 1: ", end='')

        scores = cross_val_score(model, X_train[pred_type][0], y_train[pred_type], cv=5, scoring=rmsle)

        print (scores.mean())

        print ("      . Dataset 2: ", end='')

        scores = cross_val_score(model, X_train[pred_type][1], y_train[pred_type], cv=5, scoring=rmsle)

        print (scores.mean())

        print ("      . Dataset 3: ", end='')

        scores = cross_val_score(model, X_train[pred_type][2], y_train[pred_type], cv=5, scoring=rmsle)

        print (scores.mean())
import time

from tqdm import tqdm

from sklearn.model_selection import KFold

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone



########################################################################################################################################

class CustomEnsemble (BaseEstimator, RegressorMixin, TransformerMixin):

    

    def __init__(self, models, meta_model):

        self.models = models

        self.modelsNames = [a.__str__().split("(")[0] for a in self.models]

        self.meta_model = meta_model

        

    def fit(self,X,y):

        predictions = np.zeros((X.shape[0], len(self.models)))

        for i, model in enumerate(self.models):

            model.fit (X, y)

            predictions[:,i] = model.predict(X)

        self.meta_model.fit(predictions, y)

            

    def predict(self,X):

        predictions = np.zeros((X.shape[0], len(self.models)))

        for i, model in enumerate(self.models):

            predictions[:,i] = model.predict(X)

        return self.meta_model.predict(predictions)

    

    def __str__ (self):

        return "<CustomEnsemble (meta={}, models={})>".format(self.meta_model.__str__().split("(")[0], self.modelsNames)

    

    def __repr__ (self):

        return self.__str__()

########################################################################################################################################



# Defining function for making models combinations

def make_combinations (iterable):

    from itertools import combinations

    my_combs = []

    for item in iterable.copy():

        iterable.remove(item)

        for i in range(len(iterable)):

            for comb in combinations(iterable, i+1):

                my_combs.append((item, comb))

        iterable.append(item)

    return my_combs



cases_models = [

    #TheilSenRegressor(),

    XGBRegressor(),

    LGBMRegressor()

]



fatal_models = [

    LinearRegression(),

    Ridge(),

    BayesianRidge(),

    Lasso(),

    #TheilSenRegressor(),

    XGBRegressor(),

    LGBMRegressor()

]



cases_combs = make_combinations(cases_models)

fatal_combs = make_combinations(fatal_models)



models = {}

models['cases'] = []

models['fatal'] = []

for comb in cases_combs:

    models['cases'].append(CustomEnsemble(meta_model=comb[0], models=comb[1]))

for comb in fatal_combs:

    models['fatal'].append(CustomEnsemble(meta_model=comb[0], models=comb[1]))



best_score = {}

best_score['cases'] = 10e3

best_score['fatal'] = 10e3

best_model = {}

best_model['cases'] = None

best_model['fatal'] = None

best_dataset = {}

best_dataset['cases'] = None

best_dataset['fatal'] = None



if False:

    print (" --> I'll test {} models! :D".format(len(models['cases']) + len(models['fatal'])))

    with tqdm(total = len(models['cases']) + len(models['fatal'])) as pbar:

        for pred_type in X_train:

            #print (" * Predicting {}...".format(pred_type))

            for model in models[pred_type]:

                ##

                score = cross_val_score(model, X_train[pred_type][0], y_train[pred_type], cv=5, scoring=rmsle).mean()

                if (score < best_score[pred_type]):

                    best_score[pred_type] = score

                    best_model[pred_type] = model

                    best_dataset[pred_type] = X_train[pred_type][0]

                ##

                score = cross_val_score(model, X_train[pred_type][1], y_train[pred_type], cv=5, scoring=rmsle).mean()

                if (score < best_score[pred_type]):

                    best_score[pred_type] = score

                    best_model[pred_type] = model

                    best_dataset[pred_type] = X_train[pred_type][1]

                ##

                score = cross_val_score(model, X_train[pred_type][2], y_train[pred_type], cv=5, scoring=rmsle).mean()

                if (score < best_score[pred_type]):

                    best_score[pred_type] = score

                    best_model[pred_type] = model

                    best_dataset[pred_type] = X_train[pred_type][2]

                ##

                pbar.update(1)

else:

    best_model['cases'] = CustomEnsemble(meta_model=LGBMRegressor(), models=[XGBRegressor()])

    best_score['cases'] = 0.23218969104329368

    best_dataset['cases'] = X_train['cases'][2]

    best_model['fatal'] = CustomEnsemble(meta_model=XGBRegressor(), models=[LGBMRegressor(), Lasso()])

    best_score['fatal'] = 0.1117526321099606

    best_dataset['fatal'] = X_train['fatal'][2]

    

print ("Cases:\n=> Best model: {}\n=> Best score: {:.4f}\n\nFatalities:\n=> Best model: {}\n=> Best score: {:.4f}".format(best_model['cases'], best_score['cases'], best_model['fatal'], best_score['fatal']))
if FULL_RUN:

    models['cases'] = [

        CustomEnsemble(meta_model=TheilSenRegressor(), models=[XGBRegressor(), LGBMRegressor(), Lasso()]),

        CustomEnsemble(meta_model=LGBMRegressor(), models=[XGBRegressor(), TheilSenRegressor()]),

        CustomEnsemble(meta_model=XGBRegressor(), models=[LGBMRegressor(), TheilSenRegressor()]),

        LGBMRegressor(),

    ]



    models['fatal'] = [

        CustomEnsemble(meta_model=TheilSenRegressor(), models=[LGBMRegressor(), Lasso(), XGBRegressor()]),

        CustomEnsemble(meta_model=LGBMRegressor(), models=[TheilSenRegressor(), XGBRegressor(), Lasso()]),

        CustomEnsemble(meta_model=XGBRegressor(), models=[TheilSenRegressor(), LGBMRegressor(), Lasso()]),

        LGBMRegressor(),

    ]



    for pred_type in X_train:

        print (" * Predicting {}...".format(pred_type))

        for model in models[pred_type]:

            score = cross_val_score(model, X_train[pred_type][2], y_train[pred_type], cv=5, scoring=rmsle).mean()

            print ("-> Score: {}".format(score))

            if (score < best_score[pred_type]):

                best_score[pred_type] = score

                best_model[pred_type] = model

                best_dataset[pred_type] = X_train[pred_type][2]

                print ("Score got better for model {}".format(model))

else:

    best_model['cases'] = CustomEnsemble(meta_model=TheilSenRegressor(), models=[XGBRegressor(), LGBMRegressor(), Lasso()])

    best_score['cases'] = 0.22054589547642367
def plotStatus (location):

    plt.figure(figsize=(14,8))

    plt.title('COVID-19 cases on {}'.format(location))

    df = df_train[df_train[location] == 1]

    test = df_test[df_test[location] == 1]

    intersection = df_intersection[df_intersection[location] == 1]

    idx = df_test[df_test[location] == 1].index

    legend = []

    plt.xlabel('#Days since dataset')

    plt.ylabel('Number')

    plt.plot(df['Date'], df['ConfirmedCases'])

    plt.plot(test['Date'], test['ConfirmedCases'])

    plt.plot(intersection['Date'], intersection['ConfirmedCases'])

    legend.append('{} confirmed cases'.format(location))

    legend.append('{} predicted cases'.format(location))

    legend.append('{} actual cases'.format(location))

    plt.legend(legend)

    plt.show()

    legend = []

    plt.figure(figsize=(14,8))

    plt.title('COVID-19 fatalities on {}'.format(location))

    plt.xlabel('#Days since dataset')

    plt.ylabel('Number')

    plt.plot(df['Date'], df['Fatalities'])

    plt.plot(test['Date'], test['Fatalities'])

    plt.plot(intersection['Date'], intersection['Fatalities'])

    legend.append('{} fatalities'.format(location))

    legend.append('{} predicted fatalities'.format(location))

    legend.append('{} actual fatalities'.format(location))

    plt.show()



def rmsle (location):

    idx = df_test[(df_test[location] == 1) & (df_test['Date'] <= df_intersection['Date'].max())].index

    my_sub = df_test.loc[idx][['ConfirmedCases', 'Fatalities']]

    cases_pred = my_sub['ConfirmedCases'].values

    fatal_pred = my_sub['Fatalities'].values

    idx = df_intersection[df_intersection[location] == 1].index

    cases_targ = df_intersection.loc[idx]['ConfirmedCases'].values

    fatal_targ = df_intersection.loc[idx]['Fatalities'].values

    cases = np.sqrt(mean_squared_log_error( cases_targ, cases_pred ))

    fatal = np.sqrt(mean_squared_log_error( fatal_targ, fatal_pred ))

    return cases, fatal



def avg_rmsle():

    idx = df_intersection.index

    my_sub = df_test.loc[idx][['ConfirmedCases', 'Fatalities']]

    cases_pred = my_sub['ConfirmedCases'].values

    fatal_pred = my_sub['Fatalities'].values

    cases_targ = df_intersection.loc[idx]['ConfirmedCases'].values

    fatal_targ = df_intersection.loc[idx]['Fatalities'].values

    cases_score = np.sqrt(mean_squared_log_error( cases_targ, cases_pred ))

    fatal_score = np.sqrt(mean_squared_log_error( fatal_targ, fatal_pred ))

    score = (cases_score + fatal_score)/2

    return score



def handle_predictions (predictions, lowest = 0):

    #predictions = np.round(predictions, 0)

    # Predictions can't be negative

    predictions[predictions < 0] = 0

    # Predictions can't decrease from greatest value on train dataset

    predictions[predictions < lowest] = lowest

    # Predictions can't decrease over time

    for i in range(1, len(predictions)):

        if predictions[i] < predictions[i - 1]:

            predictions[i] = predictions[i - 1]

    #return predictions.astype(int)

    return predictions
cols = []

lag_range = np.arange(1,15,1)

for lag in lag_range:

    cols.append("ConfirmedCases_Lag_{}".format(lag))

    cols.append("Fatalities_Lag_{}".format(lag))

test_intersection_mask = (df_test['Date'] <= df_intersection['Date'].max())

train_intersection_mask = (df_intersection['Date'] >= df_test['Date'].min())

df_test.loc[test_intersection_mask, cols] = df_intersection.loc[train_intersection_mask, cols].values
model_cases = best_model['cases']

model_fatal = best_model['fatal']



input_cols = list(set(my_cases_cols + my_fatal_cols))



model_cases.fit(X_train['cases'][2], y_train['cases'])

model_fatal.fit(X_train['fatal'][2], y_train['fatal'])



use_predictions = False

pred_dt_range = range(int(df_test['Date'].min()), int(df_test['Date'].max()) + 1)

locations = [col for col in df_train.columns if col.startswith('Location')]

random_validation_set = ['Location_Brazil_Brazil', 'Location_US_New York', 'Location_Afghanistan_Afghanistan', 'Location_China_Zhejiang', 'Location_Italy_Italy']#random.sample(states, 10)

pred_input = locations



start_time = time.time()

with tqdm(total = len(list(pred_input))) as pbar:

    for location in pred_input:

        for d in pred_dt_range:

            mask = (df_test['Date'] == d) & (df_test[location] == 1)

            if (d > df_intersection['Date'].max()):

                for lag in lag_range:

                    mask_org = (df_test['Date'] == (d - lag)) & (df_test[location] == 1)

                    try:

                        df_test.loc[mask, 'ConfirmedCases_Lag_' + str(lag)] = df_test.loc[mask_org, 'ConfirmedCases'].values

                    except:

                        df_test.loc[mask, 'ConfirmedCases_Lag_' + str(lag)] = 0

                    try:

                        df_test.loc[mask, 'Fatalities_Lag_' + str(lag)] = df_test.loc[mask_org, 'Fatalities'].values

                    except:

                        df_test.loc[mask, 'Fatalities_Lag_' + str(lag)] = 0

            X_test  = df_test.loc[mask, input_cols]

            # Cases

            X_test_cases = X_test[my_cases_cols].values

            X_test_cases = scaler['cases'][2].transform(X_test_cases)

            next_cases = model_cases.predict(X_test_cases)

            # Fatal

            X_test_fatal = X_test[my_fatal_cols].values

            X_test_fatal = scaler['fatal'][2].transform(X_test_fatal)

            next_fatal = model_fatal.predict(X_test_fatal)

            # Update df_test

            if (d > np.max(df_train['Date'].values)):

                if (next_cases < 0):

                    next_cases = 0

                if (next_cases < X_test['ConfirmedCases_Lag_1'].values[0]):

                    next_cases = X_test['ConfirmedCases_Lag_1'].values[0]

                df_test.loc[mask, 'ConfirmedCases'] = next_cases

                if (next_fatal < 0):

                    next_fatal = 0

                if (next_fatal < X_test['Fatalities_Lag_1'].values[0]):

                    next_fatal = X_test['Fatalities_Lag_1'].values[0]

                df_test.loc[mask, 'Fatalities'] = next_fatal

            else:

                if use_predictions:

                    if (next_cases < 0):

                        next_cases = 0

                    if (next_cases < X_test['ConfirmedCases_Lag_1'].values[0]):

                        next_cases = X_test['ConfirmedCases_Lag_1'].values[0]

                    df_test.loc[mask, 'ConfirmedCases'] = next_cases

                    if (next_fatal < 0):

                        next_fatal = 0

                    if (next_fatal < X_test['Fatalities_Lag_1'].values[0]):

                        next_fatal = X_test['Fatalities_Lag_1'].values[0]

                    df_test.loc[mask, 'Fatalities'] = next_fatal

        # Fill cases

        lowest_pred = np.max(df_train[df_train[location] == 1]['ConfirmedCases'].values)

        cases = handle_predictions (df_test[df_test[location] == 1]['ConfirmedCases'].values, lowest_pred)

        # Fill fatal

        lowest_pred = np.max(df_train[df_train[location] == 1]['Fatalities'].values)

        cases = handle_predictions (df_test[df_test[location] == 1]['Fatalities'].values, lowest_pred)

        # Update progress bar

        pbar.update(1)

        

print('Time spent for predicting everything was {} minutes'.format(round((time.time()-start_time)/60,1)))

#avg_rmsle()
cases = []

fatal = []

for a in random_validation_set:

    score = rmsle(a)

    cases.append(score[0])

    fatal.append(score[1])

    print(score)

print ("Average = {}, {}".format(np.average(cases), np.average(fatal)))
for a in random_validation_set:

    plotStatus(a)
output_cols = ['ConfirmedCases', 'Fatalities']

submission = df_test[['ForecastId'] + output_cols]

submission
submission.to_csv("submission.csv", index=False)