!pip install dateparser
import os

import pandas as pd

import numpy as np



#print(os.listdir('../input/novel-corona-virus-2019-dataset'))



folder = '../input/novel-corona-virus-2019-dataset'



all_df = pd.read_csv(os.path.join(folder, 'covid_19_data.csv'))



all_df.head()





countries = all_df['Country/Region'].unique()



print('There is data for {0} countries'.format(len(countries)))

import math

import dateparser



from datetime import datetime



def logFinder(data):



    out = []

    for d in data:

    

        try:

            out.append(math.log(d))

        except:

            out.append(0)

    return out





def differentiate(data):



    diff = []



    for i, d in enumerate(data):



        if i > 0:



            diff.append(d - data[i-1])

        else:

            diff.append(0)

            

    return diff



def find_growth_factor(data):

    

    GF = []

    for i, d in enumerate(data):

        

        if i > 1:

            

            if data[i-1] == 0:

                

                gf = 0

                

            else:

            

                gf = d/data[i-1]

            

        else:

            gf = 0

            

        GF.append(gf)

    

    return GF





def moving_average(l, n=3):



    out = []

    

    if n == 0:

        return l

    

    for i, el in enumerate(l):

        

        #if there are no collisions with the edges of the list

        if i >= n and len(l)-i >= n:



            r = sum(l[i-n:i+n])/len(l[i-n:i+n])

        

        #if the window colides with the start of the list

        elif i < n and len(l)-i >= n:



            r = sum(l[0:i+n])/len(l[0:i+n])

            

        #if the window collides with the end of the list

        elif i >= n and len(l)-i < n:

            

            r = sum(l[i-n:])/len(l[i-n:])

        #window collides with start and end of list

        else:

            r = sum(l)/len(l)

        

        out.append(r)

    

    return out



###################################

### End of function definitions ###

###################################



class countryData(object):

    

    def __init__(self, all_df, country_name, banned_provences):

        

        self.country = country_name

        self.banned_provinces = banned_provences

        

        self.extract_country_data(all_df)

        

        #clean the dataset

        self.clean_dataset()

        

        #calculate the number of new cases per day

        self.diff_dataset()

        

        #find the daily rate of growth

        self.growth_calc()

        

        #find the natural log for each day

        self.natLogDataset()

        

        

    def natLogDataset(self):

        

        self.lnInfected = logFinder(self.infected)

        self.lnDead = logFinder(self.dead)

        self.lnRecovered = logFinder(self.recovered)

        

    #clean the zeros from a list

    def clean_raw_data(self, data):

        

        out = []

        maxVal = 0

        for di, d in enumerate(data):

            

            if d > maxVal:

                maxVal = d

                

            if d == 0:

                out.append(maxVal)

                

            else:

                out.append(d)

                

        return out

    

    def clean_dataset(self):

        

        self.infected = self.clean_raw_data(self.infected)

        self.dead = self.clean_raw_data(self.dead)

        self.recovered = self.clean_raw_data(self.recovered)

    

    #differentiate every datastream in the dataset

    def diff_dataset(self):

        

        self.diff_infected = differentiate(self.infected)

        self.diff_dead = differentiate(self.dead)

        self.diff_recovered = differentiate(self.recovered)

        

    #calculate the growth factor for each datastream

    def growth_calc(self):

        

        self.growth_infected = find_growth_factor(self.diff_infected)

        self.growth_dead = find_growth_factor(self.diff_dead)

        self.growth_recovered = find_growth_factor(self.diff_recovered)

        

        

    #extract the data for a given country

    def extract_country_data(self, all_df):

        

        country_df = all_df[all_df['Country/Region'] == self.country]

        

        self.dates = sorted(list(country_df['ObservationDate'].unique()))

        self.infected = []

        self.dead = []

        self.recovered = []

        

        

        for datei, date in enumerate(self.dates):

            

            dateDf = country_df[country_df['ObservationDate'] == date]

            

            infections_sum = 0

            deads_sum = 0

            recovered_sum = 0

            

            for rowi, row in dateDf.iterrows():

                

                if row['Province/State'] not in self.banned_provinces:

                    infections_sum += row['Confirmed']

                    deads_sum += row['Deaths']

                    recovered_sum += row['Recovered']

                    

            self.infected.append(infections_sum)

            self.dead.append(deads_sum)

            self.recovered.append(recovered_sum)

            

        #filter out unwanted provinces

        #self.country_df = country_df[country_df['Province/State'] not in self.banned_provinces]



        #turn the dates into datetime objects

        self.dates[:] = [dateparser.parse(d) for d in self.dates]



################################

### End of class definitions ###

################################



print('Instanciated tools')
import numpy as np



%matplotlib inline

import matplotlib.pyplot as plt



#show where lockdown began on the plot

uk_lockdown_date = datetime(2020, 3, 23)



#the uk dataset contains data for a series of locations within the UK not connected to the UK mainland. 

#We are going to exclude those from our dataset and only examine the data for the uk mainland.

banned_uk_places = ['Channel Islands', 'Gibraltar','Cayman Islands',

                 'Montserrat', 'Bermuda', 'Isle of Man', 'Anguilla', 'British Virgin Islands',

                 'Turks and Caicos Islands', 'Falkland Islands (Islas Malvinas)',

                 'Falkland Islands (Malvinas)']



country_name = 'UK'



uk_country_data = countryData(all_df, country_name, banned_uk_places)



xlim = 2

ylim = 2



fig, ax = plt.subplots(ylim, xlim, figsize=(14, 10))



plt.subplots_adjust(hspace=0.2)



#plot the raw data

ax[0, 0].plot(uk_country_data.dates, uk_country_data.infected, label='Infected')

ax[0, 0].plot(uk_country_data.dates, uk_country_data.dead, label='Dead')

ax[0, 0].plot([uk_lockdown_date, uk_lockdown_date], [0, max(uk_country_data.infected)], 'k--', label='Lockdown')

ax[0, 0].set_xlabel('Dates')

ax[0, 0].set_ylabel('Count')

ax[0, 0].legend()

ax[0, 0].set_title('Raw data')



#plot the new cases per day

ax[0, 1].plot(uk_country_data.dates, uk_country_data.diff_infected, label='Infected')

ax[0, 1].plot(uk_country_data.dates, uk_country_data.diff_dead, label='Dead')

ax[0, 1].plot([uk_lockdown_date, uk_lockdown_date], [0, max(uk_country_data.diff_infected)], 'k--', label='Lockdown')

ax[0, 1].set_xlabel('Dates')

ax[0, 1].set_ylabel('Count per day')

ax[0, 1].legend()

ax[0, 1].set_title('New cases per day')





#plot the growth rate

ax[1, 0].plot(uk_country_data.dates, uk_country_data.growth_infected, label='Infected')

ax[1, 0].plot(uk_country_data.dates, uk_country_data.growth_dead, label='Dead')

ax[1, 0].plot([uk_lockdown_date, uk_lockdown_date], [0, max(uk_country_data.growth_infected)], 'k--', label='Lockdown')

ax[1, 0].set_xlabel('Dates')

ax[1, 0].set_ylabel('Growth rate')

ax[1, 0].legend()

ax[1, 0].set_title('Growth rate over time')





#plot the natural log of the data

ax[1, 1].plot(uk_country_data.dates, uk_country_data.lnInfected, label='Infected')

ax[1, 1].plot(uk_country_data.dates, uk_country_data.lnDead, label='Dead')

ax[1, 1].plot([uk_lockdown_date, uk_lockdown_date], [0, max(uk_country_data.lnInfected)], 'k--', label='Lockdown')

ax[1, 1].set_xlabel('Dates')

ax[1, 1].set_ylabel('Ln(count)')

ax[1, 1].legend()

ax[1, 1].set_title('Natural log of cases over time')





fig.suptitle(country_name)


plot_countries = ['UK',

                  'France',

                  'Germany',

                  'Italy',

                  'Spain',

                  'Belgium',

                  'Austria',

                  'Portugal']



countries_banned_provinces = [banned_uk_places,

                             [],

                             [],

                             [],

                             [],

                             [],

                             [],

                             []]





print(len(plot_countries), len(countries_banned_provinces))



plt.figure(figsize=(18, 8))

ax1 = plt.subplot(1, 2, 1)

ax2 = plt.subplot(1, 2, 2)



countriesData = {}

for ci, country in enumerate(plot_countries):

    

    this_country_data = countryData(all_df, country, countries_banned_provinces[ci])

    countriesData[country] = {'allData':this_country_data}

    

    ax1.plot(this_country_data.dates, this_country_data.infected, label=country)

    

    

for ci, country in enumerate(plot_countries):

    

    this_country_data = countriesData[country]['allData']

    

    ax2.plot(this_country_data.dates, this_country_data.dead, label=country)

    

ax1.set_xlabel('Dates')

ax1.set_ylabel('Count')

ax1.set_title('Deaths per country')

ax1.legend()



    

ax2.set_xlabel('Dates')

ax2.set_ylabel('Count')

ax2.set_title('Infections per country')

ax2.legend()

plt.show()
plt.figure(figsize=(18, 8))

ax1 = plt.subplot(1, 2, 1)

ax2 = plt.subplot(1, 2, 2)

for ci, country in enumerate(plot_countries):

    

    this_country_data = countriesData[country]['allData']

    

    store = False

    plot_days = []

    plot_deaths = []

    plot_infections = []

    for dayi, day in enumerate(this_country_data.dates):



        if this_country_data.dead[dayi] >= 10:

    

            day_of_death_10 = day

            store = True

            

        if store:

            if len(plot_days) > 0:

                plot_days.append(plot_days[-1] + 1)

            else:

                plot_days.append(0)

                

            plot_deaths.append(this_country_data.dead[dayi])

            plot_infections.append(this_country_data.infected[dayi])

        



    ax1.plot(plot_days, plot_deaths, label=country)

    ax2.plot(plot_days, plot_infections, label=country)

    

ax1.set_xlabel('Days since death 10')

ax1.set_ylabel('Count')

ax1.set_title('Deaths per country since day of death 10')

ax1.legend()



ax2.set_xlabel('Days since death 10')

ax2.set_ylabel('Count')

ax2.set_title('Infections per country since day of death 10')

ax2.legend()



plt.show()
plt.figure(figsize=(18, 8))

ax1 = plt.subplot(1, 2, 1)

ax2 = plt.subplot(1, 2, 2)

for ci, country in enumerate(plot_countries):

    

    this_country_data = countriesData[country]['allData']

    

    store = False

    plot_days = []

    Lnplot_deaths = []

    Lnplot_infections = []

    for dayi, day in enumerate(this_country_data.dates):

        

        

        if this_country_data.dead[dayi] >= 10:

    

            day_of_death_100 = day

            store = True

            

        if store:

            if len(plot_days) > 0:

                plot_days.append(plot_days[-1] + 1)

            else:

                plot_days.append(0)

                

            Lnplot_deaths.append(this_country_data.lnDead[dayi])

            Lnplot_infections.append(this_country_data.lnInfected[dayi])

        



    ax1.plot(plot_days, Lnplot_deaths, label=country)

    ax2.plot(plot_days, Lnplot_infections, label=country)

    

ax1.set_xlabel('Days since death 10')

ax1.set_ylabel('Ln(Count)')

ax1.set_title('Natural log of deaths per country since day of death 10')

ax1.legend()



ax2.set_xlabel('Days since death 10')

ax2.set_ylabel('Ln(Count)')

ax2.set_title('Natural log of infections per country since day of death 10')

ax2.legend()



plt.show()
import numpy as np



def fit_line(X, Y):

    

    lin = np.polyfit(X, Y, deg=1, full=True)



    m = lin[0][0]

    c = lin[0][1]



    fit_Y = []

    for x in X:

        fit_Y.append((m * x) + c)

        

    return [fit_Y, m, c]



###################################

### End of function definitions ###

###################################



colours = ['k', 'r', 'b', 'g', 'y', 'pink', 'c', 'orange']



plt.figure(figsize=(18, 8))

ax1 = plt.subplot(1, 2, 1)

ax2 = plt.subplot(1, 2, 2)

for ci, country in enumerate(plot_countries):

    

    this_country_data = countriesData[country]['allData']

    

    store = False

    plot_days = []

    Lnplot_deaths = []

    Lnplot_infected = []

    for dayi, day in enumerate(this_country_data.dates):

        

        

        if this_country_data.dead[dayi] >= 10:

    

            day_of_death_10 = day

            store = True

            

        if store:

            if len(plot_days) > 0:

                plot_days.append(plot_days[-1] + 1)

            else:

                plot_days.append(0)

                

            Lnplot_deaths.append(this_country_data.lnDead[dayi])

            Lnplot_infected.append(this_country_data.lnInfected[dayi])

        

    [LnDeaths_fit, mD, cD] = fit_line(plot_days, Lnplot_deaths)

    [LnInfections_fit, mI, cI] = fit_line(plot_days, Lnplot_infected)

    

    countriesData[country].update({'day_of_death_10':day_of_death_10,

                             'plot_days':plot_days,

                             'LnDead':Lnplot_deaths,

                             'LnInfected':Lnplot_infected})

    

    ax1.plot(plot_days, LnDeaths_fit, linestyle = ':', color=colours[ci%len(colours)])

    ax1.plot(plot_days, Lnplot_deaths, label=country, color=colours[ci%len(colours)])

    

    ax2.plot(plot_days, LnInfections_fit, linestyle = ':', color=colours[ci%len(colours)])

    ax2.plot(plot_days, Lnplot_infected, label=country, color=colours[ci%len(colours)])

    

ax1.set_xlabel('Days since death 10')

ax1.set_ylabel('Ln(Count)')

ax1.set_title('Natural log of deaths per country since day of death 10')

ax1.legend()



ax2.set_xlabel('Days since death 10')

ax2.set_ylabel('Ln(Count)')

ax2.set_title('Natural log of infections per country since day of death 10')

ax2.legend()
country = 'Italy'



this_country_data = countriesData[country]['allData']





store = False

plot_days = []

Lnplot_deaths = []

deaths = []

#cycle through the days in order to extract just the data for the day of death 10 onwards

for dayi, day in enumerate(this_country_data.dates):



    if this_country_data.dead[dayi] >= 10:



        day_of_death_100 = day

        store = True



    if store:

        if len(plot_days) > 0:

            plot_days.append(plot_days[-1] + 1)

        else:

            plot_days.append(0)



        Lnplot_deaths.append(this_country_data.lnDead[dayi])

        deaths.append(this_country_data.dead[dayi])



    

#calculate the fits for the first 20 days/all days

[LnDeaths_fit_all, m, c] = fit_line(plot_days, Lnplot_deaths)

[LnDeaths_fit_exponential_part, m, c] = fit_line(plot_days[:20], Lnplot_deaths[:20])





#create the plot comparing the two fits

plt.figure(figsize=(16, 8))



#plot data for first 20 days

ax1 = plt.subplot(1, 2, 1)

ax1.plot(plot_days[:20], Lnplot_deaths[:20], color='r', label=country)

ax1.plot(plot_days[:20], LnDeaths_fit_exponential_part, color='r', linestyle=':', label='fit')

ax1.set_xlabel('Days since death 10')

ax1.set_ylabel('Ln(deaths)')

ax1.legend()

ax1.set_title('Deaths during exponential growth stage of epidemic')



#plot data for all days

ax2 = plt.subplot(1, 2, 2)

ax2.plot(plot_days, Lnplot_deaths, color='r', label=country)

ax2.plot(plot_days, LnDeaths_fit_all, color='r', linestyle=':', label='fit')

ax2.set_xlabel('Days since death 10')

ax2.set_ylabel('Ln(deaths)')

ax2.legend()

ax2.set_title('Deaths to date')



plt.show()

def calculateRMS(A, B):

    

    if len(A) != len(B):

        

        raise Exception('lists must be the same length')

    

    square_sum = 0

    for ai, a in enumerate(A):

        

        square_sum += (A[ai] - B[ai])**2

        

    return (square_sum ** 0.5) / len(A)

        

    

###################################

### End of function definitions ###

###################################





#cycle through the countries in our dataset

RMS_countries = {}

for ci, country in enumerate(countriesData.keys()):

    

    #now for this country calculate how far each day's data deviates from a straight line fit using the RMSE

    this_country_data = countriesData[country]

    

    #set up the framewords we're going to be storing data in

    RMSesDead = []

    RMSesInfected = []

    MsDead = []

    MsInfected = []

    lnDead = []

    lnInfected = []

    days = []

    for di, dead in enumerate(this_country_data['allData'].dead):

        

        #only want data from post death no. 10

        if dead >= 10:



            lnDead.append(countriesData[country]['allData'].lnDead[di])

            lnInfected.append(countriesData[country]['allData'].lnInfected[di])

        

            if len(lnDead) > 2:

                [LnDeaths_fit, mD, cD] = fit_line(list(range(len(lnDead))), lnDead)

                [LnInfecteds_fit, mI, cI] = fit_line(list(range(len(lnInfected))), lnInfected)

                

                #print(lnDead)

                #print(LnDeaths_fit)

                

                #calculate the errors for the dead and infected fits

                sqMeanDead = calculateRMS(lnDead, LnDeaths_fit)

                sqMeanInfected = calculateRMS(lnInfected, LnInfecteds_fit)

                

                #store the calculated values in lists we can plot at the end of the cell

                RMSesDead.append(sqMeanDead)

                RMSesInfected.append(sqMeanInfected)

                

                MsDead.append(mD)

                MsInfected.append(mI)

                

            else:                

                

                #if there's not enough data for a fit yet just set the values to zero

                RMSesDead.append(0)

                RMSesInfected.append(0)

                

                MsDead.append(0)

                MsInfected.append(0)

                

    #store these results

    countriesData[country]['RMSE_infected'] = RMSesInfected

    countriesData[country]['RMSE_dead'] = RMSesDead

    countriesData[country]['MsInfected'] = MsInfected

    countriesData[country]['MsDead'] = MsDead



    

#plot these results

plt.figure(figsize=(18, 8))

ax1 = plt.subplot(1, 2, 1)

ax2 = plt.subplot(1, 2, 2)

for ci, country in enumerate(countriesData.keys()):

    

    ax1.plot(countriesData[country]['plot_days'], countriesData[country]['RMSE_dead'], color = colours[ci%len(colours)], label=country)

    ax2.plot(countriesData[country]['plot_days'], countriesData[country]['RMSE_infected'], color = colours[ci%len(colours)], label=country)

    

ax1.set_xlabel('Days since death 10')

ax1.set_ylabel('RMSE')

ax1.set_title('RMSE of straight line fit to natural log of death count since day of death 10')

ax1.legend()



ax2.set_xlabel('Days since death 10')

ax2.set_ylabel('RMSE')

ax2.set_title('RMSE of straight line fit to natural log of infections count since day of death 10')

ax2.legend()

plt.figure(figsize=(18, 8))

ax1 = plt.subplot(1, 2, 1)

ax2 = plt.subplot(1, 2, 2)



for ci, country in enumerate(countriesData.keys()):

    

    #now for this country calculate how far each day's data deviates from a straight line fit using the RMSE

    this_country_data = countriesData[country]

    

    #set up the framewords we're going to be storing data in

    newDeaths = []

    newInfections = []

    for di, dead in enumerate(this_country_data['allData'].dead):

        

        #only want data from post death no. 10

        if dead >= 10:

            newDeaths = this_country_data['allData'].diff_dead[di:]

            newInfections = this_country_data['allData'].diff_infected[di:]

            

            break

            

    #smooth out the daily totals in order to account for issues with reporting

    newDeaths = moving_average(newDeaths, n=7)

    newInfections = moving_average(newInfections, n=7)

    

    #moving_average(l, n=3)

        

    #plot the new events per day since day 10

    ax1.plot(countriesData[country]['plot_days'], newDeaths, color = colours[ci%len(colours)], label=country)

    ax2.plot(countriesData[country]['plot_days'], newInfections, color = colours[ci%len(colours)], label=country)

            

    

ax1.set_xlabel('Days since death 10')

ax1.set_ylabel('Count')

ax1.set_title('Deaths per day (since day of death 10)')

ax1.legend()



ax2.set_xlabel('Days since death 10')

ax2.set_ylabel('Count')

ax2.set_title('Newly confirmed infections per day (since day of death 10)')

ax2.legend()