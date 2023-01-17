import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
path = "../input/johns-hopkins-csse-covid19-time-series-data/"



### CONSTRUCT TIME SERIES DATE FOR CONFIRMED CASES BY COUNTRY ###



timeseries_confirmed_df = pd.read_csv(path+"time_series_covid19_confirmed_global.csv")

#timeseries_confirmed_df.head(10)



# WRITE NEW FILE AGGREGATING CONFIRMED CASE COUNT BY COUNTRY INSTEAD OF 'COUNTRY/REGION'"

fout = open("time_series_covid19_confirmed_COUNTRY_ONLY.csv",'w')

fout.write("Country")

dates = timeseries_confirmed_df.columns[4:]



for date in dates:

    fout.write(","+date)

fout.write("\n")



countries = list(timeseries_confirmed_df["Country/Region"].value_counts().index)

#print(countries)



for country in countries:

    fout.write(country)

    for date in dates:

        count = np.sum(np.array(timeseries_confirmed_df.set_index("Country/Region").loc[country, date]))

        fout.write(","+str(count))

    fout.write("\n")

fout.close()





### CONSTRUCT TIME SERIES DATEFRAME FOR DEATHS BY COUNTRY ###



timeseries_deaths_df = pd.read_csv(path+"time_series_covid19_deaths_global.csv")

#timeseries_deaths_df.head()



# WRITE NEW FILE AGGREGATING DEATH TOLL BY COUNTRY INSTEAD OF COUNTRY/REGION"

fout = open("time_series_covid19_deaths_COUNTRY_ONLY.csv",'w')

fout.write("Country")

dates = timeseries_deaths_df.columns[4:]

for date in dates:

    fout.write(","+date)

fout.write("\n")



countries = list(timeseries_deaths_df["Country/Region"].value_counts().index)

#print(countries)



for country in countries:

    fout.write(country)

    for date in dates:

        count = np.sum(np.array(timeseries_deaths_df.set_index("Country/Region").loc[country, date]))

        fout.write(","+str(count))

    fout.write("\n")

fout.close()

### CONSTRUCT TIME SERIES DATA FOR CONFIRMED CASES BY US STATE ###



timeseries_confirmed_df = pd.read_csv(path+"time_series_covid19_confirmed_US.csv")



fout = open("time_series_covid19_confirmed_US_STATE.csv",'w')

fout.write("Province_State")

dates = timeseries_confirmed_df.columns[11:]

#print(dates)

for date in dates:

    fout.write(","+date)

fout.write("\n")



states = list(timeseries_confirmed_df["Province_State"].value_counts().index)

#print("states ", states)

#print("len(states) ", len(states))



for state in states:

    fout.write(state)

    for date in dates:

        count = np.sum(np.array(timeseries_confirmed_df.set_index("Province_State").loc[state, date]))

        fout.write(","+str(count))

    fout.write("\n")

fout.close()

### CONSTRUCT TIME SERIES DATA FOR DEATHS BY US STATE ###



timeseries_deaths_df = pd.read_csv(path+"time_series_covid19_deaths_US.csv")



fout = open("time_series_covid19_deaths_US_STATE.csv",'w')

fout.write("Province_State")

dates = timeseries_deaths_df.columns[12:]



for date in dates:

    fout.write(","+date)

fout.write("\n")



states = list(timeseries_deaths_df["Province_State"].value_counts().index)



for state in states:

    fout.write(state)

    for date in dates:

        count = np.sum(np.array(timeseries_deaths_df.set_index("Province_State").loc[state, date]))

        fout.write(","+str(count))

    fout.write("\n")

fout.close()
timeseries_country_cases_df = pd.read_csv("time_series_covid19_confirmed_COUNTRY_ONLY.csv")

timeseries_country_cases_df.head()
timeseries_state_cases_df = pd.read_csv("time_series_covid19_confirmed_US_STATE.csv")

timeseries_state_cases_df.head()
timeseries_country_deaths_df = pd.read_csv("time_series_covid19_deaths_COUNTRY_ONLY.csv")

timeseries_country_deaths_df.tail()
timeseries_state_deaths_df = pd.read_csv("time_series_covid19_deaths_US_STATE.csv")

timeseries_state_deaths_df.head()
sum = 0

def display_val(val):

    return int(np.round(val*sum/100,0))



#timeseries_confirmed_df.head()

#print(len(timeseries_confirmed_df))

#print(np.sum(np.array(timeseries_confirmed_df.loc["Canada", "3/31/20"])))

count = 0



months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

limit = [500, 15000, 50000, 100000, 200000, 350000]



fig, ax = plt.subplots(2, 3, figsize=(50, 50))

fig.tight_layout(pad=30)

fig.suptitle("Number of COVID-19 Cases by Country Over Time", size=50)



for i in range(2):

    for j in range(3):

        sum = 0

    

        labels = []

        values = []

        month_index = i*3 + j



        for k in range(len(timeseries_country_cases_df)):

            date = str(month_index+2)+"/"+ str(days_per_month[month_index+1]) + "/2020"

            if timeseries_country_cases_df.loc[k,date] > limit[month_index]:

                labels.append(timeseries_country_cases_df.loc[k, "Country"])

                values.append(timeseries_country_cases_df.loc[k, date])

            count+=1

    

        sum = np.sum(np.array(values))

        #print("labels ", labels)

        #print("values ", values)

        ax[i][j].set_title(str(date), size=60)

        ax[i][j].pie(values, labels=labels, autopct=display_val, textprops= {'fontsize': 40})

        ax[i][j].axis('equal')



plt.show()
months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

limit = [0, 4000, 20000, 40000, 80000, 120000]



fig, ax = plt.subplots(2, 3, figsize=(50, 50))

fig.tight_layout(pad=30)

fig.suptitle("Number of COVID-19 Cases per US state Over Time", size=50)



for i in range(2):

    for j in range(3):

        global sum

        sum = 0

    

        labels = []

        values = []

        month_index = i*3 + j



        for k in range(len(timeseries_state_cases_df)):

            date = str(month_index+2)+"/"+ str(days_per_month[month_index+1]) + "/2020"

            if timeseries_state_cases_df.loc[k,date] > limit[month_index]:

                labels.append(timeseries_state_cases_df.loc[k, "Province_State"])

                values.append(timeseries_state_cases_df.loc[k, date])

            count+=1

    

        sum = np.sum(np.array(values))

        #print("labels ", labels)

        #print("values ", values)

        ax[i][j].set_title(str(date), size=50)

        ax[i][j].pie(values, labels=labels, autopct=display_val, textprops = {'fontsize': 40})

        ax[i][j].axis('equal')



plt.show()

dates =  np.array(timeseries_state_cases_df.columns)[1:]

dates_abbrev = []

for date in dates:

    dates_abbrev.append(date[:-5])

#print("len(dates) ", len(dates))

mindate = 0

maxdate = 213



fig, ax = plt.subplots(2, figsize=(30, 20))

fig.tight_layout(pad=10)



states = ["Florida", "New York", "California", "Washington", "Illinois", "New Jersey", "Texas", "Georgia"]



for state in states:

    cases = np.array(timeseries_state_cases_df.set_index('Province_State').loc[state])

    deaths = np.array(timeseries_state_deaths_df.set_index('Province_State').loc[state])

    

    #print('texas_cases ', texas_cases)

    ax[0].plot(dates_abbrev[mindate:maxdate], cases[mindate:maxdate])

    ax[0].set_xlabel("Date (month/day)", size=30)

    ax[0].set_ylabel("Confirmed Cases", size=30)

    ax[0].set_xticks(range(0, len(dates), 6))

    ax[0].tick_params(labelsize=20)

    ax[0].set_title("COVID-19 Confirmed Cases in U.S. States From January to August 2020", size=40)

    ax[0].annotate(state, (dates_abbrev[-1], cases[-1]), size=20)

    

    ax[1].plot(dates_abbrev[mindate:maxdate], deaths[mindate:maxdate])

    ax[1].set_xlabel("Date (month/day)", size=30)

    ax[1].set_ylabel("Deaths", size=30)

    ax[1].set_xticks(range(0, len(dates), 6))

    ax[1].tick_params(labelsize=20)

    ax[1].set_title("COVID-19 Deaths in the U.S. States From January to August 2020", size=40)

    ax[1].annotate(state, (dates_abbrev[-1], deaths[-1]), size=20)





#plt.xticks(range(0, len(dates), 6), rotation=60)



#plt.rc('font', size=25)

#plt.rc('figure', titlesize=50)

#plt.rc('axes', labelsize=30)

#plt.rc('xtick', labelsize=20)

#plt.rc('ytick', labelsize=20)



plt.show()
dates =  np.array(timeseries_country_cases_df.columns)[1:]

dates_abbrev = []

for date in dates:

    dates_abbrev.append(date[:-5])

#print("len(dates) ", len(dates))

mindate = 0

maxdate = 213



fig, ax = plt.subplots(2, figsize=(30, 20))

fig.tight_layout(pad=10)



countries = ["China", "US", "Italy", "Brazil", "India", "France", "United Kingdom"]



for country in countries:

    

    cases = np.array(timeseries_country_cases_df.set_index('Country').loc[country])

    ax[0].plot(dates_abbrev[mindate:maxdate], cases[mindate:maxdate])

    ax[0].annotate(country, (dates_abbrev[-1], cases[-1]), size=20)

    ax[0].set_xticks(range(0, len(dates), 6))

    ax[0].tick_params(labelsize=20)

    ax[0].set_title("COVID-19 Cases By Country From January to August 2020", size=40)

    

    deaths = np.array(timeseries_country_deaths_df.set_index('Country').loc[country])

    ax[1].plot(dates_abbrev[mindate:maxdate], deaths[mindate:maxdate])

    ax[1].annotate(country, (dates_abbrev[-1], deaths[-1]), size=20) 

    ax[1].set_xticks(range(0, len(dates), 6))

    ax[1].tick_params(labelsize=20)

    ax[1].set_title("COVID-19 Deaths By Country From January to August 2020", size=40)

    



plt.show()