# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.





%matplotlib inline

import matplotlib.pyplot as plt





df = pd.read_csv("../input/City_MedianListingPrice_AllHomes.csv")



df.head(5)

df.describe()
# view the row counts for each different 'Metro' area

df['Metro'].value_counts()
#get the value counts for the 'State' column of the data set

df['State'].value_counts()
topStates = df['State'].value_counts()[:6] #returns series where the state names are the indices

states = [] # list will hold states with most data



for i,value in enumerate(topStates):

    states.append(topStates.index[i]) # isolate string indices representing states with 6 most records in data set

    

# display box plot of median home data grouped by states with the 6 most rows of data

def boxPlots():

    

    fig, axes = plt.subplots(2,3, figsize=(12,8))

    

    for state, ax in zip(states,axes.ravel()):

        tempdf = df.loc[df['State'] == state,['State','2017-09']]

        ax.set_title(state)

        tempdf.boxplot(ax=ax)

        

    plt.suptitle("median city home prices in each state")   

    plt.show()

    return



boxPlots()
# the supplied series contains all the median home prices for a given city in a single month

def averageSeries(series): 

    s2 = series[series.notnull()] # makes new series with only rows that have data. removes nulls

    #print(s2)

    return np.average(s2) # can now average values and skip nulls





# plots the line graphs the average home prices for each state over time

def linePlots():

    import matplotlib.dates as mdates

    columns = df.columns.tolist()

    columns1 = columns[5:]

    columns1 = ['State'] + columns1

    dates = pd.date_range(start='2010-01', end='2017-09', freq='MS')

    for state in states:

        tempdf = df.loc[df["State"] == state, columns1]

        annualAverages = []

        

        for col in columns[5:]:

            annualAverages.append(averageSeries(tempdf[col])) # list of averages. one for each month for a single state

    

        fig, ax = plt.subplots(1,1, figsize=(9,6))

        ax.set_title('average home price in '+state)  

        plt.sca(ax)

        fig.autofmt_xdate()

        ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')

        ax.plot(dates, annualAverages)



    

linePlots()
def linePlotOverLay():

    import matplotlib.dates as mdates

    columns = df.columns.tolist()

    columns1 = columns[5:]

    columns1 = ['State'] + columns1

    colors =('red','green','darkblue','black','gold','pink')

    dates = pd.date_range(start='2010-01', end='2017-09', freq='MS')

    fig, ax = plt.subplots(1,1,figsize=(12,8))

    ax.set_title('average home prices in states with most data') 

    

    for state,color in zip(states,colors):

        tempdf = df.loc[df["State"] == state, columns1]

        annualAverages = []

        

        for col in columns[5:]:

            annualAverages.append(averageSeries(tempdf[col])) # list of averages. one for each month for a single state

        



        plt.sca(ax)

        fig.autofmt_xdate()

        ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')

        ax.plot(dates, annualAverages, c=color, label=state) 

        plt.legend(loc=2,bbox_to_anchor=(1.01,1.02))

        

    plt.show()

    

linePlotOverLay()
def compareStateToMetro(state,metro):

    import matplotlib.dates as mdates

    columns = df.columns.tolist()

    columns1 = columns[5:]

    columns2 = ['State'] + columns1

    dates = pd.date_range(start='2010-01', end='2017-09', freq='MS')

    fig, ax = plt.subplots(1,1,figsize=(10,6))

    ax.set_title(metro+' compared with '+state) 

    

    stateWideAverages = []

    metroWideAverages = []

    for col in columns1:

        stateWideAverages.append(np.nanmean(df.loc[df["State"]==state,[col]]))

        

    for col in columns1:

        metroWideAverages.append(np.nanmean(df.loc[df["Metro"]==metro,[col]]))

        

    fig.autofmt_xdate()

    ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')

    ax.plot(dates, stateWideAverages, c='red', label=state+" state")

    ax.plot(dates,metroWideAverages,c='blue',label=metro)  

    plt.legend(loc=2,bbox_to_anchor=(1.01,1))

    

    plt.show()

    

compareStateToMetro("NY","New York")

compareStateToMetro("PA",'Philadelphia')

compareStateToMetro("CA","Los Angeles-Long Beach-Anaheim")

compareStateToMetro("FL",'Miami-Fort Lauderdale')
 # three columns: stateaverage home prices, metrohomeprices, months (need to extract as a column name into value)

def describeStateVsMetro(state,metro):

    months = df.columns.tolist()[5:] # selects only the column names that correlate to monthly price data

    tempdf = df.loc[df['State'] == state, months] # dataframe will isolate only the data for the desired state

    

    caAverages = tempdf.mean() # pandas mean() will ignore nan. caAverages is a Series object. 

    #each month column will be an index and have mean value as its correlated value.

    

    tempdf = df.loc[df['Metro'] == metro, months]

    

    laAverages = tempdf.mean()  # similar set of values for cities in the metro area

    

    dic = {

        'CaMonthlyAveragePrice' : caAverages,

        'LaMonthlyAveragePrice' : laAverages}

    

    newDf = pd.DataFrame(dic, index=months)

    print(newDf.describe())

    

describeStateVsMetro('CA',"Los Angeles-Long Beach-Anaheim")
def describeStateVsMetro2(state,metro,date): # three columns one stateaverage home prices, metrohomeprices, months (need to extract as acolumn name into value)



    

    tempdf = df.loc[df['State'] == state, [date]]

    print("statistics for California\n%s\n"% tempdf.describe())

    tempdf = df.loc[df['Metro'] == metro, [date]]

    print("statistics for LA-Anaheim metro\n%s\n" % tempdf.describe())

    

describeStateVsMetro2('CA', "Los Angeles-Long Beach-Anaheim", '2012-02')

describeStateVsMetro2('CA', "Los Angeles-Long Beach-Anaheim", '2014-02')
def histogram(state, metro, date):

    statedf = df.loc[df['State'] == state, [date]]

    metrodf = df.loc[df['Metro'] == metro, [date]]

    

    fig , ax = plt.subplots(1,1,figsize=(12,8))

    statemean = statedf[date].mean()

    metromean = metrodf[date].mean()

   

    statedf.hist(ax=ax,color='blue',alpha=1,label='StateWideValues',bins=40,histtype='step',linewidth=3.0, hatch='o')

    metrodf.hist(ax=ax,color='red',alpha=1,label="LAMetroValues",bins=40, histtype='bar',linewidth=3.0)

    ax.axvline(statemean, color='blue',linestyle='--',label='state mean')

    ax.axvline(metromean, color='red',linestyle='--',label='metro mean')

    plt.sca(ax)

    plt.xticks(np.linspace(0,3000000,11),rotation=45)

    plt.title("state vs metro average city listings for 2014-02")

    plt.legend()

    plt.show()

        

histogram('CA',"Los Angeles-Long Beach-Anaheim",'2014-02') 