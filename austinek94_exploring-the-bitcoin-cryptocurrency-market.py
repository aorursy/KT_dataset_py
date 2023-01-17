# Importing pandas

import pandas as pd



# Importing matplotlib and setting aesthetics for plotting later.

import matplotlib.pyplot as plt

%matplotlib inline

%config InlineBackend.figure_format = 'svg' 

plt.style.use('fivethirtyeight')



# Reading datasets/coinmarketcap_06122017.csv into pandas

dec6 = pd.read_csv('../input/datacamp-projects/Exploring the Bitcoin Cryptocurrency Market/datasets/coinmarketcap_06122017.csv')



# Selecting the 'id' and the 'market_cap_usd' columns

market_cap_raw = dec6[['id','market_cap_usd']]



# Counting the number of values

print(market_cap_raw.count())
# Filtering out rows without a market capitalization

#cap = market_cap_raw[market_cap_raw['market_cap_usd']>0]



#another convenient alternative

cap = market_cap_raw.query('market_cap_usd > 0')



# Counting the number of values again

print(cap.count())
#Declaring these now for later use in the plots

TOP_CAP_TITLE = 'Top 10 market capitalization'

TOP_CAP_YLABEL = '% of total cap'



# Selecting the first 10 rows and setting the index

cap10 = cap.head(10).set_index(cap.id[:10])



# Calculating market_cap_perc

cap10 = cap10.assign(market_cap_perc = lambda x: (x.market_cap_usd / cap.market_cap_usd.sum()) * 100)



# Plotting the barplot with the title defined above 

ax = cap10.plot.bar( x='id',y='market_cap_perc',title=TOP_CAP_TITLE)



# Annotating the y axis with the label defined above

ax.set_ylabel(TOP_CAP_YLABEL)
# Colors for the bar plot

COLORS = ['orange', 'green', 'orange', 'cyan', 'cyan', 'blue', 'silver', 'orange', 'red', 'green']



# Plotting market_cap_usd as before but adding the colors and scaling the y-axis  

ax = cap10.plot.bar( x='id',y='market_cap_usd',title=TOP_CAP_TITLE, color =COLORS, log=True)



# Annotating the y axis with 'USD'

ax.set_ylabel("USD")



# Final touch! Removing the xlabel as it is not very informative

ax.set_xlabel('')
# Selecting the id, percent_change_24h and percent_change_7d columns

volatility = dec6[['id', 'percent_change_24h','percent_change_7d']]



# Setting the index to 'id' and dropping all NaN rows

volatility = volatility.set_index('id').dropna()



# Sorting the DataFrame by percent_change_24h in ascending order

volatility = volatility.sort_values('percent_change_24h', ascending=True)



# Checking the first few rows

print(volatility.head())
#Defining a function with 2 parameters, the series to plot and the title

def top10_subplot(volatility_series, title):

    # Making the subplot and the figure for two side by side plots

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))

    

    # Plotting with pandas the barchart for the top 10 losers

    ax = (volatility_series[:10].plot.bar(color='darkred', ax=axes[0]))

    

    # Setting the figure's main title to the text passed as parameter

    fig.suptitle(title)

    

    # Setting the ylabel to '% change'

    ax.set_ylabel('% change')

    

    # Same as above, but for the top 10 winners

    ax = (volatility_series[-10:].plot.bar(color='darkblue', ax=axes[1]))

    

    # Returning this for good practice, might use later

    return fig, ax



DTITLE = "24 hours top losers and winners"



# Calling the function above with the 24 hours period series and title DTITLE  

fig, ax = top10_subplot(volatility.percent_change_24h,DTITLE)
# Sorting in ascending order

volatility7d = volatility.sort_values('percent_change_7d', ascending=True)



WTITLE = "Weekly top losers and winners"



# Calling the top10_subplot function

fig, ax = top10_subplot(volatility7d.percent_change_7d, WTITLE)
# Selecting everything bigger than 10 billion 

largecaps = cap.query('market_cap_usd>10000000000')



# Printing out largecaps

print(largecaps.head())
# Making a nice function for counting different marketcaps from the

# "cap" DataFrame. Returns an int.

# INSTRUCTORS NOTE: Since you made it to the end, consider it a gift :D

def capcount(query_string):

    return cap.query(query_string).count().id



# Labels for the plot

LABELS = ["biggish", "micro", "nano"]



# Using capcount count the biggish cryptos

biggish = capcount('market_cap_usd>300000000')



# Same as above for micro ...

micro = capcount('market_cap_usd>50000000 & market_cap_usd<300000000')



# ... and for nano

nano =  capcount('market_cap_usd<50000000')



# Making a list with the 3 counts

values = [biggish,micro, nano]

#print(values)

# Plotting them with matplotlib 

plt.bar(range(len(values)), values, tick_label =LABELS)