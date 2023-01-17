import pandas as pd
import numpy as np
data = pd.read_csv('../input/stats.csv')
data.head()
data_united = data[data.team=='Manchester United']
#Extracting only data that is valid for Manchester United.
features_wins = ['wins','season'] 
data_winsVseason = data_united[features_wins] #Extracting out wins for every season.
features_loss= ['losses','season']
data_lossVseason = data_united[features_loss] #Extracting out losses for every season.
import matplotlib.pyplot as plt

#Plotting Wins and Losses v/s Season
plt.rcParams['figure.figsize'] = [12, 8] #creates a graph of 10x7 inches
plt.plot(data_winsVseason.season, data_winsVseason.wins, label = 'Wins', marker='o') #plot Wins v Season
plt.plot(data_lossVseason.season, data_lossVseason.losses, label = 'Losses', marker='o') #plot Loss vs Season, with markers and label
plt.ylim(0,30) #for setting y limits from 15 to 30.
plt.grid(which='major', axis='both', linestyle='-.', linewidth=0.75) #plotting grid. See pyplot.grid in Google.
plt.xticks(rotation=60, fontsize=14) #rotate x axis labels and increase font.
plt.yticks(fontsize=14) #increase yticks fontsize.
plt.legend(fontsize=14) #show labels as legend
plt.title('Wins and Losses v/s Season', fontsize = 20)

try: #try catch is used because here x-coord is a string like 2006-2007 and hence gives a TypeError. So TypeError errors are overlooked in except statement.
    
    plt.annotate('Moyes comes in', xy=('2013-2014',19), xytext=('2013-2014', 21),arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8), fontsize =10) 
    #For the wins curve. The label is Moyes comes in.
    #xy is the position of marker point. xytext is position of label text. 
    
    plt.annotate('Moyes comes in', xy=('2013-2014',12), xytext=('2013-2014', 14),arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8), fontsize =10) 
    #For losses curve. The label is Moyes comes in.
    
    plt.annotate('Van Gaal comes in', xy=('2014-2015',20), xytext=('2014-2015', 23),arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8), fontsize =10) 
    #For wins curve when LVG comes in.
    
    plt.annotate('Van Gaal comes in', xy=('2014-2015',8), xytext=('2014-2015', 11),arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8), fontsize =10)
    #For losses curve when LVG comes in.
    
    plt.annotate('Mourinho comes in', xy=('2016-2017',18), xytext=('2016-2017', 20),arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8), fontsize =10) 
    #For wins curve when Mou comes in. 
    
    plt.annotate('Mourinho comes in', xy=('2016-2017',5), xytext=('2016-2017', 8),arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8), fontsize =10) 
    #For losses curve when Mou comes in. 
    
    plt.text('2009-2010',16,'The Famous SAF Era', horizontalalignment='center', verticalalignment='center', fontsize =16, bbox=dict(facecolor='red', alpha=0.5)) 
    #For the middle big text
except TypeError: 
    pass

