import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/PakistanDroneAttacksWithTemp Ver 10 (October 19, 2017).csv", encoding='latin1')

# Removing the last row as it contained junk values
df = df.drop(df.index[[403, 404]])

# Dropiing the Axes that are not required
df=df.drop(['Special Mention (Site)', 'References','Time'], axis=1)

# Printing the number of missing values in the data
for col in df:
    print (col, ": ", df[col].isnull().sum())

def filling_nan(data):
    
    # Initializing 'col' which stores the Column names of the columns whose null values need to be filled by 0
    col = ['Al-Qaeda', 'Taliban', 'Civilians Min', 'Civilians Max', 'Foreigners Min', 'Foreigners Max', 
           'Total Died Min', 'Total Died Mix', 'Injured Min', 'Injured Max']
    
    data[col] = data[col].fillna(0)                                  # Filling the above columns with 0
    data['Women/Children  '] = data['Women/Children  '].fillna('N')  # Filling the Women/Children column's missing values with 'N'
    data['Temperature(C)'] = data['Temperature(C)'].fillna(data['Temperature(C)'].mean())
    data['Temperature(F)'] = data['Temperature(F)'].fillna(data['Temperature(F)'].mean())
    
    return data
data = filling_nan(df)

# Printing the number of missing values in the data
for col in df:
    print (col, ": ", df[col].isnull().sum())
def dt(data):
    data['DateTime'] = pd.Series()
    for i in range(0, len(data.Date)):
        
        # Converting to date-time format
        frame = data['Date'][i]
        data['DateTime'][i] =  datetime.datetime.strptime(frame, '%A, %B %d, %Y')
    
    return data
data = dt(data)
data['Terrorists'] = data['Al-Qaeda'] + data['Taliban']                   # Adding Taliban and Al-Qaeda personnel killed
data['Civilians'] = (data['Civilians Min'] + data['Civilians Max'])/2     # Taking average of Max and Min Civilians killed
data['Civilians'] = np.ceil(data['Civilians'])                            # Rounding up the average
data['Injured'] = (data['Injured Min'] + data['Injured Max'])/2           # Taking average of Max and Min Civilians injured
data['Injured'] = np.ceil(data['Injured'])                                # Rounding up the average
data['Total Died'] = (data['Total Died Min'] + data['Total Died Mix'])/2           # Taking average of Max and Min Civilians injured
data['Total Died'] = np.ceil(data['Total Died'])
data['Foreigners'] = (data['Foreigners Min'] + data['Foreigners Max'])/2  # Taking average of Max and Min Foreigners killed
data['Foreigners'] = np.ceil(data['Foreigners'])                          # Rounding up the average

data['Innocents'] = data['Total Died'] - data['Terrorists']                # Marking all the non-terrorists killed



# In some observations, the Total Died was not calculated accurately and there were negative values representing the 
# number of people killed. In those cases, we used different method to calculate the Civilian Casualties
for i in range(0, len(data.Innocents)):
    if data['Innocents'][i] <0:
        data['Innocents'][i] = data['Civilians'][i] + data['Foreigners'][i]
        
    # Changing the Labels in 'Women/Children' column to Binary values
    if data['Women/Children  '][i] == 'N':
        data['Women/Children  '][i] = 0
    elif data['Women/Children  '][i] == 'Y':
        data['Women/Children  '][i] = 1

# Accuracy is the ratio of terrorists killed to the total number of people killed in a drone strike
data['Accuracy'] = (data['Terrorists'])/(data['Terrorists'] + data['Innocents'])
data['Accuracy'] = data['Accuracy'].fillna(0)

# Removing the columns that are not required for further analysis, replacement have been made for some of these columns
data = data.drop(['Date', 'Province', 'Civilians Min', 'Civilians Max', 'Foreigners Min',
                  'Foreigners Max', 'Total Died Min', 'Total Died Mix', 'Injured Min', 'Injured Max', 
                 'Temperature(C)', 'Temperature(F)'], axis=1)
data.head()
# Creating new columns for Year, Day of week and Month from the given Date-Time format
data['Year'] = np.nan
data['Weekday'] = np.nan
data['Month'] = np.nan

# Extracting the Year of attack, Day of attack and the month of Attack
for i in range(0, len(data.Year)):
    data.loc[:,'Year'][i]= data.loc[:,'DateTime'][i].year
    data.loc[:,'Weekday'][i] = data['DateTime'][i].weekday()
    data.loc[:,'Month'][i] = data['DateTime'][i].month

# Converting Year and Weekday values from float to int
data['Year'] = data['Year'].astype("int")
data['Weekday'] = data['Weekday'].astype("int")
    
# Grouping the new column by the Year, Weekday & Month and corresponding Drone Attacks
by_year = data.groupby(['Year'])['No of Strike'].sum()
by_weekday = data.groupby(['Weekday'])['No of Strike'].sum()
by_month = data.groupby(['Month'])['No of Strike'].sum()

data['City'] = data['City'].replace('South waziristan', 'South Waziristan')
data['City'] = data['City'].replace('south Waziristan', 'South Waziristan')
data['City'] = data['City'].replace('south waziristan', 'South Waziristan')
data['City'] = data['City'].replace('Lower Kurram Agency', 'Kurram')
data['City'] = data['City'].replace('Kurram Agency', 'Kurram')
data['City'] = data['City'].replace('Khyber Agency', 'Khyber')
data['City'] = data['City'].replace('Hungu', 'Hangu')
data['City'].unique()
data_byyear = pd.DataFrame()
data_byyear['Terrorists'] = data.groupby(['Year'])['Terrorists'].sum()
data_byyear['Innocents'] = data.groupby(['Year'])['Innocents'].sum()
data_byyear['Injured'] = data.groupby(['Year'])['Injured'].sum()
data_byyear['No of strikes'] = data.groupby(['Year'])['No of Strike'].sum()
data_byyear['Accuracy'] = data.groupby(['Year'])['Accuracy'].mean()
data_byyear['Total Died'] = data.groupby(['Year'])['Total Died'].sum()
data_byyear['Avg Innocents Killed'] = data.groupby(['Year'])['Innocents'].mean()
data_byyear.head()
# Initializing start-date and end-date
start_date = datetime.date(2004,1,1)
end_date= datetime.date(2017,12,31)

# Creating object containing all the dates from 2004 to 2017
days = pd.date_range(start_date, end_date)

# Making the object the index of the new DataFrame
datadate = pd.DataFrame(index=days)
datadate.index.name='Date'

datadate['Terrorists']=np.nan
datadate['Civilians']=np.nan
datadate['Injured']=np.nan
datadate['Foreigners']=np.nan
datadate['Month']=np.nan
datadate['Year']=np.nan
datadate['Strikes']=np.nan
datadate['Deaths']=np.nan

# Storing Month and Year values in new columns for future use
for i in range(len(datadate)):
    datadate['Month']=days.strftime("%B")
    datadate['Year']=days.year

# Filling the new DataFrame with values. These values come from the dates in which Drone Attack took place.
# All remaining dates will contain nan
for i in range(len(data)):
    if(data['DateTime'][i] in datadate.index):
        
        datadate['Deaths'][data['DateTime'][i]]=data['Total Died'][i]
        datadate['Strikes'][data['DateTime'][i]]=data['No of Strike'][i]    
        datadate['Terrorists'][data['DateTime'][i]]=data['Terrorists'][i]
        datadate['Civilians'][data['DateTime'][i]]=data['Civilians'][i]
        datadate['Injured'][data['DateTime'][i]]=data['Injured'][i]
        datadate['Foreigners'][data['DateTime'][i]]=data['Foreigners'][i]

# Replacing nan values with 0
datadate = datadate.fillna(0)

datadate.head()
fig1 = plt.figure(figsize=(14, 10))
fig1 = plt.bar(left=list(by_year.keys()), height=by_year.values)
plt.xlabel("Year")
plt.ylabel("No. of attacks")
plt.title("Drone attacks by Year")
plt.xticks(list(by_year.keys()))
plt.grid()
plt.show()
fig2 = plt.figure(figsize=(14, 10))
fig2 = plt.bar(left=list(by_weekday.keys()), height=by_weekday.values)
plt.xlabel("Day of the Week")
plt.ylabel("No. of attacks")
plt.title("Drone attacks by Weekday")
plt.xticks(range(0,7), ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'))
plt.show()
fig3 = plt.figure(figsize=(14, 10))
fig3 = plt.bar(left=list(by_month.keys()), height=by_month.values)
plt.xlabel("Month")
plt.ylabel("No. of attacks")
plt.title("Drone attacks by Months")
plt.xticks(range(1,13), ('January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'))
plt.show()
"""Precision is being calculated by taking the number of Terrorists killed and dividing it by the total number 
   of people killed (including 'Civilians' and 'Foreigners') """

fig = plt.figure(figsize=(14, 10))
fig = plt.plot(data_byyear.Accuracy.keys(), data_byyear.Accuracy)
plt.xlabel("YEAR", fontsize=16)
plt.ylabel("SCORE (0-1)", fontsize=18)
plt.title("Precision of Drone attacks over the Years", fontsize=22)
plt.xticks(range(2004, 2018))
plt.ylim(0, 1)
plt.grid()
plt.show()
fig = plt.figure(figsize=(14, 10))
x = data_byyear.Accuracy.keys()

plt.plot(x, data_byyear.Terrorists, color='r', label = 'Terrorists Killed')
plt.plot(x, data_byyear.Innocents, color='g', label = 'Civilians Killed')
plt.plot(x, data_byyear.Injured, color='b', label = 'People Injured')
plt.xticks(range(2004, 2018))
plt.xlabel("YEAR", fontsize=16)
plt.ylabel("Number of people", fontsize=18)
plt.title("Trend of Terrorists and Civilians killed along with Civilians Injured", fontsize=22)
plt.legend(fontsize = 16)
plt.show()
def timeperiodyearTA(data):
    
    a1=[]
    data['Year'] = np.nan
    data['Weekday'] = np.nan
    data['Month'] = np.nan

    for i in range(0, 403):
        data.loc[:,'Year'][i]= data.loc[:,'DateTime'][i].year
    
# Grouping the new column by the Year
    by_year = data.groupby('Year').size()
    df7 = pd.DataFrame(by_year)
    timeyear = [2004.0,2005.0,2006.0,2007.0,2008.0,2009.0,2010.0,2011.0,2012.0,2013.0,2014.0,2015.0,2016.0,2017.0]
    df7['Taliban'] = 0
    df7['Al-Qaeda'] = 0
    for i in range(0, 403):
            for x in range(len(timeyear)):
                if (data.loc[:,'Year'][i] == timeyear[x]):

#Calculating total no Taliban personnel killed
                    df7['Taliban'][timeyear[x]] = df7['Taliban'][timeyear[x]] + data['Taliban'][i]

#Calculating total no Taliban personnel killed
                    df7['Al-Qaeda'][timeyear[x]] = df7['Al-Qaeda'][timeyear[x]] + data['Al-Qaeda'][i]
    x = df7.index.values
    y1= df7['Taliban']
    y2= df7['Al-Qaeda']

#plotting the graph for analysis of drone strike under Bush and Obama
    fig1= plt.figure(figsize=(14, 10))
    plt.plot(x,y1)
    plt.plot(x,y2,color="red")
    plt.ylabel("No. of Terrorist killed")
    plt.title("BUSH VS OBAMA", fontsize = 30)
    plt.xlabel("YEAR")
    plt.legend()
    plt.axvspan(2004, 2009, color='red', alpha=0.3)
    plt.axvspan(2009, 2017, color='blue', alpha=0.3)
    plt.show()
    return df7
timeperiodyearTA(df)
def areaTA(data,z):
# Filtering the areas belonging to Taliban and Al-Qaeda
    b=[]
    c=[]
    d=[]

# b[] is used to store the Locations of Taliban
# c[] is used to store the Longitude of Taliban
# d[] is used to store the Latitude of Taliban

    for i in range(len(data.index)):
        if (data['Taliban'][i].any() != 0.0):
            b.append(data['Location'][i])
            c.append(data['Longitude'][i])
            d.append(data['Latitude'][i])
    df2 = pd.DataFrame(b)
    df2.columns=['Taliban']
    df2['Longitude'] = pd.DataFrame(c)
    df2['Latitude'] = pd.DataFrame(d)
    df3 = df2.drop_duplicates()
    e=[]
    f=[]
    g=[]

# e[] is used to store the Locations of Taliban
# f[] is used to store the Longitude of Taliban
# g[] is used to store the Latitude of Taliban

    for i in range(len(data.index)):
        if (data['Al-Qaeda'][i].any() != 0.0):
            e.append(data['Location'][i])
            f.append(data['Longitude'][i])
            g.append(data['Latitude'][i])
    df4 = pd.DataFrame(e)
    df4.columns=['Al-Qaeda']
    
    df4['Longitude'] = pd.DataFrame(f)
    df4['Latitude'] = pd.DataFrame(g)
    #df2.columns=['Al-Qaeda']
    #df2.sort_values('Taliban').drop_duplicates('Taliban')
    df5 = df4.drop_duplicates()
# z is a flag variable and determines whether data frame for Taliban or Al-Qaeda is to be sent 
# z=1 returns the Taliban Data Frame
# z=0 returns the Al-Qaeda Data Frame
    if (z==1):
        return df3
    else:
        return df4
areaTA(df,1)
    

def Mapping(df):
    import folium
    dfmap1=areaTA(df,1)
    dfmap0=areaTA(df,0)

# folium package is used for plotting on a world map
# dfmap1 contains Taliban data
# dfmap0 contains Al-Qaeda data
    
    subset1 = dfmap1[['Latitude', 'Longitude']]
    tuples1 = [tuple(x) for x in subset1.values]

    subset0 = dfmap0[['Latitude', 'Longitude']]
    tuples0 = [tuple(x) for x in subset0.values]

#tuples0 contains tuple of longitude and latitude of Al-Qaeda
#tuples1 contains tuple of longitude and latitude of Taliban

    c1 = [x for x in tuples1 if not any(isinstance(i, float) and np.isnan(i) for i in x)]
    c0 = [x for x in tuples0 if not any(isinstance(i, float) and np.isnan(i) for i in x)]
    mapit = folium.Map(location=[32.9,70.1],zoom_start = 8)
    for coord in c1:
        folium.Marker( location=[ coord[0], coord[1] ],icon=folium.Icon(color='red',icon='info-sign')).add_to( mapit )
    for coord in c0:
        folium.Marker( location=[ coord[0], coord[1] ],icon=folium.Icon(color='green')).add_to( mapit )

    mapit.save( 'map.html')
    return mapit
Mapping(df)
# Using Wordcloud library to visualize the high-frequency words (https://github.com/amueller/word_cloud)
from wordcloud import WordCloud
a=[]

# Ignoring the Null Values
for i in range(len(data)):
    data['Comments'][i]=str(data['Comments'][i])
    if (data['Comments'][i]!=np.nan):
        a.append(data['Comments'][i])
        
a = [x for x in a if str(x) != 'nan']
        
# Generating the picture using WordCloud 
wordcloud = WordCloud().generate(' '.join(a))

# Plotting the image generated
plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

N_Strikes = data_byyear['No of strikes']

fig = plt.figure(figsize=[24, 20])
fig = plot_acf(N_Strikes)          # Plotting Auto-correlation function
fig = plot_pacf(N_Strikes)         # Plotting Partial auto-correlation function
plt.show()
from statsmodels.tsa import arima_model

data_byyear.index = pd.to_datetime(data_byyear.index, format='%Y')

# Using p=3 and q=1, building the ARMA model
npdf = data_byyear['No of strikes'].astype(float)
results = arima_model.ARMA(npdf,(3, 1)).fit()
print (results)

# Forecasting using the ARMA model defined above
data_byyear['forecast_N_Strikes'] = results.predict(start='2012-01-01', end='2017-01-01', dynamic= False) 

# Plotting the forecast along with the actual data
data_byyear[['No of strikes', 'forecast_N_Strikes']].plot(figsize=(14, 10))
plt.xlabel('YEAR')
plt.ylabel('Log of No of strikes')
plt.show()
from sklearn.metrics import mean_absolute_error

# Creating Series of Actual Values
N_Strikes = data_byyear['No of strikes']
N_Strikes_sub = N_Strikes.loc['2011-01-01': '2017-01-01']

forecast = results.predict(start='2011-01-01', end='2017-01-01', dynamic= True)

# Mean Absolute Error
MAE = mean_absolute_error(N_Strikes_sub, forecast)
print ("Mean Absolute Error (MAE): {}".format(MAE))

# Mean Forecast Error
forecast_error = [N_Strikes_sub[i]-forecast[i] for i in range(len(N_Strikes_sub))]
MFE = (sum(forecast_error))/len(N_Strikes)
print ("Mean Forecast Error (MFE): {}".format(MFE))
plt.figure(figsize=(14, 10))

# Plotting Time-series
plt.plot(data_byyear['No of strikes'], label= 'Time-series data')

# Plotting Rolling Mean
rollmean = pd.Series(data_byyear['No of strikes']).rolling(window=2).mean()
plt.plot(rollmean, label = 'Rolling Mean')

# Plotting Standard Deviation
std_dev = pd.Series(data_byyear['No of strikes']).rolling(window=2).std()
plt.plot(std_dev, label = 'Rolling Standard Deviation')

# Plotting Log( ) function of Time-Series
data_byyear['Log_No of strikes'] = np.log(data_byyear['No of strikes'])
plt.plot(data_byyear['Log_No of strikes'], label = 'Log of No. of Strikes')

plt.legend()
### Plotting Log of the 'No. of Strikes'

mov_mean = pd.Series(data_byyear['Log_No of strikes']).rolling(window=2).mean()
plt.figure(figsize=(14, 10))
plt.plot(data_byyear['Log_No of strikes'], label = 'Log of time-series')
plt.plot(mov_mean, label = 'Rolling Mean of Log')
plt.legend()
# Storing the Log of Time-series in a new variable
ts_log = data_byyear['Log_No of strikes']

ts_log_diff1 = ts_log - ts_log.shift()
data_byyear['First Difference'] = ts_log_diff1
ts_log_diff2 = ts_log_diff1 - ts_log_diff1.shift()
data_byyear['Second Difference'] = ts_log_diff2
data_byyear['Second Difference'].dropna(inplace=True)
plt.figure(figsize=(12, 8))
plt.ylabel('Log of Time-Series (No. of Strikes)')
plt.xlabel('YEAR')
plt.title('Log of No. of Strikes')
plt.plot(ts_log_diff2, label = 'Second Difference of the Log function')
plt.legend()
from statsmodels.tsa.stattools import adfuller

#Perform Dickey-Fuller test:
print ('Results of Dickey-Fuller Test:')
dftest = adfuller(data_byyear['Second Difference'], autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print (dfoutput)
from statsmodels.stats.stattools import durbin_watson
print ("Durbin-Watson statistic of stationary time-series ", durbin_watson(data_byyear['Second Difference']))

# The Durbin-Watson statistic implies a strong auto-correlation in the errors
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig = plt.figure(figsize=[24, 20])
fig = plot_acf(data_byyear['Second Difference'])          # Plotting Auto-correlation function
fig = plot_pacf(data_byyear['Second Difference'])         # Plotting Partial auto-correlation function
plt.show()
# Using p = 1 and q = 0, building the ARMA model

npdf = data_byyear['Second Difference'].astype(float)
results = arima_model.ARMA(npdf,(1, 0)).fit()

# Creating Series of Actual Values
N_Strikes = data_byyear['Second Difference']
N_Strikes_sub = N_Strikes.loc['2012-01-01': '2017-01-01']

forecast = results.predict(start='2012-01-01', end='2017-01-01', dynamic= False) 

# Plotting the forecast along with the actual data
plt.figure(figsize=(14, 10))
plt.plot(data_byyear['Second Difference'], label = 'Second Difference of Log')
plt.plot(forecast, label = 'Forecast')
plt.xlabel('YEAR')
plt.ylabel('Second Difference of Log of No of strikes')
plt.title('ARMA Model of Stationary Time-Series')
plt.legend()
plt.show()
# Using p = 1 and q = 0, building the ARMA model
npdf = data_byyear['Second Difference'].astype(float)
results = arima_model.ARMA(npdf,(1, 0)).fit()

# Creating Series of Actual Values
N_Strikes = data_byyear['Second Difference']
N_Strikes_sub = N_Strikes.loc['2012-01-01': '2017-01-01']

forecast = results.predict(start='2012-01-01', end='2017-01-01', dynamic= False) 

# Mean Absolute Error
MAE = mean_absolute_error(N_Strikes_sub, forecast)
print ("Mean Absolute Error (MAE): {}".format(MAE))

# Mean Forecast Error
forecast_error = [N_Strikes_sub[i]-forecast[i] for i in range(len(N_Strikes_sub))]
MFE = (sum(forecast_error))/len(N_Strikes)
print ("Mean Forecast Error (MFE): {}".format(MFE))
from scipy import stats
from statsmodels.graphics.api import qqplot

# Printing the Normal Test results to check whether residuals come from a normal distribution
resid = results.resid
print (stats.normaltest(resid))

# Contructing qq-plot of residuals 
fig = plt.figure(figsize=(12,8))
qqplot(resid, line='s', fit=True)