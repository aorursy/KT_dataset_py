import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.svm import SVR
import numpy as np
import calendar
import seaborn as sns
from IPython.display import display

sns.set()
rootPath = "../"
filename = "input/airplane-crashes-since-1908/Airplane_Crashes_and_Fatalities_Since_1908.csv"
fullPath = rootPath + filename
dataFrame = pd.read_csv(fullPath)
dataFrame.sample(5)
total,features = dataFrame.shape
print("Total Number of recorded Crashes = " + str(total))
print('Total Number of crash Features:  ' + str(features))
print('**********************************************')
print('Initial Features List')
print(dataFrame.dtypes)
print('')
print('**********************************************')
print('Total Number of missing data \n') 
print(dataFrame.isnull().sum().sort_values())
print('')
print('**********************************************')
print() 

# Transform date information from string to standard date format
dataFrame['Date'] = pd.to_datetime(dataFrame['Date']) 

#extract "Year" from the date data and convert to np array
# This is one way of extracting data from the Pandas dataframe
planeCrashYear = dataFrame['Date'].apply(lambda x : x.year).values
years, frequency = np.unique(planeCrashYear,return_counts=True)


#plot the graphs for years and frequency using matplotlib and seaborn
fig = plt.figure(figsize=(18, 6))
ax1 = fig.add_subplot(1,1,1)
ax1.set_xticks(np.arange(years.min(),years.max(),2))
plt.xlim(years.min()-1, years.max()+1)
plt.xticks(rotation=90)
plt.bar(years,frequency,align="center",alpha=0.2)
plt.plot(years,frequency,'_',color="red",linewidth=1,marker='.')
plt.xlabel('Years')
plt.ylabel('Flight Crashes')
plt.grid(True)
plt.show()

# Predicting flight crashed in near future from 2009 using Support Vector Regression

YEARS_OF_PREDICTION = 15

X = [[i] for i in years]
svd_model = SVR(kernel='rbf',C=1e3,gamma= 0.005)
svd_model.fit(X,frequency)
predict = lambda i : svd_model.predict(i)
p_years = np.arange(years.min(),years.max()+YEARS_OF_PREDICTION+1,1).reshape(-1,1)
p_frequency = predict(p_years)

df_flightCrashPrediction = pd.DataFrame({'Years': p_years.flatten(),
                                         'Crashes': p_frequency}).set_index('Years')


## Predicting the flight crashes in coming years using Support Vector Regression from sklearn.
fig = plt.figure(figsize=(18, 6))
ax1 = fig.add_subplot(1,1,1)
ax1.set_xticks(np.arange(years.min(),years.max()+YEARS_OF_PREDICTION +2,2))
plt.xlim(years.min()-1, years.max()+YEARS_OF_PREDICTION+1)
plt.xticks(rotation=90)
plt.plot(years,frequency,color="red",linewidth=1,marker='.')
plt.plot(p_years,p_frequency,'--',color="blue",linewidth=1,marker='.')
plt.xlabel('Years')
plt.legend(['Actual AirCraft Crashes','Predicted AirCraft Crashes',])
plt.title("Aircraft Crashes Prediction in coming years using Support Vector Regression")




#extract "Month" from the date - this time create a seperate data frame and the sort by month. 
planeCrashMonth = dataFrame.Date.map(lambda x : x.month).value_counts().to_frame().sort_index()
#plot the graphs for years and frequency
plt.figure(figsize=(16, 8))
color = ['#009BFF','#00F3FF','#04FF00','#04FF00','#BDFF00','#FF0000','#FF0000','#FF5900','#FFB600','#FFB600','#00F3FF']
plt.bar(planeCrashMonth.index,planeCrashMonth.Date,align="center",alpha=0.4,color=color)

plt.xticks(np.arange(14), calendar.month_name[0:13])

plt.title('Plot of aircraft accidents in each month - the color shows season of the year')
plt.xlabel('Months')
plt.ylabel('Flight Crashes')
plt.grid(True)

# There are some NAN entires for Aboard and Ground columns and needs to be filled first. We will assume zero passengers on
# board for these entires so they doesnot effect the analysis.
dataFrame['Aboard'] = dataFrame['Aboard'].fillna(0)
dataFrame['Fatalities'] = dataFrame['Fatalities'].fillna(0)

#extract "Year" from the date - this time create a new column in the orignal dataframe.
dataFrame['Year'] = dataFrame.Date.map(lambda x : x.year)

passengerPerYear = dataFrame.groupby(['Year'])[["Aboard","Fatalities"]].sum()

plt.figure(figsize=(16, 8))
plt.bar(passengerPerYear.index,passengerPerYear["Aboard"],linewidth =0,width=0.4, align='center', color ='#4444ff')
plt.bar(passengerPerYear.index,passengerPerYear["Fatalities"],linewidth =0,width=0.4,align='center', color="#ff5555")
plt.plot(passengerPerYear.index,passengerPerYear["Aboard"],'_', mew = 5,color ='#44ff88')

plt.xlim(passengerPerYear.index.min()-1, passengerPerYear.index.max()+1)
plt.xticks(np.arange(passengerPerYear.index.min()-1, passengerPerYear.index.max()+1,2))
plt.xticks(rotation=90)
plt.grid(True)
plt.legend(['Total Passengers','Survivors','Fatalities'])
plt.xlabel('Years')
plt.ylabel('Total Passengers')
plt.show()


#computing the survival rate 
passengerPerYear['SurvivalRate']= ((passengerPerYear["Aboard"] - passengerPerYear["Fatalities"])/passengerPerYear["Aboard"])*100
plt.figure(figsize=(16, 7))

plt.bar(passengerPerYear.index, passengerPerYear.SurvivalRate,width=1,linewidth=0,align='center',alpha=0.7,
        bottom= (100-passengerPerYear.SurvivalRate),color ='#4444ff')
plt.bar(passengerPerYear.index,100 - passengerPerYear.SurvivalRate,width=1, linewidth=0,alpha=0.7, 
        color="#ff5555",align='center')


plt.xlim(passengerPerYear.index.min()-2, passengerPerYear.index.max()+2)
plt.xticks(np.arange(passengerPerYear.index.min(), passengerPerYear.index.max()+2,2))
plt.xticks(rotation=45)
plt.yticks(np.arange(0,101,10))
plt.grid(True)
plt.title("Passange Survival and Fatality ratio for each year")
plt.legend(["Survival Rate","Fatality Rate"],frameon=True)
plt.ylabel('Percentage (%)')
plt.xlabel('Years')
plt.show()


#predicting the survival rate of the passengers over the comming years.
X = [[i] for i in years]
svd_model = SVR(kernel='rbf',C=1e3,gamma= 0.005)
svd_model.fit(X,(100 - passengerPerYear['SurvivalRate']))
predict = lambda i : svd_model.predict(i)
p_years = np.arange(years.min(),years.max()+YEARS_OF_PREDICTION+1,1).reshape(-1,1)
p_fatality = predict(p_years)


plt.figure(figsize=(16, 7))
plt.bar(passengerPerYear.index, passengerPerYear.SurvivalRate,width=1,linewidth=0,align='center',alpha=0.7,
        bottom= (100-passengerPerYear.SurvivalRate),color ='#4444ff')
plt.bar(passengerPerYear.index,100 - passengerPerYear.SurvivalRate,width=1, linewidth=0,alpha=0.7, 
        color="#ff5555",align='center')
plt.plot(p_years,p_fatality,'--',color="lightgreen",linewidth=3,marker='.')

plt.xlim(years.min()-1, years.max()+YEARS_OF_PREDICTION+1)
plt.xticks(np.arange(years.min(),years.max()+YEARS_OF_PREDICTION +2,2))
plt.xticks(rotation=45)
plt.yticks(np.arange(0,101,10))
plt.grid(True)
plt.title("Passange Survival and Fatality ratio for each year")
plt.legend(["Predicted Fatality(%)","Survival Rate(%)","Fatality Rate(%)"],frameon=True)
plt.ylabel('Percentage (%)')
plt.xlabel('Years')
plt.show()


# Adding a new feature to hold the country of accident.
dataFrame['Location'] =  dataFrame['Location'].fillna(",")
dataFrame['Region'] = dataFrame.Location.map(lambda x: str(x).split(',')[-1].strip())
CrashByRegions = dataFrame.groupby('Region')['Region'].count()

print("TOP 10 REGIONS WITH HIGHEST FLIGHT CRASH HISTORY")
display(CrashByRegions.sort_values(ascending=False).head(10))
print(" ")
#CODE COMMENTED
#plt.figure(figsize=(16, 7))
#brazil_flights = dataFrame[['Year']].loc[dataFrame.Region == 'Brazil'].groupby('Year')[['Year']].count()
#plt.bar(brazil_flights.index, brazil_flights.values)
#plt.xlabel('YEARS')
#plt.ylabel('NUMBER OF CRASHES')
#plt.title('CRASH HISTORY OF BRAZIL')
#plt.grid(True)
#plt.show()
print("This information can be used for getting the flight accident history based on regions such as Malaysia")
print(" ")
print("****************************   FLIGHT CRASH HISTORY OF MALAYSIA     ***********************************")
display(dataFrame[["Date","Aboard","Fatalities","Location","Summary"]].loc[dataFrame['Region'] == 'Malaysia'])
df_totalFlight = pd.read_csv('../input/worldbankairpassenger/API_IS.AIR.PSGR_DS2_en_csv_v2_887266.csv')
total_flights = df_totalFlight.fillna(0).drop(['Country Name','Country Code','Indicator Name','Indicator Code'],axis=1).sum().to_frame()
total_flights.index = total_flights.index.astype(float)
total_flights = total_flights.drop([1960,1961,1962,1963,1964,1965,1966,1967,1968,1969,2019])
#####################
X = [[i] for i in total_flights.index]
svd_model = SVR(kernel='rbf',C=1e3,gamma= 0.001)
svd_model.fit(X,total_flights.values/1e10)
predict = lambda i : svd_model.predict(i)
p_years = np.arange(1970,2030,1).reshape(-1,1)
p_totalflights = predict(p_years)

plt.figure(figsize=(14, 6))
plt.plot(X,total_flights.values)
plt.plot(p_years,p_totalflights*1e10)
plt.xlim(1970, 2025)
plt.xticks(np.arange(1970,2025,2))
plt.xticks(rotation=45)
plt.ticklabel_format(useOffset=False)
plt.title("Total Flights in the world in millions")
plt.xlabel('Years')
plt.ylabel('Total Flights in Millions')
plt.legend(['Actual','Predicted'])
plt.tight_layout()
plt.show()

p_totalflights = p_totalflights*1e10
df_Flights = pd.DataFrame({'Years': p_years.ravel(),
                                           'Total Flights':p_totalflights,
                                          'TotalCrashes': 0}).set_index('Years')
#move all the crash info from same year to new dataframe
df_Flights['TotalCrashes'].loc[df_Flights.index] = df_flightCrashPrediction.Crashes.reindex(df_Flights.index.values)
df_Flights.fillna(0)
df_Flights['Safety Index'] = (df_Flights['Total Flights'] - df_Flights.TotalCrashes)/df_Flights['Total Flights']*100

# Plotting
plt.figure(figsize=(14, 6))
plt.plot(df_Flights.index.values,df_Flights['Safety Index'].values)
plt.xlim(1970, 2025)
plt.xticks(np.arange(1970,2025,2))
plt.xticks(rotation=45)
plt.ticklabel_format(useOffset=False)
plt.title("Safety Index of flight over Years (Percentage %)")
plt.xlabel('Years')
plt.ylabel('Flight safety measure(%)')
plt.tight_layout()
plt.show()