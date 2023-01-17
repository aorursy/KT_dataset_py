import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import warnings

warnings.simplefilter('ignore')
freezers = pd.read_csv('../input/Temperature_Data.csv')
freezers.head()
#setting the first column as index

freezers = freezers.set_index('0')
type(freezers.index[0])
#converting index into datetime

freezers.index = pd.to_datetime(freezers.index)
type(freezers.index[0])
#freezers.info()
#checking if hours and minute can be retrived through the index

(freezers.index[1].minute)
#finding out the index at which columns of different sites start/end to create a seperate dataframe per site 

#to use later for analysis

indexing = {}

j=1

for i in freezers.columns:

    #splitting the column names which were of the format - "Site-1 > Freezer-1"

    print(((i.split(">")[0]).split("-")[1]),"-",j)

    #key of dictionary "indexing" is the site number and the value will be updated with the last column number of the site

    indexing[((i.split(">")[0]).split("-")[1]).strip()] = j

    j+=1            
#the dictionary

indexing
#creating seperate dataframes per site - can be used to analyse a particular site if needed later

site_1_freezers = freezers.iloc[:,0:indexing['1']]

site_2_freezers = freezers.iloc[:,indexing['1']:indexing['2']]

site_3_freezers = freezers.iloc[:,indexing['2']:indexing['3']]

site_4_freezers = freezers.iloc[:,indexing['3']:]
site_4_freezers.head()
freezers.head()
#taking all observations at a particular hour and minute of a day

freezers_specific_time = freezers[(freezers.index.hour==2)&(freezers.index.minute==12)]
#observations at 2:12:00 for all the 15 days

freezers_specific_time
#variance associated with the filtered values above

(freezers_specific_time.var())
#variance associated if we take all observations

freezers.var()
#comparing both the variances

comparing_variance = pd.DataFrame({'total variance':freezers.var(),'variance at specific time':freezers_specific_time.var()})
comparing_variance['%decrease'] = ((comparing_variance['total variance']-comparing_variance['variance at specific time'])/comparing_variance['total variance'])*100
#we find that there is considerable decrease in variance with most of the observations

comparing_variance
#finding list of dates where we have null values in a column. Have taken 'Site-3 > Freezer-15' as an example to test

indexes_list = freezers['Site-3 > Freezer-15'].index[freezers['Site-3 > Freezer-15'].apply(np.isnan)]
#number of null values. This should be equal to the length of the indexes_list generated above

freezers['Site-3 > Freezer-15'].isna().sum()
#661 dates are returned which is indeed the number of missing value for that particular column

indexes_list
#filling null values across the dataframe

for col in freezers.columns:

    #list of dates where we have null values for a column 'col'

    indexes_list = freezers[col].index[freezers[col].apply(np.isnan)]    

    #for one date at a time, filling the null value

    for date in indexes_list:

        column_name = col

        obs_hour = date.hour

        obs_min = date.minute

        #list of observations having same hour and minute as the freezer having null value

        df_hour_min = freezers[(freezers.index.hour==obs_hour)&(freezers.index.minute==obs_min)][column_name]

        #finding mean of those observations

        mean_value = df_hour_min.mean()

        #print(mean_value)

        #setting the null value as the mean value

        freezers.ix[date][column_name] = mean_value
freezers.isnull().any()
#Now finding the operational temperature range of each freezer at each site -

print("\n")

print("======Operational Temperature Range======")

print("\n")

for col in freezers.columns:

    print(col,"- [",'\033[1m',freezers[col].min(),":",freezers[col].max(),"\033[0m","]")
#Calculating the daily average of door open duration per freezer

freezers['Site-4 > Freezer-9'].describe()
freezers['Site-1 > Freezer-5'].describe()
import plotly

import cufflinks as cf

from plotly.offline import iplot,init_notebook_mode

init_notebook_mode(connected=True)

cf.go_offline()



plt.figure(figsize=(20,10))

freezers['Site-1 > Freezer-5'].iplot()
#this function is used to calculate the average time for which the freezer door stays open for each time it is opened

#it takes the difference of the timestamp when the temperature rises above threshold and when the temp comes back

#below the threshold. These differences are stored in a list and finally average is taken as the output.

def find_avg_door_opening_time(col,threshold_value):

    time_freezer_opened = []

    c=0  #keeps a track if the past temp value was above threshold or below; if c==1 -> last temp value was above threshold

    j=0 #keeps a count of index

    for value in freezers[col]:

        if ((value>threshold_value) & (c==0)):

            c=1

            start_time_index = j

        if ((value<threshold_value) & (c==1)):

            end_time_index = j

            c=0

            time_difference = ((freezers[col].ix[[end_time_index]].index[0])-

            (freezers[col].ix[[start_time_index]].index[0]))

            time_freezer_opened.append(time_difference.seconds//60)

        j+=1

    average_time_freezer_opened_mean = np.mean(time_freezer_opened) 

    average_time_freezer_opened_median = np.median(time_freezer_opened)

    return average_time_freezer_opened_median,average_time_freezer_opened_mean
#calling the above function for all the columns

mins=[]

hours=[]

col_name=[]

avg_time_mean = []

avg_time_median = []

std_deviation = []

for col in freezers.columns:

    mean_value = freezers[col].mean()

    #max_value = freezers[col].max()

    st_deviation = freezers[col].std()

    #TRY 1 - threshold = mean_value + st_deviation + 5

    #TRY 2 - threshold = mean_value + ((max_value-mean_value)*0.5)

    #finally settled with this threshold value

    threshold = mean_value + st_deviation

    #num_mins calculates the total number of minutes the freezer was opened across 15 days

    num_mins = freezers[freezers[col] > threshold][col].count()

    num_hours = num_mins/60

    #average_open_time_median calculates the median time for which the freezer door stays open for each time it is opened

    average_open_time_median,average_open_time_mean = find_avg_door_opening_time(col,threshold)

    col_name.append(col)

    mins.append(num_mins)

    hours.append(num_hours)

    std_deviation.append(st_deviation)

    avg_time_mean.append(average_open_time_mean)

    avg_time_median.append(average_open_time_median)
freezer_opened = pd.DataFrame({"freezer":col_name,"mins_opened":mins,"hours_opened":hours,"num_mins_door_remains_open(median)":avg_time_median,"num_mins_door_remains_open(mean)":avg_time_mean})
freezer_opened['avg_hours_door_opens_per_day'] = freezer_opened['hours_opened']/15
freezer_opened.set_index('freezer')
freezer_opened = freezer_opened.drop(['mins_opened','hours_opened'],axis=1)
freezer_opened
#1 Site-1 > Freezer-1 average minutes the door stays open ~60 mins

freezers['Site-1 > Freezer-1'].describe()
#threshold for this freezer was set at mean+std i.e. ~10

plt.figure(figsize=(20,10))

freezers['Site-1 > Freezer-1'].iplot()
#2 Site-2 > Freezer-5 average minutes the door stays open ~200 mins

freezers['Site-2 > Freezer-5'].describe()
#threshold for this freezer was set at mean+std i.e. ~18

plt.figure(figsize=(20,10))

freezers['Site-2 > Freezer-5'].iplot()
#3 Site-4 > Freezer-5 average minutes the door stays open ~400 mins

freezers['Site-4 > Freezer-5'].describe()
#threshold for this freezer was set at mean+std i.e. ~21

plt.figure(figsize=(20,10))

freezers['Site-4 > Freezer-5'].iplot()
#1 Site-1 > Freezer-9 --> median 1 vs mean 14

freezers['Site-1 > Freezer-9'].describe()
#threshold for this freezer was set at mean+std i.e. ~9

plt.figure(figsize=(20,10))

freezers['Site-1 > Freezer-9'].iplot()
#2 Site-2 > Freezer-9 --> median 3 vs mean 50

freezers['Site-2 > Freezer-9'].describe()
#threshold for this freezer was set at mean+std i.e. ~10

plt.figure(figsize=(20,10))

freezers['Site-2 > Freezer-9'].iplot()
#3 Site-3 > Freezer-13 --> median 18 vs mean 42

plt.figure(figsize=(20,10))

freezers['Site-3 > Freezer-13'].iplot()
#calculating the deviation of mean value door opening time from median value of door opening time

freezer_opened['%deviation in mean from median'] = ((freezer_opened['num_mins_door_remains_open(mean)']-

                                                     freezer_opened['num_mins_door_remains_open(median)'])

                                                    /freezer_opened['num_mins_door_remains_open(median)'])*100
freezer_opened.head()
#ranking freezer based on their deviation.

#the more the deviation, the more chances of the freezer being malfunctioning. why? - the freezer finds it hard

#to stay in the normal operating range and breaches the threshold value for most of the durations

#this leads to a jump in the mean calculated for the total time for which the door stays open as the criteria set for

#calculating door opening time was to see when the temp crosses the threshold. On the other hand the median stays close

#to the normal operating temperature!

freezer_rankings_malfunctioning = freezer_opened[['freezer','%deviation in mean from median']]

freezer_rankings_malfunctioning = freezer_rankings_malfunctioning.sort_values(by='%deviation in mean from median')

freezer_rankings_malfunctioning = freezer_rankings_malfunctioning.reset_index(drop=True)

freezer_rankings_malfunctioning
#taking into account the door opening time in addition to deviation. The freezers which have low door opening time will be 

#ranked lower while those having high door opening ranks will be ranked poorly.

freezer_opened['%deviation*dooropentime'] = (np.abs(freezer_opened['%deviation in mean from median']))*freezer_opened['num_mins_door_remains_open(median)']
freezer_rankings_dooropening = freezer_opened[['freezer','%deviation*dooropentime']]

freezer_rankings_dooropening = freezer_rankings_dooropening.sort_values(by='%deviation*dooropentime')

freezer_rankings_dooropening = freezer_rankings_dooropening.reset_index(drop=True)

freezer_rankings_dooropening
# creating one comprehensive rank table - 

freezer_rankings_malfunctioning['malfunctioning_rank'] = freezer_rankings_malfunctioning.index+1
freezer_rankings_dooropening['dooropening_rank'] = freezer_rankings_dooropening.index+1
comprehensive_ranking = pd.merge(freezer_rankings_malfunctioning,freezer_rankings_dooropening,on="freezer")
comprehensive_ranking = comprehensive_ranking.drop(['%deviation in mean from median','%deviation*dooropentime'],axis=1)
comprehensive_ranking
#Freezer 5 in Site 4 has a malfunctioning rank of 10 which is good while the door opening rank of 44 i.e. second last

#the plot shows the reason why - the freezer isnt malfunctioning, its just that it remained open for atleast 3 days straight

#between 15th May and 18th May.

freezers['Site-4 > Freezer-5'].iplot()
#The opposite case - 

#Freezer 8 at Site 4 has a good door opening rank of 4 but a rather bad rank at malfunctioning of 28.

#Plotting it shows us that indeed - even though the door isnt opened for longer, the fact that the temperature varies

#so much from 3 degrees to 12 degrees so frequently indicates that the freezer might be having difficulty maintaining the temp.



freezers['Site-4 > Freezer-8'].iplot()
from statsmodels.tsa.seasonal import seasonal_decompose

#freq = 1440 as every 1440minutes(24hrs) the data is bound to repeat

decomposition = seasonal_decompose(freezers['Site-2 > Freezer-2'],freq=1440)



trend = decomposition.trend

seasonal = decomposition.seasonal

residual = decomposition.resid

plt.figure(figsize=(10,6))

plt.subplot(411)

plt.plot(freezers['Site-2 > Freezer-2'], label='Original')

plt.legend(loc='best')

plt.subplot(412)

plt.plot(trend, label='Trend')

plt.legend(loc='best')

plt.subplot(413)

plt.plot(seasonal,label='Seasonality')

plt.legend(loc='best')

plt.subplot(414)

plt.plot(residual, label='Residuals')

plt.legend(loc='best')

plt.tight_layout()
#divide into train and validation set

train = freezers[:int(0.7*(len(freezers['Site-2 > Freezer-2'])))]['Site-2 > Freezer-2']

valid = freezers[int(0.7*(len(freezers['Site-2 > Freezer-2']))):]['Site-2 > Freezer-2']



#plotting the data

plt.figure(figsize=(20,6))

ax = train.plot(label='Train')

valid.plot(ax=ax,label='Validation')

plt.legend()
#building the model without accounting for seasonality

from pmdarima.arima import auto_arima

model = auto_arima(train, trace=True, error_action='ignore', suppress_warnings=True)

model.fit(train)



forecast = model.predict(n_periods=len(valid))

forecast = pd.DataFrame(forecast,index = valid.index,columns=['Prediction'])



#plot the predictions for validation set

plt.figure(figsize=(20,6))

plt.plot(train, label='Train')

plt.plot(valid, label='Valid')

plt.plot(forecast, label='Prediction')

plt.legend(loc='best')

plt.show()
#building the model accounting for seasonality. But was taking too much time to process and hence i commented it out.

"""from pmdarima.arima import auto_arima

model = auto_arima(train, trace=True, error_action='ignore', suppress_warnings=True,seasonal=True,m=1440,D=1)

model.fit(train)



forecast = model.predict(n_periods=len(valid))

forecast = pd.DataFrame(forecast,index = valid.index,columns=['Prediction'])



#plot the predictions for validation set

plt.plot(train, label='Train')

plt.plot(valid, label='Valid')

plt.plot(forecast, label='Prediction')

plt.legend(loc='best')

plt.show()"""