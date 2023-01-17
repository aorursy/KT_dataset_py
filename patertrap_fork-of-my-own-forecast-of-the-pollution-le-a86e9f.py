#THIS IS TO INSTALL THE WORKING VERSION OF FBPROPHET. WITH THE LATEST ONE, THERE IS AN ISSUE WITH NUMPY AND OTHER LIBRARIES

#THE ONES TO USE ARE:

#pyparsing==2.3.0

#tqdm==4.29.0

#numpy==1.15.4



#ALL EXPLAINED IN: https://github.com/facebook/prophet/issues/808



!pip3 uninstall --yes fbprophet

!pip3 install fbprophet --no-cache-dir --no-binary :all:
#%matplotlib  inline



import numpy as np 

import pandas as pd

import math

import glob

import folium #For maps



import matplotlib.pyplot as plt

import seaborn as sns #This is for cool plots. Recommendation from Linda



import missingno as missing #This is for missing data

from fbprophet import Prophet

from scipy.special import boxcox, inv_boxcox



from datetime import datetime, timedelta



#This is my excuse to learn the basics of time series

from statsmodels.tsa.arima_model import ARIMA

from statsmodels.tsa.statespace.sarimax import SARIMAX

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from statsmodels.tsa.stattools import adfuller

from statsmodels.tsa.seasonal import seasonal_decompose

from scipy import stats

import statsmodels.api as sm

from itertools import product

from math import sqrt

from sklearn.metrics import mean_squared_error



print("Everything imported correctly")
#I'm not used to this system. In kaggle use as path: "../input/filename.csv"

#AVOID the name of the folder.



#The stations file I know it because it's the only one I can open with Excel. The other ones are too big

stations=pd.read_csv("../input/stations.csv")

#stations.head()



#To read the ones from the years, use a loop with glob. Great invention.



path_years="../input/csvs_per_year/csvs_per_year/" #For simplicity later

all_files=glob.glob(path_years+"/*.csv") #DON'T FORGET THE /!!!!!



#Now I allocate the data frames and lists and everything I need

frame=pd.DataFrame()

ls=[] #List is a reserved word in python, I can't name my list as "list" so I put ls



#I read the files now. For every file in the files directory with a similar name, read them and append them to my data frame



#For memory, header=0 allows me to change the columns names later on.

#index_col=False will evaluate as 0. Don't do it, it's ambiguous.

for file_1 in all_files:

    df=pd.read_csv(file_1,index_col=None, header=0)

    ls.append(df)

#Concat places the second df below the first one and so on. It rewrites the index as I would normally do    

frame=pd.concat(ls)

#Have a look to see what I got

frame.head()

#Comment the head to avoid many lines in the output

#Looks ok so far, with a lot of NaNs. Let's worry about it in a couple of cells

#I could print all, but just 5 will do. I have the addresses, which I understand, but the maps use lat/lon

stations.head()



#Select the lat and lon columns to have the positions and pass them to folium.



#Inner bracket is for the list, outer bracket is for indexing

locations=stations[['lat','lon']]

location_list=locations.values.tolist() #Take the locations, with the values, make a list

#Check the difference when printing locations and location_list locations only prints ['lat','lon']

#location_list prints the values



popup=stations['name']



#Again, map is a reserved word

#Make an empty map, with the zoom and the centering

map_stations=folium.Map(location=[40.44, -3.7], zoom_start=11)



#For every point, add a marker. The points start at 0 indexing and finish at the end of the list

# so, put a marker at every location[point], which is already containing stations['lat'],stations['lon']

for point in range(0,len(location_list)):

    folium.Marker(location_list[point], popup=popup.iloc[point]).add_to(map_stations)

#Now show the maps

map_stations

#El carte inglés?? hahhaha

#Don't forget the ; !!! Otherwise it only gives some memory address or something.

missing.matrix(frame);

#This is slow, so try not to run it too much. White blocks are missing data (NaN) and black is some data.

#Black lines/blocks may also be incorrect. Careful with that. The number on the bottom left is the amount 
#This visualizes the amount of non-nulls. Black is data, you can see it in the date and location column

missing.bar(frame);
#In my case this is useless but it's still very cool. It shows how the nullity of one variable is correlated with other variables

# That is, shows if the absence of a variable is usually correlated with the absence or presence of another one

#Useful? Yes, probably. Cool? Yes!

#Commented for speed

#missing.heatmap(frame);
#frame.head() to check the name of the columns. I probably forget again along the way



#Take only the relevant part of the frame dataframe

cols=['date','station','O_3']

o3=frame[cols]



#Convert to ppb and put date and time in something that python understands and ignore the warnings.

o3['date']=pd.to_datetime(o3['date'])

o3.loc[:,'ppb']=24.45*o3.loc[:,'O_3']/48

o3.head()

#Let's see how many stations were active along time. I remember they remove some when the EU was complaining.





plt.rcParams["figure.figsize"] = [16,9] #For good looking plots in modern screens

plt.plot(o3.groupby(['date']).station.nunique());

plt.ylabel('Number of stations operating');

plt.xlabel('Year');
#Sort the df by number of NaNs, print and then I'll select what I want.

#First, see the shape of the df, to see how many rows we have in total. It's about 3.8e5



o3.shape #Don't take the frame variable, o3 is the same but much smaller



#Now group them by most real values and then sort them



#Careful with this! from left to right, grouped_df is the df made by taking the o3 df. Then you group by station. And then, for every station, you count all the O_3 data

#NaN are ignored.



grouped_stations=o3.groupby('station').O_3.count()

#grouped_stations.head()



#If I take grouped_stations and sort, there is one column with the counts, not the O_3 name. Therefore, sort_values ONLY TAKES THE ARGUMENTS ascending, NOT THE COLUMN (AXIS)

sorted_stations=grouped_stations.sort_values(ascending=False)

sorted_stations.head()





#And then figure out which one is the best station. The column is not named station, but id

#I will print the row that matches id to my best station.

stations[stations.id == sorted_stations.index[0]]

      

      

o3_PN=o3[o3.station==sorted_stations.index[0]]

#o3_PN.head() #To check that it is ok



#But I need to work in ppb, remember that. And I said I would use 8 h moving averages.



#Create a new column, which is the ppb, taking the 8 values window and calculate the mean at each time.

#Obviously, sort them by date/hour, otherwise the rolling average is meaningless

o3_PN=o3_PN.sort_values(['date'])

o3_PN.loc[:,'ppb_moving']=o3_PN.loc[:,'ppb'].rolling(8).mean()

#o3_PN.head(15) #The first values are NaN obviously.



o3_PN=o3_PN.sort_values(['date'])

#o3_PN.head(15)

#I can't fill the NaNs with 0, otherwise this would introduce fake data. I'll fill them with the average of the column



o3_PN['ppb_moving']=o3_PN['ppb_moving'].fillna(o3_PN['ppb_moving'].mean()/2)

o3_PN.head(15)

y=o3_PN['ppb_moving']

y=y.reset_index(drop=True) #It drops one column, saving memory

#y=y.drop(columns=['index'], axis=1)



x=np.linspace(0,len(y),num=len(y))

#len(x)

#len(y)





#Plot with sns. First time



#ax=sns.scatterplot(x=y.index,y=y)

#ax.axes.set_xlim(y.index.min(),y.index.max());



#I still like matplotlib better. Probably because it's almost matlab



#plt.scatter(o3_PN.index,y);



#Since there are too many points, I'll only plot a few. Maybe between 10e3 and 15e3



plt.scatter(x,y);

plt.xlim(0, len(y));



#I use the 'ppb_moving' average column. I deal with 8h periods. And I copy the interesting part of the df to start fresh



data_rel=o3_PN[['date','ppb']] #NOTICE THE [[]] it's a list of columns, which go between []!!



#As I can see above, the data by average 8h is a lot, and I assigned it to many hours, more or less how I wanted to. It's better if I take everything, resample to days and th

#I will take the maximum of every day and that's it. It may be a conservative analysis, but my goal is not to learn about O3 and pollution, but to learn to play with time series



#CAREFUL TO CONVERT THE DATE COLUMN TO A DATE FORMAT, NOW IT'S INT!!



data_rel.loc[:,'ds']=pd.to_datetime(data_rel.loc[:,'date'])

data_rel.set_index('ds', inplace=True)

data_rel.fillna(method='bfill')

data_rel=data_rel.resample('D', how='max')

#data_rel.head() #Looks good to me



len(data_rel['ppb']) #Length is 6330. I should use 4-5k to train and the rest to validate more or less. Or I will validate with 2016 and the following. This gives me 2.5 years for validation



#As I said, I can't use the train_test_split function here because it's not random. I split manually.



#365 days*2.5 years (approx, it's a bit more)



cut=math.floor(3*365)



train=data_rel[1:-cut]

test=data_rel[-cut:]



#NOW I ACCOMODATE ALL TO THE FBPROPHET STYLE. THYE WANT 2 COLUMNS, 'ds' and 'y'. OTHERWISE IT CRASHES. ALL EXPLAINED BELOW IN A COUPLE OF CELLS



train.drop('date', axis=1, inplace=True)

train.dropna()

train.reset_index(inplace=True)

test.drop('date', axis=1, inplace=True)

test.dropna()

test.reset_index(inplace=True)

train.head()



#For the SARIMA part, create the copy now

train_2=train

test_2=test



#And then let's visualize. This is my favourite way to check the splitting, and I can use Matlab colours. Or if I ommit c='whatever' I get blue and orange, a bit ugly in my opinion



#plt.scatter(train.index,train.ppb,c='b'); #train

#plt.scatter(test.index,test.ppb,c='r'); #test



#Looks nice and the train/test ratio is around 85%. Should be fine.



#train.ppb=np.log1p(train.ppb)

#train.info()

plt.scatter(train.index,train.ppb);



#I didn't know that, but Prophet requires that the columns are called 'ds' and 'y'. I have to drop the one called date now because I don't want to change all the code above.



train.columns=['ds','y']

test.columns=['ds','y']



#Now I create the prophet object and train it. I'll do a few with different settings to see how it changes



#Daily seasonality=False

m = Prophet();

m.fit(train);



#Daily seasonality=True

#m_d = Prophet(daily_seasonality=True);

#m_d.fit(train);



#And now predict, again, two predictions



per=len(test.ds)

future=m.make_future_dataframe(periods=per, freq='D')

future.head()

forecast = m.predict(future)



#Although the names are almost self explanatory, ds are the dates, yhat is the predicted value and yhat_lower/upper are the top/bottom of the interval



forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

#future_d=m_d.make_future_dataframe(periods=cut, freq='D')

#forecast.tail()



#I should try and add more seasonality later, now plot. The included plotter is very nice.



from  fbprophet.plot import add_changepoints_to_plot

fig = m.plot(forecast)

a = add_changepoints_to_plot(fig.gca(), m, forecast) #I add the changepoints prophet found
#Now plot the components (Known as the meteoswiss style plot). The components are the Trend, the weekly seasonality and the yearly seasonality (Check above)

fig2 = m.plot_components(forecast);
#Now I plot together the test values and the yhat. To check visually if my model makes some sense. I'll calculate the error metrics later, but first I need to check that my plot looks reasonable.



tester=pd.concat([test.set_index('ds'),forecast.set_index('ds')],axis=1,join='inner')

#join='inner' overlaps the indexes, so since I have 2 indexes that are identical, only one is kept.

tester.head()



fig=plt.subplot()

plt.plot(tester.y);

plt.plot(tester.yhat);

plt.legend()
#Now I'll calculate the RMSE to see how good my model is and compare with different parameters of prophet (And with other models).



tester['e']=tester.y-tester.yhat #Can't use the .e notation to create a new column

rmse_default=np.round(np.sqrt(np.mean(tester.e**2)),4)

mape_default=np.round((np.mean(np.abs(100*tester.e/tester.y))),4)

              

#print(rmse_default)

#print(mape_default)

print("RMSE = ",rmse_default)

print("MAPE = ",mape_default)
m_c = Prophet(changepoint_prior_scale=0.1) #Looks like this doesn't change much the result

m_c.fit(train);



future_c=m_c.make_future_dataframe(periods=per, freq='D')

future_c.head()

forecast_c = m_c.predict(future)

fig_c = m_c.plot(forecast)

a_c = add_changepoints_to_plot(fig_c.gca(), m_c, forecast_c) #I add the changepoints prophet found





fig_c=m_c.plot_components(forecast_c)

#Looks very similar as before
tester_c=pd.concat([test.set_index('ds'),forecast_c.set_index('ds')],axis=1,join='inner')



fig_c=plt.subplot()

plt.plot(tester_c.y);

plt.plot(tester_c.yhat);

plt.legend()



tester_c['e']=tester_c.y-tester_c.yhat #Can't use the .e notation to create a new column

rmse_c=np.round(np.sqrt(np.mean(tester_c.e**2)),4)

mape_c=np.round((np.mean(np.abs(100*tester_c.e/tester_c.y))),4)

              

#print(rmse_default)

#print(mape_default)

print("RMSE = ",rmse_c)

print("MAPE = ",mape_c)

m_d = Prophet(daily_seasonality=True) #Looks like this doesn't change much the result.

m_d.fit(train);



future_d=m_d.make_future_dataframe(periods=per, freq='D')

future_d.head()

forecast_d = m_d.predict(future_d)

fig_d = m_d.plot(forecast_d)

a_d = add_changepoints_to_plot(fig_d.gca(), m_d, forecast_d) #I add the changepoints prophet found
#fig_d = m_d.plot(forecast_d)

#a_d = add_changepoints_to_plot(fig_d.gca(), m_d, forecast_d) #I add the changepoints prophet found

fig_d=m_d.plot_components(forecast_d)

#Looks very similar as before
tester_d=pd.concat([test.set_index('ds'),forecast_d.set_index('ds')],axis=1,join='inner')



fig_d=plt.subplot()

plt.plot(tester_d.y);

plt.plot(tester_d.yhat);

plt.legend()



tester_d['e']=tester_d.y-tester_d.yhat #Can't use the .e notation to create a new column

rmse_d=np.round(np.sqrt(np.mean(tester_d.e**2)),4)

mape_d=np.round((np.mean(np.abs(100*tester_d.e/tester_d.y))),4)

              

print("RMSE_default = ", rmse_default)

print("MAPE_default = ", mape_default)

print("RMSE = ",rmse_c)

print("MAPE = ",mape_c)

print("RMSE_d = ",rmse_d)

print("MAPE_d = ",mape_d)
#Let's bring back the train/test sets. For ease of use.



cols=['ds','y']

train_2.columns=[cols]

test_2.columns=[cols]

a=train_2.isna().sum()

print(a)



#Apparently there are 5 NaN values. Let's drop them, I have a lot of them.



train_2=train_2.dropna()

test_2=test_2.dropna()

a=train_2.isna().sum()

print(a)



#freq=365 because I have year seasonality, if I had daily or montly or weekly I would put 1, 30 or 7

#https://www.statsmodels.org/dev/generated/statsmodels.tsa.seasonal.seasonal_decompose.html

seasonal_decompose(train_2.y, model='additive', freq=365).plot();

p=train_2.iloc[:,1]

#print(p)

print("Dickey–Fuller test: p=%f" % adfuller(p)[1])

print("adf value is: adf=%f" % adfuller(p)[0])



ax=plt.subplot(221)

plot_acf(train_2.y[0:].values.squeeze(), lags=750, ax=ax)



ax=plt.subplot(212)

plot_pacf(train_2.y[0:].values.squeeze(), lags=750, ax=ax)

plt.tight_layout()
p_s=range(0,3)

d=1

q_s=range(0,3)



parameters=product(p_s,q_s) #It's like a for loop, but more efficient.

parameters_ls=list(parameters)

print(parameters_ls)



%%time



#%%time measures the time of hte cell. %time measures only a line

#Let's select a model



results=[]

best_aic=float("inf") #float("inf") may be more robust

for param in parameters_ls:

    try:

        model=SARIMAX(train_2.y, order=(param[0],d,param[1]),enforce_invertibility=False).fit(disp=-1)

    except ValueError:

        print("Bad parameter combination: ",param)

        continue

    aic=model.aic

    if aic<best_aic:

        best_model=model

        best_aic=aic

        best_param=param

    results.append([param, model.aic])

    

    

    
result_table=pd.DataFrame(results)

result_table.columns=['Parameters','AIC']

result_table.sort_values(by='AIC', ascending=True)
print(best_model.summary())
#Now dickey-fuller it to see how this is performing. AIC looks huge



difull_res=adfuller(best_model.resid[13:])[0]

difull_p=adfuller(best_model.resid[13:])[1]

print("P-value is: ",difull_p)

print("Result is: ", difull_res)
best_model.plot_diagnostics()

plt.show()
#check=train_2.isnull().any()

#check

#print(best_model.index)

#future_bs=best_model.forecast(steps=cut)

yhat=best_model.forecast(len(test_2.y))

yhat.tail() #This checks that it looks OK

type(yhat)

test_2['yhat_ARIMA']=yhat.values

test_2.tail()
fig_AR=plt.subplot()

plt.plot(test_2.y);

plt.plot(test_2.yhat_ARIMA);

plt.legend()



#I'm just predicting the average!! I'll leave this here for the moment and finish it later