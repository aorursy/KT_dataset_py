import requests

import pandas as pd

import matplotlib.pyplot as plt

import pandasql

import datetime

import numpy as np

%matplotlib inline
#I'm using simfin to gather Chipotle (CMG) data as they have a pretty user friendly API

#You can create an free API key and do it yourself here: https://simfin.com/data/api

#since kaggle won't allow requests I created the data on my computer and uploaded it here

#the Quarter column was created by me locally



revenues = pd.read_csv("/kaggle/input/cmg-revenues/cmg.csv")

revenues.rename(columns={'Unnamed: 0':'Quarter'},inplace=True)

#let's just view the revnues quickly

fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

ax.plot(revenues.CMG)

ax.set_xlabel('Quarter')

ax.set_ylabel('Revenue')

label_range = list(range(0,len(revenues.CMG),2))

ax.set_xticks(label_range)

ax.set_xticklabels(revenues.Quarter[label_range])

plt.xticks(rotation=45)

plt.show()
#now to read in the google trends data

#taken from 'chipotle mexican grill' search over the last 5 years from today (weekly data)

google = pd.read_csv("/kaggle/input/google-cmg/google CMG.csv")

#have to build quarters into the google data for a merge



years = range(2012, 2021) #remember not inclusive

fiscal_quarters = pd.DataFrame(data=None,columns=['QS_date','QE_date','Quarter'])



for y in years:

    #Q1

    fiscal_quarters = fiscal_quarters.append( {'QS_date':datetime.datetime(year=y,month=1,day=1), 'QE_date':datetime.datetime(year=y,month=3,day=31), 'Quarter': 'Q1_' + str(y)[-2:] },ignore_index=True)

    #Q2

    fiscal_quarters = fiscal_quarters.append( {'QS_date':datetime.datetime(year=y,month=4,day=1), 'QE_date':datetime.datetime(year=y,month=6,day=30), 'Quarter': 'Q2_' + str(y)[-2:] },ignore_index=True)

    #Q3

    fiscal_quarters = fiscal_quarters.append( {'QS_date':datetime.datetime(year=y,month=7,day=1), 'QE_date':datetime.datetime(year=y,month=9,day=30), 'Quarter': 'Q3_' + str(y)[-2:]},ignore_index=True)

    #Q4

    fiscal_quarters = fiscal_quarters.append( {'QS_date':datetime.datetime(year=y,month=10,day=1), 'QE_date':datetime.datetime(year=y,month=12,day=31), 'Quarter': 'Q4_' + str(y)[-2:] },ignore_index=True)





#now join to revenues

sql_code = '''

            select a.*,avg(b.chipotle) as goog_avg, c.CMG

            from fiscal_quarters as a 

                 left join 

                 google as b

            on a.QS_date <= b.week and b.week <= QE_date

                left join

                revenues as c

            on a.Quarter = c.Quarter

            where b.isPartial is False

            group by a.QE_date

            order by QE_date

'''

combined_data = pandasql.sqldf(sql_code,locals())

#for data exploration, graphing and analysis let's remove NaN rows 

#i.e. cut off if any revenues have not been reported

#note this doesn't take into account if there's a missing historical number

#that will need to be built in a future version



#we'll need Q/Q and Y/Y time series later so create them now

combined_data['qq_goog'] = (combined_data.goog_avg/combined_data.goog_avg.shift(1)-1)*100

combined_data['yy_goog'] = (combined_data.goog_avg/combined_data.goog_avg.shift(4)-1)*100



combined_data['qq_cmg'] = (combined_data.CMG/combined_data.CMG.shift(1)-1)*100

combined_data['yy_cmg'] = (combined_data.CMG/combined_data.CMG.shift(4)-1)*100



cmg_x_forecast = combined_data.copy()



combined_data.dropna(how='any',subset=['CMG'],inplace=True)
#now let's start to have some fun!

#First let's just look at a basic scatter plot of revenue vs. aggregated google trends data

#quick scatter plot to see if there's a linear relationship

plt.scatter(x=combined_data.goog_avg,y=combined_data.CMG)

combined_data.corr()



#considering there's a .61 correlation from just google trends data it's worth exploring a little more

#(plus this is just a fun excercise)

#so now let's view the time series utilizing left and right y axes

fig, ax1 = plt.subplots()

ax1.set_xlabel('Quarter')

ax1.set_ylabel('Revenue',color="tab:red")

ax1.plot(combined_data.CMG, color="tab:red")

ax1.tick_params(axis='y', labelcolor="tab:red")

#plt.xticks(range(0,10),all_data.Quarter[0:9])



ax2 = ax1.twinx()

ax2.set_ylabel('Google Trend', color="tab:blue")

ax2.plot(combined_data.goog_avg,color="tab:blue")

ax2.tick_params(axis='y', labelcolor="tab:blue")



_label_range = list(range(0,len(combined_data.Quarter)+1,2))

_labels = combined_data.Quarter[_label_range]

ax1.set_xticks(_label_range)

ax1.set_xticklabels(_labels)

ax1.tick_params(rotation=45)
#now you can see the .61 correlation visually - it's not great but, again, this is for the fun of the excercise

#on an absolute basis (revenue vs. google trends) it doesn't give a sense of forecasting ability

#but let's create some yearly and quartely time series that might be more useful



plt.plot(combined_data.qq_goog,color="tab:red",label="Google Q/Q")

plt.plot(combined_data.qq_cmg,color="tab:blue",label="CMG Q/Q")

plt.xlabel("Quarter")

plt.ylabel("Q/Q Change (%)")

plt.title("Google trends CMG vs. Reported CMG Revenues on Q/Q delta")

plt.legend()

plt.show()

print(np.corrcoef(combined_data.qq_cmg[1:],combined_data.qq_goog[1:])) #.36 correlation.. wamp wamp
#well the q/q time series isn't great but let's consider yearly (y/y) series

#we're just going to repeat the same plot exploration but using y/y series



plt.plot(combined_data.yy_goog,color="tab:red",label="Google Y/Y")

plt.plot(combined_data.yy_cmg,color="tab:blue",label="CMG Y/Y")

plt.xlabel("Quarter")

plt.ylabel("Y/Y Change (%)")

plt.title("Google trends CMG vs. \nReported CMG Revenues on Y/Y delta")

plt.legend()

plt.show()

print(np.corrcoef(combined_data.yy_cmg[4:],combined_data.yy_goog[4:])) #.31 correlation.. wamp wamp
#well it doesn't look like either time series will be a great fit but this is a fun excercise after all

#let's explore how a regression model would work anyway

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression

from sklearn import metrics
#now we can actually do the regression!

#remember to start at position 4 due to Y/Y trends

X = combined_data.yy_goog[4:].values.reshape(-1,1)

y = combined_data.yy_cmg[4:].values.reshape(-1,1)



#we'll do a test_size of 20% which in this case is only 3 data points (I know - not ideal)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()

regressor.fit(X_train, y_train)
#now to run the predictor and gather results

y_pred = regressor.predict(X_test)

predicted = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})



#we can look at a bar chart to visualize differences of predicted vs. actual. Note the quarters are lost here since it's random

predicted.plot(kind='bar',figsize=(10,5))

plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')

plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

plt.show()



#further we can also view how the regression fit on the test set (i.e. the 3 data points)

plt.scatter(X_test, y_test,  color='gray')

plt.plot(X_test, y_pred, color='red', linewidth=2)

plt.show()
#Lastly, let's just forecast Chipotle's Q2 '20 revenue anyway just for fun (yes, the results will probably be bad)

print(regressor.coef_*cmg_x_forecast[ cmg_x_forecast['Quarter']=='Q2_20']['yy_goog'].values + regressor.intercept_)



#the results show +12% y/y in the midst of COVID-19 which would be a very impressive quarter



#Edit: Chipotle reported Q2 '20 revenue of -9.8% Y/Y so clearly this did not work!  Haha.  Fun excercise nevertheless!