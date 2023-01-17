import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

# Any results you write to the current directory are saved as output.

train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/train.csv')
train.head(5)

#test = pd.read_csv("../input/covid19-global-forecasting-week-3/test.csv")

train = pd.read_csv("../input/covid19-global-forecasting-week-3/train.csv")



print("Number of Country_Region -", train['Country_Region'].nunique())

print("Dates from", min(train['Date']), "to day", max(train['Date']), "- total of", train['Date'].nunique(), "days")

print("Countries with Province: ", train[train['Province_State'].isna()==False]['Country_Region'].unique())



display(train.head(5))
country_df = train[train.Country_Region=='China'].groupby('Date')['ConfirmedCases','Fatalities'].sum()

country_df['day_count'] = list(range(1,len(country_df)+1))

ydata = country_df.ConfirmedCases

xdata = country_df.day_count

country_df['rate'] = (country_df.ConfirmedCases-country_df.ConfirmedCases.shift(1))/country_df.ConfirmedCases

country_df['increase'] = (country_df.ConfirmedCases-country_df.ConfirmedCases.shift(1))



plt.plot(xdata, ydata, 'o')

plt.title("China")

plt.ylabel("Population infected")

plt.xlabel("Days")

plt.show()
from scipy.optimize import curve_fit

import pylab





def sigmoid(x,c,a,b):

     y = c*1 / (1 + np.exp(-a*(x-b)))

     return y

#country_df.ConfirmedCases

#country_df.day_count

xdata = np.array([1, 2, 3,4, 5, 6, 7])

ydata = np.array([0, 0, 13, 35, 75, 89, 91])



#([low_a,low_b],[high_a,high_b])

#low x --> low b

#high y --> high c

#a is the sigmoidal shape.

popt, pcov = curve_fit(sigmoid, xdata, ydata, method='dogbox',bounds=([0.,0., 0.],[100,2, 10.]))

print(popt)



x = np.linspace(-1, 10, 50)

y = sigmoid(x, *popt)



pylab.plot(xdata, ydata, 'o', label='data')

pylab.plot(x,y, label='fit')

pylab.ylim(-0.05, 105)

pylab.legend(loc='best')

pylab.show()
from scipy.optimize import curve_fit

import pylab





def sigmoid(x,c,a,b):

     y = c*1 / (1 + np.exp(-a*(x-b)))

     return y

#country_df.ConfirmedCases

#country_df.day_count

#xdata = np.array([1, 4, 6,8, 10, 12, 14])

#ydata = np.array([0, 0, 1300, 3500, 7500, 8900, 9100])

xdata = np.array(list(country_df.day_count)[::2])

ydata = np.array(list(country_df.ConfirmedCases)[::2])

#([low_a,low_b],[high_a,high_b])

#high x --> high b

#high y --> high c

#a is the sigmoidal shape.

population=1.2*10**9

popt, pcov = curve_fit(sigmoid, xdata, ydata, method='dogbox',bounds=([0.,0., 0.],[population,6, 100.]))



print('model start date:',country_df[country_df.day_count==1].index[0])

print('model start infection:',int(country_df[country_df.day_count==1].ConfirmedCases[0]))

print('model fitted max infection at:',int(popt[0]))

print('model sigmoidal coefficient is:',round(popt[1],2))

print('model curve stop steepening, start flattening by day:',int(popt[2]))

print('model curve flattens by day:',int(popt[2])*2)

print('which is date:',country_df[country_df.day_count==int(popt[2])*2].index[0])

display(country_df.head(3))

display(country_df.tail(3))



x = np.linspace(-1, country_df.day_count.max(), 50)

y = sigmoid(x, *popt)

pylab.plot(xdata, ydata, 'o', label='data')

pylab.plot(x,y, label='fit',alpha = 0.6)

pylab.ylim(-0.05, 100005)

pylab.legend(loc='best')

plt.title('china')

plt.xlabel('days from day 1')

plt.ylabel('confirmed cases')

pylab.show()



from scipy.optimize import curve_fit

import pylab

from datetime import timedelta



us_df = train[train.Country_Region=='US'].groupby('Date')['ConfirmedCases','Fatalities'].sum()

us_df = us_df[us_df.ConfirmedCases>=100]

us_df['day_count'] = list(range(1,len(us_df)+1))

us_df['increase'] = (us_df.ConfirmedCases-us_df.ConfirmedCases.shift(1))

us_df['rate'] = (us_df.ConfirmedCases-us_df.ConfirmedCases.shift(1))/us_df.ConfirmedCases





def sigmoid(x,c,a,b):

     y = c*1 / (1 + np.exp(-a*(x-b)))

     return y

#us_df.ConfirmedCases

#us_df.day_count

#xdata = np.array([1, 4, 6,8, 10, 12, 14])

#ydata = np.array([0, 0, 1300, 3500, 7500, 8900, 9100])

xdata = np.array(list(us_df.day_count)[::2])

ydata = np.array(list(us_df.ConfirmedCases)[::2])

#([low_a,low_b],[high_a,high_b])

#high x --> high b

#high y --> high c

#a is the sigmoidal shape.

population=1.2*10**9

popt, pcov = curve_fit(sigmoid, xdata, ydata, method='dogbox',bounds=([0.,0., 0.],[population,6, 100.]))



est_a = 750000

est_b = 0.18

est_c = 28

x = np.linspace(-1, us_df.day_count.max()+50, 50)

y = sigmoid(x,est_a,est_b,est_c)

pylab.plot(xdata, ydata, 'o', label='data')

pylab.plot(x,y, label='fit',alpha = 0.6)

pylab.ylim(-0.05, est_a*1.05)

pylab.xlim(-0.05, est_c*2.05)

pylab.legend(loc='best')

plt.xlabel('days from day 1')

plt.ylabel('confirmed cases')

plt.title('us')

pylab.show()





print('model start date:',us_df[us_df.day_count==1].index[0])

print('model start infection:',int(us_df[us_df.day_count==1].ConfirmedCases[0]))

print('model fitted max infection at:',int(est_a))

print('model sigmoidal coefficient is:',round(est_b,2))

print('model curve stop steepening, start flattening by day:',int(est_c))

print('model curve flattens by day:',int(est_c)*2)

display(us_df.head(3))

display(us_df.tail(3))

    
from scipy.optimize import curve_fit

import pylab

from datetime import timedelta



italy_df = train[train.Country_Region=='Italy'].groupby('Date')['ConfirmedCases','Fatalities'].sum()

italy_df = italy_df[italy_df.ConfirmedCases>=100]

italy_df['increase'] = (italy_df.ConfirmedCases-italy_df.ConfirmedCases.shift(1))

italy_df['rate'] = (italy_df.ConfirmedCases-italy_df.ConfirmedCases.shift(1))/italy_df.ConfirmedCases

italy_df = italy_df[italy_df.ConfirmedCases>1000]

italy_df['day_count'] = list(range(1,len(italy_df)+1))



def sigmoid(x,c,a,b):

     y = c*1 / (1 + np.exp(-a*(x-b)))

     return y

#italy_df.ConfirmedCases

#italy_df.day_count

#xdata = np.array([1, 4, 6,8, 10, 12, 14])

#ydata = np.array([0, 0, 1300, 3500, 7500, 8900, 9100])

xdata = np.array(list(italy_df.day_count)[::2])

ydata = np.array(list(italy_df.ConfirmedCases)[::2])

#([low_a,low_b],[high_a,high_b])

#high x --> high b

#high y --> high c

#a is the sigmoidal shape.

population=1.2*10**9

popt, pcov = curve_fit(sigmoid, xdata, ydata, method='dogbox',bounds=([0.,0., 0.],[population,6, 100.]))



est_a = 210000

est_b = 0.12

est_c = 32

x = np.linspace(-1, italy_df.day_count.max()+50, 50)

y = sigmoid(x,est_a,est_b,est_c)

pylab.plot(xdata, ydata, 'o', label='data')

pylab.plot(x,y, label='fit',alpha = 0.6)

pylab.ylim(-0.05, est_a*1.05)

pylab.xlim(-0.05, est_c*2.05)

pylab.legend(loc='best')

plt.xlabel('days from day 1')

plt.ylabel('confirmed cases')

plt.title('italy')

pylab.show()





print('model start date:',italy_df[italy_df.day_count==1].index[0])

print('model start infection:',int(italy_df[italy_df.day_count==1].ConfirmedCases[0]))

print('model fitted max infection at:',int(est_a))

print('model sigmoidal coefficient is:',round(est_b,2))

print('model curve stop steepening, start flattening by day:',int(est_c))

print('model curve flattens by day:',int(est_c)*2)

display(italy_df.head(3))

display(italy_df.tail(3))
from scipy.optimize import curve_fit

import pylab

from datetime import timedelta



world_df = train.groupby('Date')['ConfirmedCases','Fatalities'].sum()

world_df = world_df[world_df.ConfirmedCases>=100]

world_df['increase'] = (world_df.ConfirmedCases-world_df.ConfirmedCases.shift(1))

world_df['rate'] = (world_df.ConfirmedCases-world_df.ConfirmedCases.shift(1))/world_df.ConfirmedCases

world_df = world_df[world_df.ConfirmedCases>1000]

world_df['day_count'] = list(range(1,len(world_df)+1))



def sigmoid(x,c,a,b):

     y = c / (1 + np.exp(-a*(x-b)))

     return y

#world_df.ConfirmedCases

#world_df.day_count

#xdata = np.array([1, 4, 6,8, 10, 12, 14])

#ydata = np.array([0, 0, 1300, 3500, 7500, 8900, 9100])

xdata = np.array(list(world_df.day_count)[::2])

ydata = np.array(list(world_df.ConfirmedCases)[::2])

#([low_a,low_b],[high_a,high_b])

#high x --> high b

#high y --> high c

#a is the sigmoidal shape.

population=1.2*10**9

popt, pcov = curve_fit(sigmoid, xdata, ydata, method='dogbox',bounds=([0.,0., 0.],[population,6, 100.]))



est_a = 3400000

est_b = 0.12

est_c = 75

x = np.linspace(-1, world_df.day_count.max()+150, 50)

y = sigmoid(x,est_a,est_b,est_c)

pylab.plot(xdata, ydata, 'o', label='data')

pylab.plot(x,y, label='fit',alpha = 0.6)

pylab.ylim(-0.05, est_a*1.05)

pylab.xlim(-0.05, world_df.day_count.max()+100)

pylab.legend(loc='best')

plt.xlabel('days from day 1')

plt.ylabel('confirmed cases')

plt.title('world')

pylab.show()





print('model start date:',world_df[world_df.day_count==1].index[0])

print('model start infection:',int(world_df[world_df.day_count==1].ConfirmedCases[0]))

print('model fitted max infection at:',int(est_a))

print('model sigmoidal coefficient is:',round(est_b,2))

print('model curve stop steepening, start flattening by day:',int(est_c))

display(world_df.head(3))

display(world_df.tail(3))
us_df = train[train.Country_Region=='US'].groupby('Date')['ConfirmedCases','Fatalities'].sum()

us_df = us_df[us_df.ConfirmedCases>=100]

us_df['day_count'] = list(range(1,len(us_df)+1))

us_df['increase'] = (us_df.ConfirmedCases-us_df.ConfirmedCases.shift(1))

us_df['rate'] = (us_df.ConfirmedCases-us_df.ConfirmedCases.shift(1))/us_df.ConfirmedCases



italy_df = train[train.Country_Region=='Italy'].groupby('Date')['ConfirmedCases','Fatalities'].sum()

italy_df = italy_df[italy_df.ConfirmedCases>=100]

italy_df['day_count'] = list(range(1,len(italy_df)+1))

italy_df['increase'] = (italy_df.ConfirmedCases-italy_df.ConfirmedCases.shift(1))

italy_df['rate'] = (italy_df.ConfirmedCases-italy_df.ConfirmedCases.shift(1))/italy_df.ConfirmedCases



#print(country_df.head(5))

#print(us_df.head(5))



plt.figure(figsize=(24,8))

x = country_df.day_count

y = country_df.increase

x_2 = us_df.day_count

y_2 = us_df.increase

x_3 = italy_df.day_count

y_3 = italy_df.increase

plt.subplot(1,2,1)

plt.plot(x,y,label = 'china')

plt.plot(x_2,y_2,label ='us')

plt.plot(x_3,y_3,label = 'italy')

plt.title('Daily Increase Total')

plt.legend()





x = country_df.day_count

y = country_df.rate

x_2 = us_df.day_count

y_2 = us_df.rate

x_3 = italy_df.day_count

y_3 = italy_df.rate

plt.subplot(1,2,2)

plt.plot(x,y,label = 'china')

plt.plot(x_2,y_2,label ='us')

plt.plot(x_3,y_3,label = 'italy')

plt.title('Daily Increase %')

plt.legend()
#Run only once

print('before eliminating province:',train.shape)

have_state = train[train['Province_State'].isna()==False]['Country_Region'].unique()

temp = train.iloc[[],:]



for i in have_state:

    temp = pd.concat([temp,train[train.Country_Region==i].groupby(['Date','Country_Region'])['ConfirmedCases','Fatalities'].sum().reset_index()],sort=True)



train = train[train['Province_State'].isna()==True]

train = pd.concat([train,temp],sort=True)

print('after eliminating province:',train.shape)
import numpy as np

threshold = 50000



print('Number ofCountries in dataset:',len(train['Country_Region'].unique()))

print('Number of Countries with >',threshold,'confirmed cases:',len(train[train.ConfirmedCases>threshold]['Country_Region'].unique()))

plt.figure(figsize=(10,8))

ax1 = plt.subplot(1,1,1)

for i in train[train.ConfirmedCases>threshold]['Country_Region'].unique():

    y = train[train.Country_Region == i ].groupby(['Date','Country_Region'])['ConfirmedCases','Fatalities'].sum().reset_index().ConfirmedCases

    x = train[train.Country_Region == i ].groupby(['Date','Country_Region'])['ConfirmedCases','Fatalities'].sum().reset_index().Date

    plt.plot(x,y,label=i)

plt.legend(loc='upper left')

start = ax1.get_xlim()[0]

stop = ax1.get_xlim()[1]



ax1.set_xticks(list(np.linspace(start, stop, 10)))

plt.xticks(rotation=90)

#plt.set_xticks(range())