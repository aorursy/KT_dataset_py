import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.dates as mdates

from sklearn.metrics import mean_squared_error, r2_score

import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv('/kaggle/input/covid19-india-cases-marchmay-2020/data_16June.csv')

d = data.values

dates = d[:,0]

days = np.array(d[:,1],dtype='int16')

cummulative_cases = np.array(d[:,2], dtype='float64')

cummulative_deaths = np.array(d[:,3], dtype='float64')

cummulative_recovered = np.array(d[:,4], dtype='float64')
def daily_data (y):

  daily = [None]*len(y)

  daily[0] = y[0]

  for i in range(1,len(y)):

    daily[i] = y[i]-y[i-1]

  return np.array(daily)



daily_new_cases = daily_data(cummulative_cases)

daily_new_deaths = daily_data(cummulative_deaths)

daily_new_recovered = daily_data(cummulative_recovered)
def proportion (x,y):

  prop = [None]*len(y)

  for i in range(len(y)):

    prop[i] = (y[i]/x[i])*100

  return np.array(prop)



prop_death = proportion (cummulative_cases, cummulative_deaths)

prop_recovered = proportion (cummulative_cases, cummulative_recovered)
def fit (x,y):

  for i in range (0,101,5):

    f = np.polyfit(x,y,deg=i)

    fval = np.polyval(f,x)

    print('Degree = %d \tMSE = %10.2f \t R^2 Score = %10.6f' %(i,mean_squared_error(y,fval),r2_score(y,fval)))
def my_plot(x,y,dates,n):

  f = np.polyfit(x,y,deg=n)

  fval = np.polyval(f,x)



  date_list = []

  pos = []

  for i in range(len(dates)):

    if i%5 == 0:

        date_list.append(str(dates[i]).split()[0])

        pos.append(i)



  plt.plot(y,'ro',markersize=2)

  plt.plot(fval,'g',linewidth=1)

  plt.xticks(ticks=pos, rotation='vertical',labels=date_list)

  plt.grid(which = 'both',axis='both')

  plt.text(days[-1],y[-1],str(int(y[-1])))

  plt.ylabel('Number of Cases')

  plt.legend(['Actual Data (https://www.mohfw.gov.in/)','Fitted curve'])

  if n == 1:

    print('\nFitted curve for degree %d is Y = %fx + %f\n' %(n,f[0],f[1]))

  elif n == 2:

    print('\nFitted curve for degree %d is Y = %fx^2 + %fx + %f\n' %(n,f[0],f[1],f[2]))

  elif n == 3:

    print('\nFitted curve for degree %d is Y = %fx^3 + %fx^2 + %fx + %f\n' %(n,f[0],f[1],f[2],f[3]))

  elif n == 4:

    print('\nFitted curve for degree %d is Y = %fx^4 + %fx^3 + %fx^2 + %fx + %f\n' %(n,f[0],f[1],f[2],f[3],f[4]))

  elif n == 5:

    print('\nFitted curve for degree %d is Y = %fx^5 + %fx^4 + %fx^3 + %fx^2 + %fx + %f\n' %(n,f[0],f[1],f[2],f[3],f[4],f[5]))

  else:

    pass
fit (days, cummulative_cases)

my_plot(days, cummulative_cases, dates, 5)

plt.title('Cummulative cases of COVID-19 cases in India Mar-May 2020')

plt.show()
fit (days, cummulative_deaths)

my_plot(days, cummulative_deaths, dates, 5)

plt.title('Cummulative Deaths of COVID-19 in India Mar-May 2020')

plt.show()
fit (days, cummulative_recovered)

my_plot(days, cummulative_recovered, dates, 5)

plt.title('Cummulative recovered cases of COVID-19 in India Mar-May 2020')

plt.show()
fit (days, daily_new_cases)

my_plot (days, daily_new_cases, dates, 5)

plt.title('Daily new cases of COVID-19 in India Mar-May 2020')

plt.show()
fit (days, daily_new_deaths)

my_plot (days, daily_new_deaths, dates, 5)

plt.title('Daily new deaths of COVID-19 in India Mar-May 2020')

plt.show()
fit (days, daily_new_recovered)

my_plot (days, daily_new_recovered, dates, 5)

plt.title('Daily new recovered cases of COVID-19 in India Mar-May 2020')

plt.show()
date_list = []

pos = []

for i in range(len(dates)):

  if i%5==0:

      date_list.append(str(dates[i]).split()[0])

      pos.append(i)



plt.plot(prop_death, 'r')

plt.text(len(prop_death)-6,prop_death[-1],str(round(prop_death[-1],2)))

plt.plot(prop_recovered, 'g')

plt.text(len(prop_recovered)-6,prop_recovered[-1],str(round(prop_recovered[-1],2)))

plt.legend(['Proportion of Death','Proportion of Recovered'])

plt.ylabel('Proportion (%)')

plt.grid(which='both',axis='both')

plt.ylim([0,100])

plt.title('Proportion of deaths and recovered cases COVID-19 in India Mar-May 2020')

plt.xticks(ticks=pos, rotation='vertical',labels=date_list)

plt.show()