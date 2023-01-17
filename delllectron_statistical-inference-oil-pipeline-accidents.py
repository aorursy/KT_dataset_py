import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import math



plt.style.use('_classic_test_patch')

sns.set_style('whitegrid')
data = pd.read_csv('../input/pipeline-accidents/database.csv')

cols_of_interest = ['Accident Date/Time','Accident State','Pipeline Location','Liquid Type','Net Loss (Barrels)','All Costs']

data = data[cols_of_interest]
#show dataframe

data.head()
data[['Net Loss (Barrels)', 'All Costs']].describe()
#minimize the staandard deviation by dividing it to 1000000

data['All Costs'] = data['All Costs'] / 1000000

#convert data to pandas datetime object

data['Accident Date/Time'] = pd.to_datetime(data['Accident Date/Time'])
fig, ax = plt.subplots(1,2,figsize=(15,5))



sns.boxplot(data['All Costs'], ax=ax[0]);

sns.boxplot(data['Net Loss (Barrels)'], ax=ax[1]);



ax[0].set_title('Accident Costs in Million USD');

ax[1].set_title('Net Loss (Barrels)');
totalTimespan = data['Accident Date/Time'].max() - data['Accident Date/Time'].min()

totaltime_hour = (totalTimespan.days*24 + totalTimespan.seconds/(3600))

totaltime_month = (totalTimespan.days + totalTimespan.seconds/(3600*24)) *12/365



lmda_h = len(data) / totaltime_hour

lmda_d = len(data) / totalTimespan.days

lmda_m = len(data) / totaltime_month



print('Number of Accidents per Hour: ', lmda_h);

print('Number of Accidents per Day: ', lmda_d);

print('Number of Accidents per Month: ', lmda_m);
PX = {}

#calculate poisson probabilities from range 0-66

for x in range(66):

    PX[x] = math.pow(2.71828, -33) * math.pow(33, x) / math.factorial(x)

p_poisson = pd.DataFrame(PX.items(), columns=['X', 'PX'])





#commulative distribution function

def cdf(data):

    n = len(data)

    x = np.sort(data)

    y = np.arange(1, n+1) / n

    return x,y



np.random.seed(101)

samples_poisson = np.random.poisson(33, 5000)

x,y = cdf(samples_poisson)
#show probability mass function

fig, ax = plt.subplots(1,2, figsize=(15,5))

plt.tight_layout(3)

ax[0].set_title('PROBABILITY MASS FUNCTION', fontsize=18);

ax[0].set_xlabel('X', fontsize=14);

ax[0].set_ylabel('PX', fontsize=14);

ax[0].plot(p_poisson.X, p_poisson.PX, lw=2, marker='.', color='salmon');



#show commulative distribution function

plt.figure(figsize=(10,5))

ax[1].set_title('COMMULATIVE DISTRIBUTION FUNCTION', fontsize=18);

ax[1].set_xlabel('NUMBER OF ACCIDENTS PER MONTH', fontsize=14)

ax[1].set_ylabel('PX', fontsize=14);

ax[1].plot(x, y, lw=2, marker='.', color='steelblue');
data.sort_values(by=['Accident Date/Time'], ascending = True, inplace = True)

data['timetoAccident'] = data['Accident Date/Time'].diff() #compute time between each accident

data['timetoAccident_h'] = data.apply(lambda x: x['timetoAccident'].days * 24 + x['timetoAccident'].seconds/3600, axis = 1)

data= data[data.timetoAccident.notnull()]



mean_time = np.mean(data['timetoAccident_h']) #parameter for exponential distribution

print('Mean time between accidents: ', mean_time)
x,y = cdf(data['timetoAccident_h'])

#get 5000 exponential smaples

samples_exp = np.random.exponential(mean_time, 5000)

x_exp, y_exp = cdf(samples_exp)
plt.figure(figsize=(10,5))



plt.plot(x, y);

plt.plot(x_exp, y_exp);

plt.title('COMMULATIVE DISTRIBUTION FUCTION');

plt.xlabel('Time Between Accidents (Hour)');

plt.ylabel('PX');



plt.legend(['Actual sample', 'Exponential Dist.']);
plt.figure(figsize=(10,5))

plt.title('PROBABILITY DENSITY FUNCTION');

plt.xlabel('Time Between Accidents (Hour)');

sns.distplot(samples_exp, bins=40, color='steelblue');
plt.figure(figsize = (10,5))

plt.xlim([0,100])



for i in range(100):

    samples_bs = np.random.choice(data['timetoAccident_h'], size = len(data['timetoAccident_h']))

    x_bs,y_bs = cdf(samples_bs)

    plt.plot(x_bs,y_bs, lw=1, color='salmon')



x,y = cdf(data['timetoAccident_h'])

plt.plot(x,y, lw=2, color='green',)

plt.title('COMMULATIVE DISTRIBUTION FUNCTION')

plt.xlabel('Time Between Accidents (Hour)');

plt.ylabel('PX');

plt.legend(['BOOTSTRAP SAMPLES']);
def bs_replicate(data, func):

    samples_bs = np.random.choice(data, size=len(data))

    return func(samples_bs)

def draw_bs_replicate(data, func, size):

    replicates = np.empty(size)

    

    for i in range(size):

        replicates[i] = bs_replicate(data, func)

    return replicates
bootstrap_replicates = draw_bs_replicate(data['timetoAccident_h'], np.mean, 5000)

plt.figure(figsize=(10,5))

plt.title('BOOTSTRAP REPLICATES')

plt.xlabel('TIME BETWEEN ACCIDENTS IN HOURS')

sns.distplot(bootstrap_replicates, bins=40, color='green');
confidence_int = np.percentile(bootstrap_replicates, [2.5,97.5])

print('95% confidence interval: ', confidence_int[0], '-',confidence_int[1], '(Hours)')
accidents_pyear = [round((confidence_int[0]/24)*365), round((confidence_int[1]/24)*365)]

print('Number of Accidents per year: ', accidents_pyear[0], '-', accidents_pyear[1])
annual_cost = [(data['All Costs'].median())*321*1000000, (data['All Costs'].median())*349*1000000]



print('ANNUAL COST OF OIL PIPELINE ACCIDENTS: ', round(annual_cost[0]), '-', round(annual_cost[1]), 'USD')