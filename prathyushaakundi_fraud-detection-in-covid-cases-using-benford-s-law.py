

import numpy as np 

import pandas as pd 

import seaborn as sns

import random

import matplotlib.pyplot as plt

import math
def firstDigit(n) : 

  

    while n >= 10:  

        n = n / 10



    return int(n) 
BENFORD = [30.1, 17.6, 12.5, 9.7, 7.9, 6.7, 5.8, 5.1, 4.6]
plt.figure(figsize=(15,10))

plt.plot(BENFORD)

plt.title('Benford law distribution of first digits')
#Source: https://towardsdatascience.com/frawd-detection-using-benfords-law-python-code-9db8db474cf8

def chi_square_test(data_count,expected_counts):

    """Return boolean on chi-square test (8 degrees of freedom & P-val=0.05)."""

    chi_square_stat = 0  # chi square test statistic

    for data, expected in zip(data_count,expected_counts):



        chi_square = math.pow(data - expected, 2)



        chi_square_stat += chi_square / expected



    print("\nChi-squared Test Statistic = {:.3f}".format(chi_square_stat))

    print("Critical value at a P-value of 0.05 is 15.51.")    

    return chi_square_stat < 15.51
covid_daily = pd.read_csv('../input/corona-virus-report/day_wise.csv')

covid_daily.head()
confirmed_fd = []

confirmed = covid_daily.Confirmed.values



for i in confirmed:

    confirmed_fd.append(firstDigit(i))
confired_fd_counts = pd.Series(confirmed_fd).value_counts().values



confired_fd_percent = (confired_fd_counts/np.sum(confired_fd_counts))*100



plt.figure(figsize=(15,10))

plt.plot(confired_fd_percent)

plt.plot(BENFORD)

plt.legend(['Covid worldwide','BENFORD'])
chi_square_test(confired_fd_percent,BENFORD)
india_daily = pd.read_csv('../input/covid19-in-india/covid_19_india.csv')
india_daily.head(-5)
india_daily['State/UnionTerritory'].unique()
india_daily['State/UnionTerritory'].replace('Telengana','Telangana', inplace=True)



india_daily['State/UnionTerritory'].replace('Telangana***','Telangana', inplace=True)

india_daily['State/UnionTerritory'].replace('Telengana***','Telangana', inplace=True)
india_daily['State/UnionTerritory'].unique()
#Containers for states that failed to conform with Benford's Law

global cured_sus_states

global confirmed_sus_states

global death_sus_states



confirmed_sus_states = []

cured_sus_states = []

death_sus_states = []
def covid_distribution(state_name, col):

    state = india_daily[india_daily['State/UnionTerritory'] == state_name]

    state = state[state[col]!=0]

    if(len(state)>100):

        state_fd = []

        state_confirmed = state[col].values



        for i in state_confirmed:

            state_fd.append(firstDigit(i))





        confired_fd_counts = pd.Series(state_fd).value_counts().sort_index().values

        #Consider only natural numbers

        if(0 in state_fd):

            confired_fd_counts = confired_fd_counts[1:]

        confired_fd_percent = (confired_fd_counts/np.sum(confired_fd_counts))*100



        if(chi_square_test(confired_fd_percent,BENFORD)):

            title = '{0}  {1} conforms with Benfords Law'.format(state_name, col)

        else:

            title = ' {0} {1} state seem to have some manipulation'.format(state_name, col)



            if(col=='Confirmed'):

                confirmed_sus_states.append(state_name)

            elif(col=='Cured'):

                cured_sus_states.append(state_name)

            else:

                death_sus_states.append(state_name)



        plt.figure(figsize=(15,10))

        plt.plot(confired_fd_percent)

        plt.plot(BENFORD)

        plt.legend(['Statewise distribution','Benfords Distribution'])

        plt.title(title)

        plt.show()

    else:

        print('{0} for #{1} cases doesnt have enough records to run Benfords law'.format(state_name,col))

    

    

    

    
states = ['Kerala', 'Telangana', 'Delhi', 'Rajasthan', 'Uttar Pradesh',

       'Haryana', 'Ladakh', 'Tamil Nadu', 'Karnataka', 'Maharashtra',

       'Punjab', 'Jammu and Kashmir', 'Andhra Pradesh', 'Uttarakhand',

       'Odisha', 'Puducherry', 'West Bengal', 'Chhattisgarh',

       'Chandigarh', 'Gujarat', 'Himachal Pradesh', 'Madhya Pradesh',

       'Bihar', 'Manipur', 'Mizoram', 'Andaman and Nicobar Islands',

       'Goa',  'Assam', 'Jharkhand', 'Arunachal Pradesh',

       'Tripura', 'Nagaland', 'Meghalaya', 

       'Sikkim', 'Dadra and Nagar Haveli and Daman and Diu']
for state in states:

    covid_distribution(state,'Confirmed' )
print('List of states that have suspicious entries for number of Confirmed cases are: ',confirmed_sus_states)
for state in states:

    covid_distribution(state,'Cured' )
print('List of states that have suspicious entries for number of Cured cases are: ',cured_sus_states)
for state in states:

    covid_distribution(state,'Deaths' )
print('List of states that have suspicious entries for number of Deaths are: ',death_sus_states)
len(death_sus_states)
#States that have suspicious behaviour on all three: Confirmed, Cured, Deaths



states_all_sus = set(death_sus_states).intersection(set(cured_sus_states)).intersection(set(confirmed_sus_states))
print(states_all_sus)