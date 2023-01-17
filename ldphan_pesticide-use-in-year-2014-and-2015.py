#!/usr/bin/env python3

# Save these below code lines as file pu_preprocessing.py and run the notebook-file .ipynb below



__author__ = 'Long Phan'



'''

Description: Import parameters from config.ini and execute data pre_processing 

'''

import configparser

import pandas as pd

import numpy as np

from sklearn.preprocessing import Imputer

import matplotlib.pyplot as plt

from datetime import datetime

from matplotlib.pylab import rcParams



import timeit





def get_arguments():

    """

    read config-parameters

    """

    data_set2014_path='../input/2014.csv'

    data_set2015_path='../input/2015.csv'

    data_dictionary_path='../input/dictionary.csv'

    

    kwargs = {"data_set2014_path": data_set2014_path, "data_set2015_path": data_set2015_path,

              "data_dictionary_path": data_dictionary_path}



    return kwargs





def group_columns_mean(data, columns, column):

    """

    return mean value of the given columns grouped by the given column

    :param data:

    :param columns:

    :param column:

    :return:

    """

    return data[columns].groupby(column).mean()





def get_data_state_mean_est(data):

    return group_columns_mean(data, columns=['STATE_CODE', 'LOW_ESTIMATE', 'HIGH_ESTIMATE'], column='STATE_CODE')





def get_data_state_mean_est_spec(data, state_code, estimate='LOW_ESTIMATE'):

    state_mean = get_data_state_mean_est(data)



    if estimate == 'HIGH_ESTIMATE' and state_code in state_mean.index:

        return state_mean.at[state_code, 'HIGH_ESTIMATE']

    elif state_code in state_mean.index:

        return get_data_state_mean_est(data).at[state_code, 'LOW_ESTIMATE']

    else:

        print("No valid data about state ", state_code)

        return





def get_state_list(data):

    return get_data_state_mean_est(data).index





def nan_columns(dat):

    """

    return name of all columns which have NaN_value

    :param dat:

    :return:

    """

    kc = dat.isnull().any()

    # print(kc.keys())

    key_true = [key for key, value in kc.iteritems() if value]



    return key_true





def nan_rows(data):

    """

    return all rows containing NaN values in type DataFrame

    :param data:

    :return:

    """

    return data[data.isnull().any(axis=1)]





def process_missing_data(data):



    # Replace missing value as NaN by 0 following requirement 'LOW_ESTIMATE'

    data[['LOW_ESTIMATE']] = data[['LOW_ESTIMATE']].replace(np.NaN, 0)



    # Find all nans_row similarly as nans = lambda df: df[df.isnull().any(axis=1)]

    # and replace missing value as NaN by mean of the neighboring states following requirement 'high_estimate'

    # TODO: apply(lambda ...)

    nans_dat = nan_rows(data)

    for index, row in nans_dat.iterrows():

        new_value = get_data_state_mean_est_spec(data, nans_dat['STATE_CODE'].values[0], 'HIGH_ESTIMATE')

        data.loc[index, 'HIGH_ESTIMATE'] = new_value



    return data





def run():

    """

    Read data from data_set and convert to time series format

    """

    argument = get_arguments()



    data2014 = pd.read_csv(argument['data_set2014_path'])

    data2015 = pd.read_csv(argument['data_set2015_path'])

    dictionary = pd.read_csv(argument['data_dictionary_path'])



    return process_missing_data(data2014), process_missing_data(data2015), dictionary





def get_max_min(data):

    """

    return max value and min value of every year in type Series

    """

    max_year_high_est = (data[['YEAR', 'HIGH_ESTIMATE']]).max()

    max_year_low_est = (data[['YEAR', 'LOW_ESTIMATE']]).max()



    min_year_high_est = (data[['YEAR', 'HIGH_ESTIMATE']]).min()

    min_year_low_est = (data[['YEAR', 'LOW_ESTIMATE']]).min()



    return max_year_high_est, max_year_low_est, min_year_high_est, min_year_low_est





d2014, d2015, states = run()



d2014_states = d2014.merge(states, how='left', on=['STATE_CODE', 'COUNTY_CODE'])

d2015_states = d2015.merge(states, how='left', on=['STATE_CODE', 'COUNTY_CODE'])

# import and preprocess data

%matplotlib inline

rcParams['figure.figsize'] = 20, 8



d2014_states

d2015_states
data_years = [d2014_states, d2015_states]

for dy in data_years:

    dy_mx = dy.as_matrix()

    x = dy_mx[:, 2]   # State_Code

    yl = dy_mx[:, 4]  # LOW_ESTIMATE

    yh = dy_mx[:, 5]  # HIGH_ESTIMATE

    print("YEAR ", dy['YEAR'][1])

    print(yl.min(), yl.max(), yh.min(), yh.max())

    print("\n")
dy_mx = d2014_states.as_matrix()

x = dy_mx[:, 2]   # State_Code



yl = dy_mx[:, 4]  # LOW_ESTIMATE

plt.scatter(x, yl, label="LOW_ESTIMATE")



yh = dy_mx[:, 5]  # HIGH_ESTIMATE

plt.scatter(x, yh, marker='x', label="HIGH_ESTIMATE")



plt.xticks(range(x.max()+1))

plt.yticks(range(0, 6000000, 200000))

plt.xlabel("State_Code")

plt.ylabel("Usage pesticide-use")

plt.title("Pesticide-use in all states in year 2014")

plt.legend()
# input state_code 

state_code = 16



# choose d2015_states for data object in year 2015

data_states = d2015_states  



state = data_states.loc[data_states['STATE_CODE'] == state_code]



county_code = state['COUNTY_CODE'].drop_duplicates()



state_mx = state.as_matrix()

x = state_mx[:, 3]  # County_Code



yl = state_mx[:, 4]  # LOW_ESTIMATE

yh = state_mx[:, 5]  # HIGH_ESTIMATE



plt.scatter(x, yl, label="LOW_ESTIMATE", color='green')

plt.scatter(x, yh, marker='x',label="HIGH_ESTIMATE", color='red')



plt.xticks(range(county_code.max()))

plt.yticks(range(0, 1500000, 100000))



plt.xlabel("County_Code")

plt.ylabel("Usage pesticide-use")

plt.title("Pesticide-use in state %i " %state_code + "in year %i" %data_states['YEAR'][1])

plt.legend()
states_code = d2014_states[['COMPOUND']].groupby(d2014_states['STATE_CODE']).describe().index

state_compounds = d2014_states[['COMPOUND', 'STATE_CODE']]



# Choose state_code = 1

# state_code = 1

# filter = d2014_states['STATE_CODE']== state_code

# st = state_compounds[filter]

# st_cp1 = st.groupby(st['COMPOUND']).count()

# st_cp1.sort_values(axis=0, ascending=False, by='STATE_CODE', kind='quicksort').head(10)



# Loop to show result in all States, choose the first 20 most used Compounds

# remove head(20) to show all Compounds

for sc in states_code:

    filter = d2014_states['STATE_CODE']== sc

    st = state_compounds[filter]

    st_cp = st.groupby(st['COMPOUND']).count()

    print("\nSTATE_CODE ", sc)    

    print(st_cp.sort_values(axis=0, ascending=False, by='STATE_CODE', kind='quicksort').head(20))

    

# Find index of State_code, e.g. states_code[41] = 49

# for idx, value in enumerate(states_code):

#     print(idx, value)
# Visualize the above result, choose the first 10 STATE_CODE 

# replace "states_code[0:10]" by "states_code" to plot for all the States

for sc in states_code[0:10]:

    filter = d2014_states['STATE_CODE']==sc

    st = state_compounds[filter]

    st_cp = st.groupby(st['COMPOUND']).count()

    st_cp = st_cp.sort_values(axis=0, ascending=False, by='STATE_CODE', kind='quicksort').head(20)

    st_cp.plot(kind='bar')

    plt.ylabel("Frequently used")

    plt.title("The compounds are used in STATE %i" %sc)  

    
# Compact summary from the above result 

# 

# WARNING: 

# In case that STATE can have more than 1 Compound which are used the most frequently)

# This summary below only shows randomly one of them (Full detail, please check the above result)

compact_summary = d2014_states[['COMPOUND']].groupby(d2014_states['STATE_CODE']).describe()



# slicing columns multi_index

compound_freq = compact_summary.loc[:, (slice('COMPOUND'), ['top', 'freq'])]



compound_freq
for dy in data_years:

    max_year_high_est, max_year_low_est, min_year_high_est, min_year_low_est = get_max_min(dy) 

    print("MAX")

    print("---- ", max_year_high_est)

    print("---- ", max_year_low_est)

    print("MIN")

    print("---- ", min_year_high_est)

    print("---- ", min_year_low_est)

    print("\n")

    

# mean value of pesticide-use of all states 

data2014_state_mean = get_data_state_mean_est(d2014_states)

data2014_state_mean.plot(kind='bar')

plt.xticks(rotation=0)

plt.title("Mean value of pesticide-use of all states in year 2014")



data2015_state_mean = get_data_state_mean_est(d2015_states)

data2015_state_mean.plot(kind='bar')

plt.xticks(rotation=0)

plt.title("Mean value of pesticide-use of all states in year 2015")

plt.legend()
dx14 = data2014_state_mean.index

dx15 = data2015_state_mean.index

# check whether all states reported about pesticide-use in year 2014 and year 2015

dx14.equals(dx15)
# Which states were reported in 2014 but missing in 2015 and vice versa?

dxa14 = np.array(dx14)

dxa15 = np.array(dx15)

missing_states14_15 = np.setdiff1d(dxa14, dxa15), 

missing_states15_14 = np.setdiff1d(dxa15, dxa14)

missing_states14_15, missing_states15_14



filter = states['STATE_CODE']

states[filter==missing_states14_15[0][0]]['STATE'].drop_duplicates().values[0]
data_diff_state_mean = data2015_state_mean - data2014_state_mean

data_diff_state_mean
data_diff_state_mean[['LOW_ESTIMATE']] = data_diff_state_mean[['LOW_ESTIMATE']].replace(np.NaN, 0)

data_diff_state_mean[['HIGH_ESTIMATE']] = data_diff_state_mean[['HIGH_ESTIMATE']].replace(np.NaN, 0)

data_diff_state_mean.plot(kind='bar')

plt.xticks(rotation=0)

plt.title("The Differences (changes) in all States between year 2014 and year 2015")
# Classify the differences in 3 groups (high) increases, (low) increases and decreases

from sklearn.cluster import KMeans



kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 0)

y_kmeans = kmeans.fit_predict(data_diff_state_mean)



data_diff_state_mean.index[y_kmeans == 0]

data_diff_state_mean[y_kmeans == 0].plot(kind='bar')

plt.title("High changes (increasing) in pesticide-use of states")

plt.xticks(rotation=0)



data_diff_state_mean.index[y_kmeans == 1]

data_diff_state_mean[y_kmeans == 1].plot(kind='bar')

plt.title("Low changes (increasing) or small (decreasing) in pesticide-use of states")

plt.xticks(rotation=0)



data_diff_state_mean.index[y_kmeans == 2]

data_diff_state_mean[y_kmeans == 2].plot(kind='bar')

plt.title("Changes (Decreasing) in pesticide-use of states")



plt.xticks(rotation=0)

plt.xlabel('State_Code')

plt.ylabel('Pesticide-use')

plt.legend()
compound_freq.to_csv('pesticide-use_submission.csv', index=False)