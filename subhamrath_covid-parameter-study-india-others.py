import pandas as pd

import numpy as np

import scipy as sp

from matplotlib import pyplot as plt

import seaborn as sns

from scipy import stats

from scipy.optimize import minimize

from scipy.optimize import curve_fit

from scipy.optimize import differential_evolution

import warnings

from IPython.display import Image

from sklearn.preprocessing import MinMaxScaler

from sklearn.cluster import DBSCAN
covid_cnf_ts = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

covid_de_ts = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')

covid_re_ts = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')
plt.style.use('fivethirtyeight')
date_columns = list(covid_cnf_ts.iloc[:,4:].columns)

covid_cnf_ts = covid_cnf_ts[covid_cnf_ts['Country/Region'] != 'Cruise Ship']

covid_de_ts = covid_de_ts[covid_de_ts['Country/Region'] != 'Cruise Ship']

covid_re_ts = covid_re_ts[covid_re_ts['Country/Region'] != 'Cruise Ship']



covid_cnf_ts_grouped = covid_cnf_ts.groupby('Country/Region')[date_columns].sum()

covid_de_ts_grouped = covid_de_ts.groupby('Country/Region')[date_columns].sum()

covid_re_ts_grouped = covid_re_ts.groupby('Country/Region')[date_columns].sum()
covid_details = pd.concat([covid_cnf_ts_grouped.iloc[:,-1],covid_re_ts_grouped.iloc[:,-1], covid_de_ts_grouped.iloc[:,-1]], axis = 1 )

covid_details.columns =['Confirmed', 'Recovery', 'Death']

covid_details = covid_details[(covid_details['Confirmed'] > 150) & (covid_details['Recovery'] > 10) & (covid_details['Death']>0)]

covid_cnf_aggregate = covid_cnf_ts_grouped.T.sum()

covid_cnf_ts_grouped['Total'] = covid_cnf_aggregate

sort_cnf = covid_cnf_ts_grouped.sort_values(by = ['Total'], ascending = False)

affected_countries_confirmed = covid_cnf_ts_grouped

affected_countries_confirmed = affected_countries_confirmed.ix[sort_cnf.index].dropna()

affected_countries_confirmed = affected_countries_confirmed[affected_countries_confirmed.iloc[:,-2]>150]



covid_de_aggregate = covid_de_ts_grouped.T.sum()

covid_de_ts_grouped['Total'] = covid_de_aggregate

sort_de = covid_de_ts_grouped.sort_values(by = ['Total'], ascending = False)

affected_countries_death = covid_de_ts_grouped

affected_countries_death = affected_countries_death.ix[sort_de.index].dropna()

affected_countries_death = covid_de_ts_grouped[covid_de_ts_grouped.iloc[:,-2]> 20]



covid_re_aggregate = covid_re_ts_grouped.T.sum()

covid_re_ts_grouped['Total'] = covid_re_aggregate

sort_re = covid_re_ts_grouped.sort_values(by = ['Total'], ascending = False)

affected_countries_recovery = covid_re_ts_grouped

affected_countries_recovery = affected_countries_recovery.ix[sort_re.index].dropna()

affected_countries_recovery = covid_re_ts_grouped[covid_re_ts_grouped.iloc[:,-2] > 100]

fig = plt.figure(figsize = (20,20))

affected_countries_confirmed.iloc[0,:-1].plot(ax = fig.add_subplot(3,2,1))

plt.legend()

affected_countries_confirmed.iloc[1:10,:-1].T.plot(ax = fig.add_subplot(3,2,2))

# plt.legend()

affected_countries_confirmed.iloc[11:21,:-1].T.plot(ax = fig.add_subplot(3,2,3))

affected_countries_confirmed.iloc[22:32,:-1].T.plot(ax = fig.add_subplot(3,2,4))

affected_countries_confirmed.iloc[33:43,:-1].T.plot(ax = fig.add_subplot(3,2,5))

affected_countries_confirmed.iloc[44:54,:-1].T.plot(ax = fig.add_subplot(3,2,6))

plt.suptitle('COVID-19 time series of confirmed cases')

plt.show()
fig = plt.figure(figsize = (16,10))

# affected_countries_death.iloc[0,:-1].plot(ax = fig.add_subplot(1,2,1))

affected_countries_death.iloc[:7,:-1].T.plot(ax = fig.add_subplot(1,2,1))

# plt.legend()

affected_countries_death.iloc[8:13,:-1].T.plot(ax = fig.add_subplot(1,2,2))

# affected_countries_death.iloc[33:43,:-1].T.plot(ax = fig.add_subplot(3,2,5))

# affected_countries_death.iloc[44:52,:-1].T.plot(ax = fig.add_subplot(3,2,6))



plt.suptitle('COVID-19 time series of death cases')



plt.show()

recovery_cnf = covid_details['Recovery']/covid_details['Confirmed']

death_cnf = covid_details['Death']/covid_details['Confirmed']

death_recovery = covid_details['Death']/covid_details['Recovery']
covid_details
plt.figure(figsize = (20,18))

plt.subplot(3,1,1)

sns.barplot(recovery_cnf.index, recovery_cnf)

plt.xticks(rotation=45, ha="right")

plt.title('Recovery to Confirmed cases')



plt.subplot(3,1,2)

sns.barplot(death_cnf.index, death_cnf)

plt.xticks(rotation=45, ha="right")

plt.title('Death to Confirmed cases')



# plt.subplot(3,1,3)

# sns.barplot(death_recovery.index, death_recovery)

# plt.xticks(rotation=45, ha="right")

# plt.title('Death to Recovered cases')
# fig = plt.figure(figsize = (16,8))

# affected_countries_recovery.iloc[:,:-2].T.plot(ax = fig.add_subplot(1,2,2))

# # plt.legend()

# # affected_countries_death.iloc[11:21,:-1].T.plot(ax = fig.add_subplot(3,2,3))

# # affected_countries_death.iloc[22:32,:-1].T.plot(ax = fig.add_subplot(3,2,4))

# # affected_countries_death.iloc[33:43,:-1].T.plot(ax = fig.add_subplot(3,2,5))

# # affected_countries_death.iloc[44:52,:-1].T.plot(ax = fig.add_subplot(3,2,6))



def covid_likelihood(params, *data):

    '''Constructs a likelihood based on the data observed'''

    

    k = params[0]

    b= params[1]

    sd = params[2]

    y_dat = data

    f = 1/(1+np.exp(-k*(x_dat-b)))

#     print(stats.norm.logpdf(y_dat, f, sd))

    likelihood = - np.sum(stats.norm.logpdf(y_dat/y_dat[-1], f, sd))

    return likelihood



def sigmoid(x,a,b,c):

    '''Non scaled sigmoid function to model the data'''

    c = 1

    f = c/(1+np.exp(-(x-b)/a))

    return f



def sigmoid_1(x,a,b):

    '''Scaled sigmoid function to model the normalized data'''

    f = 1/(1+np.exp(-(x-b)/a))

    return f



def get_param_estimate(function, initparams):

    estimates = minimize(covid_likelihood, [1,1,1], method = 'Nelder-Mead')

    return estimates.x



def func_exp(x, a,b, c):

    c = 0

    return a * np.exp(b * x) + c



def parameter_estimations(x, y, scale_flag):

    '''Provides functionality for parameter estimations 

    with or without scaling (provided by scale_flag)'''

    if scale_flag:

        y_scale = y/y[-1]

        p0 = [2, np.argmax(y)]

        popt, pcov = curve_fit(sigmoid_1, x, y_scale, p0, method='dogbox',maxfev=100000)

        parameter = [popt[0], popt[1]]

    else:

        p0 = [2, np.argmax(y_dat), np.max(y)]

        popt, pcov = curve_fit(func_exp, x, y, p0 , maxfev = 10000)

        parameter = [popt[0], popt[1], popt[2]]

    return parameter
y_dat = affected_countries_confirmed.iloc[34,:-1].values

x_dat = np.arange(0,len(affected_countries_recovery.iloc[0,:-1].values))

#A sample parameter estimation 

parameter_estimations(x_dat, y_dat, True)
covid_parameter_dict = {}

for i in range(len(affected_countries_confirmed)):

    y_dat = affected_countries_confirmed.iloc[i,:-2].values

    x_dat = np.arange(0,len(affected_countries_recovery.iloc[0,:-2].values))

    parameter = parameter_estimations(x_dat, y_dat, True)

    covid_parameter_dict[affected_countries_confirmed.index[i]] = parameter

    
parameter_dataframe = pd.DataFrame(covid_parameter_dict)

parameter_dataframe = parameter_dataframe.T

parameter_dataframe

parameter_dataframe.columns = ['param_1', 'param_2']

plt.scatter(parameter_dataframe['param_1'], parameter_dataframe['param_2'])

plt.xlabel('Growth rate')

plt.ylabel ('Time at midpoint')

plt.show()
parameter_dataframe
scaler = MinMaxScaler()

param_scaled = scaler.fit_transform(parameter_dataframe.iloc[:,:2])

param_scaled_df = pd.DataFrame(param_scaled)

param_scaled_df.columns = ['Param_1', 'Param_2']
model_1 = DBSCAN(0.05,3).fit(param_scaled_df)

cluster_labels_1 = model_1.labels_

param_scaled_df['cluster'] = cluster_labels_1
cluster_labels_1

param_scaled_df.index = parameter_dataframe.index



o_x, o_y = param_scaled_df[param_scaled_df['cluster']==-1]['Param_1'], param_scaled_df[param_scaled_df['cluster']==-1]['Param_2']

c_x, c_y = param_scaled_df[param_scaled_df['cluster']==0]['Param_1'], param_scaled_df[param_scaled_df['cluster']==0]['Param_2']
plt.figure(figsize = (18,6))

plt.subplot(1,2,1)

plt.scatter(param_scaled_df['Param_1'], param_scaled_df['Param_2'])

plt.xlabel('Growth rate')

plt.ylabel ('Time at midpoint')

plt.title('Pre-clustering')

plt.subplot(1,2,2)

plt.scatter(o_x,o_y)

plt.scatter(c_x,c_y)

plt.xlabel('Growth rate')

plt.ylabel ('Time at midpoint')

plt.title('Post-clustering (DBSCAN)')

plt.suptitle('COVID-19 Parameter clustering')

# plt.savefig('cv_4.png')



# plt.scatter(cl2_x,cl2_y)

# plt.scatter(cl3_x,cl3_y)
outlier_1 = param_scaled_df[param_scaled_df['cluster']== -1]

outlier_1
outlier
cluster_11 = param_scaled_df[param_scaled_df['cluster'] == 0]

outlier_cnf = affected_countries_confirmed.iloc[:,:-1].ix[outlier_1.index]

fig = plt.figure(figsize =(15,9))

outlier_cnf[outlier_cnf.index == 'China'].iloc[:,:-1].T.plot(ax = fig.add_subplot(1,3,1))

outlier_cnf[outlier_cnf.index == 'Korea, South'].iloc[:,:-1].T.plot(ax = fig.add_subplot(1,3,2))

outlier_cnf[outlier_cnf.index == 'Iran'].iloc[:,:-1].T.plot(ax = fig.add_subplot(1,3,2))

plt.xlabel('Time')

outlier_cnf[(outlier_cnf.index != 'China') & (outlier_cnf.index !='Korea, South') & (outlier_cnf.index !='Iran')].iloc[:,:-1].T.plot(ax = fig.add_subplot(1,3,3))

plt.xlabel('Time')

plt.suptitle('Time series of confirmed cases of outliers ')

# 

plt.show()
cluster_cnf = affected_countries_confirmed.iloc[:,:-1].ix[cluster_11.index]

param_1_ind = param_scaled_df.loc['India']['Param_1']

param_2_ind = param_scaled_df.loc['India']['Param_2']

distance = {}

for i in param_scaled_df.index:

    par_1 = param_scaled_df.loc[i]['Param_1']

    par_2 = param_scaled_df.loc[i]['Param_2']

    dist = (par_1- param_1_ind)**2 + (par_2- param_2_ind)**2

    distance[i] = dist

import operator

sorted_d = sorted(distance.items(), key=operator.itemgetter(1))

print('Country with Covid growth parameter closest to India:', sorted_d[1:10])
country_list = [i[0] for i in sorted_d[1:15]]

closest_country_confirmed = affected_countries_confirmed.ix[country_list]

df = pd.DataFrame(affected_countries_confirmed.ix[country_list].iloc[:,-2])

df.column = ['Confirmed cases']

sns.heatmap(df, annot=True, cmap='viridis', cbar=False, fmt = 'g')

plt.title('Countries with COVID-19 parameters closest to India')

# plt.savefig('')

plt.show()
plt.figure(figsize = (18,5))

sns.barplot(country_list, np.log(affected_countries_confirmed.ix[country_list].iloc[:,-2]))

plt.xticks(rotation=45, ha="right")

plt.title('log confirmed cases by country closest to India ')

plt.show()