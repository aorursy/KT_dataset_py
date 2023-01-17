# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('../input/qvi-data/QVI_data.csv')

data.head()
data['month_year'] = pd.to_datetime(data['DATE']).dt.to_period('M')
data['PRICE'] = data['TOT_SALES'] / data['PROD_QTY']
measures_over_time = data.groupby(['STORE_NBR', 'month_year', ], as_index= False).agg(

    totSales = pd.NamedAgg(column="TOT_SALES", aggfunc="sum"),

    nCustomers = pd.NamedAgg(column="LYLTY_CARD_NBR", aggfunc="nunique"),

    nTxnPerStore = pd.NamedAgg(column="TXN_ID", aggfunc="count"),

    nChipsPerStore = pd.NamedAgg(column="PROD_QTY", aggfunc="sum"),

    avgPricePerUnit = pd.NamedAgg(column="PRICE", aggfunc="mean"))



measures_over_time['nTxnPerCust'] =measures_over_time['nTxnPerStore']/measures_over_time['nCustomers']

measures_over_time['nChipsPerTxn'] =measures_over_time['nChipsPerStore']/measures_over_time['nCustomers']



measures_over_time = measures_over_time.drop(labels = ['nTxnPerStore', 'nChipsPerStore'], axis = 1)



measures_over_time
measures_over_time.isnull().sum()
observations_per_store = measures_over_time.groupby('STORE_NBR', as_index = False).month_year.count()

store_nbr = observations_per_store[ observations_per_store.month_year == 12].STORE_NBR.array

store_nbr
measures_over_time = measures_over_time[ measures_over_time.STORE_NBR.isin(store_nbr)]
preTrialMeasures =  measures_over_time[measures_over_time.month_year < '2019-02']
trial_stores_number = [77, 86, 88]

comparison_stores_number = measures_over_time.STORE_NBR.unique().tolist()

comparison_stores_number
def calc_corr( measure):

    corrTable = pd.DataFrame( columns = ['Input Store','Comparison Store', 'Measure', 'Correlation'])

    for i in trial_stores_number:

        trial_store_measure = preTrialMeasures.loc[ preTrialMeasures.STORE_NBR == i, measure]

        for j in comparison_stores_number:

            comparison_store_measure = preTrialMeasures.loc[ preTrialMeasures.STORE_NBR == j, measure]

            corr = np.corrcoef(trial_store_measure, comparison_store_measure)[0][1]

            corrTable = corrTable.append({

                'Input Store': i, 

                'Comparison Store': j, 

                'Measure': measure, 

                'Correlation': corr}, ignore_index=True)

    return corrTable
corr_table_totSales = calc_corr( 'totSales')

corr_table_totSales
corr_table_nCustomers = calc_corr( 'nCustomers')

corr_table_nCustomers
# corr_table_avgPricePerUnit = calc_corr( 'avgPricePerUnit')

# corr_table_avgPricePerUnit
# corr_table_nTxnPerCust = calc_corr( 'nTxnPerCust')

# corr_table_nTxnPerCust
# corr_table_nChipsPerTxns = calc_corr( 'nChipsPerTxn')

# corr_table_nChipsPerTxns
# corr_table = pd.concat([corr_table_totSales, corr_table_nCustomers, corr_table_nTxnPerCust, corr_table_nChipsPerTxns, corr_table_avgPricePerUnit], axis=0)

corr_table = pd.concat([corr_table_totSales, corr_table_nCustomers], axis=0)

corr_table
month_year = preTrialMeasures.month_year.unique()

month_year
def calc_distance( measure):

    distanceTable = pd.DataFrame( columns = ['Input Store','Comparison Store', 'Year Month', 'Measure', 'Distance'])

    for i in trial_stores_number:

        for j in comparison_stores_number:

            for month in month_year:

                trial_store_measures = preTrialMeasures.loc[ (preTrialMeasures.STORE_NBR == i) &  (preTrialMeasures.month_year == month), measure].values[0]

                comparison_store_measures = preTrialMeasures.loc[ (preTrialMeasures.STORE_NBR == j) &  (preTrialMeasures.month_year == month), measure].values[0]

                distance = trial_store_measures - comparison_store_measures

                distanceTable = distanceTable.append({

                    'Input Store': i, 

                    'Comparison Store': j, 

                    'Year Month': month,

                    'Measure' : measure,

                    'Distance' : abs(distance)}, ignore_index = True )

    return distanceTable
distance_table_totSales = calc_distance('totSales')

distance_table_totSales

distance = distance_table_totSales.Distance

min_distance = distance.min()

max_distance = distance.max()



distance_table_totSales['DistanceStand'] = 1 - (distance - min_distance)/(max_distance - min_distance)



distance_table_totSales_grouped = distance_table_totSales.groupby(['Input Store', 'Comparison Store','Measure'], as_index = False).DistanceStand.mean()

distance_table_totSales_grouped
distance_table_nCustomers = calc_distance('nCustomers')

distance_table_nCustomers



distance_table_nCustomers['Distance']= distance_table_nCustomers.Distance.astype(int)



distance = distance_table_nCustomers.Distance

min_distance = distance.min()

max_distance = distance.max()



distance_table_nCustomers['DistanceStand'] = 1 - (distance - min_distance)/(max_distance - min_distance)



distance_table_nCustomers_grouped = distance_table_nCustomers.groupby(['Input Store', 'Comparison Store', 'Measure'], as_index = False).DistanceStand.mean()

distance_table_nCustomers_grouped
# distance_table_nTxnPerCust = calc_distance('nTxnPerCust')

# distance_table_nTxnPerCust



# distance = distance_table_nTxnPerCust.Distance

# min_distance = distance.min()

# max_distance = distance.max()



# distance_table_nTxnPerCust['DistanceStand'] = 1 - (distance - min_distance)/(max_distance - min_distance)

# distance_table_nTxnPerCust



# distance_table_nTxnPerCust_grouped = distance_table_nTxnPerCust.groupby(['Input Store', 'Comparison Store', 'Measure'], as_index = False).DistanceStand.mean()

# distance_table_nTxnPerCust_grouped
# distance_table_nChipsPerTxn = calc_distance('nChipsPerTxn')

# distance_table_nChipsPerTxn



# distance = distance_table_nChipsPerTxn.Distance

# min_distance = distance.min()

# max_distance = distance.max()



# distance_table_nChipsPerTxn['DistanceStand'] = 1 - (distance - min_distance)/(max_distance - min_distance)

# distance_table_nChipsPerTxn



# distance_table_nChipsPerTxn_grouped = distance_table_nChipsPerTxn.groupby(['Input Store', 'Comparison Store', 'Measure'], as_index = False).DistanceStand.mean()

# distance_table_nChipsPerTxn_grouped
# distance_table_avgPricePerUnit = calc_distance('avgPricePerUnit')

# distance_table_avgPricePerUnit



# distance = distance_table_avgPricePerUnit.Distance

# min_distance = distance.min()

# max_distance = distance.max()



# distance_table_avgPricePerUnit['DistanceStand'] = 1 - (distance - min_distance)/(max_distance - min_distance)

# distance_table_avgPricePerUnit



# distance_table_avgPricePerUnit_grouped = distance_table_avgPricePerUnit.groupby(['Input Store', 'Comparison Store', 'Measure'], as_index = False).DistanceStand.mean()

# distance_table_avgPricePerUnit_grouped
# distance = pd.concat([distance_table_totSales_grouped, distance_table_nCustomers_grouped, distance_table_nTxnPerCust_grouped, distance_table_nChipsPerTxn_grouped, distance_table_avgPricePerUnit_grouped], axis=0)

distance = pd.concat([distance_table_totSales_grouped, distance_table_nCustomers_grouped], axis=0)
scores =  pd.concat([corr_table, distance], axis=1)
scores.shape
scores
scores_without_duplicates = scores.T.drop_duplicates().T


scores_without_duplicates.Correlation = scores_without_duplicates.Correlation.astype(float)

scores_without_duplicates.DistanceStand = scores_without_duplicates.DistanceStand.astype(float)

scores_without_duplicates
scores_without_duplicates['Rank'] = 0.5 * scores_without_duplicates.Correlation + 0.5 * scores_without_duplicates.DistanceStand
scores_without_duplicates
wighted_ranks = scores_without_duplicates.groupby(['Input Store', 'Comparison Store'], as_index = False).Rank.mean()

wighted_ranks
wighted_ranks_store77 = wighted_ranks[( wighted_ranks['Comparison Store'] != 77) & ( wighted_ranks['Input Store'] == 77)]



maximum = wighted_ranks_store77['Rank'] .max()

maximum

wighted_ranks[ (wighted_ranks['Input Store'] == 77) & (wighted_ranks['Rank'] == maximum) ]
wighted_ranks_store86 = wighted_ranks[( wighted_ranks['Comparison Store'] != 86) & ( wighted_ranks['Input Store'] == 86)]



maximum = wighted_ranks_store86['Rank'] .max()

maximum

wighted_ranks[ (wighted_ranks['Input Store'] == 86) & (wighted_ranks['Rank'] == maximum) ]
wighted_ranks_store88 = wighted_ranks[( wighted_ranks['Comparison Store'] != 88) & ( wighted_ranks['Input Store'] == 88)]



maximum = wighted_ranks_store88['Rank'] .max()

maximum

wighted_ranks[ (wighted_ranks['Input Store'] == 88) & (wighted_ranks['Rank'] == maximum) ]
import seaborn as sns

import matplotlib.pyplot as plt
preTrialMeasures.month_year = preTrialMeasures.month_year.astype(str)

preTrialMeasures.info()
preTrialMeasures_store77 = preTrialMeasures[ preTrialMeasures.STORE_NBR == 77]

preTrialMeasures_store233 = preTrialMeasures[ preTrialMeasures.STORE_NBR == 233]

preTrialMeasures_other = preTrialMeasures[ preTrialMeasures.STORE_NBR != 77]



ax = plt.subplot()

sns.lineplot(x = 'month_year', y = 'totSales' , data = preTrialMeasures_store77, ax = ax)

sns.lineplot(x = 'month_year', y = 'totSales' , data = preTrialMeasures_store233, ax = ax)

sns.lineplot(x = 'month_year', y = 'totSales' , data = preTrialMeasures_other, estimator= np.mean,ax = ax)

ax.set_title('Total sales during pretrial period ')

ax.set_xlabel('Time')

ax.set_ylabel('Total Sales')

ax.legend(['store 77', 'comparison store', 'other stores'])
preTrialMeasures_store77 = preTrialMeasures[ preTrialMeasures.STORE_NBR == 77]

preTrialMeasures_store233 = preTrialMeasures[ preTrialMeasures.STORE_NBR == 233]

preTrialMeasures_other = preTrialMeasures[ preTrialMeasures.STORE_NBR != 77]



ax = plt.subplot()

sns.lineplot(x = 'month_year', y = 'nCustomers' , data = preTrialMeasures_store77, ax = ax)

sns.lineplot(x = 'month_year', y = 'nCustomers' , data = preTrialMeasures_store233, ax = ax)

sns.lineplot(x = 'month_year', y = 'nCustomers' , data = preTrialMeasures_other, estimator= np.mean, ax = ax)

ax.set_title('Total number of customers during pretrial period')

ax.set_xlabel('Time')

ax.set_ylabel('Number of customers')

ax.legend(['store 77', 'comparison store', 'other stores'])
preTrialMeasures_store86 = preTrialMeasures[ preTrialMeasures.STORE_NBR == 86]

preTrialMeasures_store155 = preTrialMeasures[ preTrialMeasures.STORE_NBR == 155]

preTrialMeasures_other = preTrialMeasures[ preTrialMeasures.STORE_NBR != 86]



ax = plt.subplot()

sns.lineplot(x = 'month_year', y = 'totSales' , data = preTrialMeasures_store86, ax = ax)

sns.lineplot(x = 'month_year', y = 'totSales' , data = preTrialMeasures_store155, ax = ax)

sns.lineplot(x = 'month_year', y = 'totSales' , data = preTrialMeasures_other, estimator= np.mean, ax = ax)

ax.set_title('Total sales during pretrial period')

ax.set_xlabel('Time')

ax.set_ylabel('Total Sales')

ax.legend(['store 86', 'comparison store', 'other stores'])
preTrialMeasures_store86 = preTrialMeasures[ preTrialMeasures.STORE_NBR == 86]

preTrialMeasures_store155 = preTrialMeasures[ preTrialMeasures.STORE_NBR == 155]

preTrialMeasures_other = preTrialMeasures[ preTrialMeasures.STORE_NBR != 86]



ax = plt.subplot()

sns.lineplot(x = 'month_year', y = 'nCustomers' , data = preTrialMeasures_store86, ax = ax)

sns.lineplot(x = 'month_year', y = 'nCustomers' , data = preTrialMeasures_store155, ax = ax)

sns.lineplot(x = 'month_year', y = 'nCustomers' , data = preTrialMeasures_other, estimator= np.mean, ax = ax)

ax.set_title('Total number of customers during pretrial period')

ax.set_xlabel('Time')

ax.set_ylabel('Number of customers')

ax.legend(['store 86', 'comparison store', 'other stores'])
preTrialMeasures_store88 = preTrialMeasures[ preTrialMeasures.STORE_NBR == 88]

preTrialMeasures_store178 = preTrialMeasures[ preTrialMeasures.STORE_NBR == 178]

preTrialMeasures_other = preTrialMeasures[ preTrialMeasures.STORE_NBR != 88]



ax = plt.subplot()

sns.lineplot(x = 'month_year', y = 'totSales' , data = preTrialMeasures_store88, ax = ax)

sns.lineplot(x = 'month_year', y = 'totSales' , data = preTrialMeasures_store178, ax = ax)

sns.lineplot(x = 'month_year', y = 'totSales' , data = preTrialMeasures_other, estimator= np.mean, ax = ax)

ax.set_title('Total sales during pretrial period')

ax.set_xlabel('Time')

ax.set_ylabel('Total Sales')

ax.legend(['store 88', 'comparison store', 'other stores'])
preTrialMeasures_store88 = preTrialMeasures[ preTrialMeasures.STORE_NBR == 88]

preTrialMeasures_store178 = preTrialMeasures[ preTrialMeasures.STORE_NBR == 178]

preTrialMeasures_other = preTrialMeasures[ preTrialMeasures.STORE_NBR != 88]



ax = plt.subplot()

sns.lineplot(x = 'month_year', y = 'nCustomers' , data = preTrialMeasures_store88, ax = ax)

sns.lineplot(x = 'month_year', y = 'nCustomers' , data = preTrialMeasures_store178, ax = ax)

sns.lineplot(x = 'month_year', y = 'nCustomers' , data = preTrialMeasures_other, estimator= np.mean, ax = ax)

ax.set_title('Total number of customers during pretrial period')

ax.set_xlabel('Time')

ax.set_ylabel('Number of customers')

ax.legend(['store 88', 'comparison store', 'other stores'])
scalingFactorForControlSales  = preTrialMeasures[preTrialMeasures.STORE_NBR == 77].totSales.sum() / preTrialMeasures[preTrialMeasures.STORE_NBR == 233].totSales.sum()

scalingFactorForControlSales



measureOverTimeSales = measures_over_time.copy()

measureOverTimeSales.loc[ measureOverTimeSales.STORE_NBR == 233, 'totSales'] = measureOverTimeSales.loc[ measureOverTimeSales.STORE_NBR == 233, 'totSales'] * scalingFactorForControlSales

scaledControlSales = measureOverTimeSales[ measureOverTimeSales.STORE_NBR == 233]

scaledControlSales
measureOverTimeSalesStore77 = measureOverTimeSales.loc[ (measureOverTimeSales.STORE_NBR == 77)]

percentageDiff  = pd.merge(measureOverTimeSalesStore77, scaledControlSales, on = 'month_year', suffixes=('_trial', '_control'))

percentageDiff
percentageDiff['percentageDiffSales'] = abs(percentageDiff.totSales_control - percentageDiff.totSales_trial)/ percentageDiff.totSales_control

diffDuringPreTrial = percentageDiff[percentageDiff.month_year < '2019-02'].percentageDiffSales

diffDuringPreTrial
stdDev_pretrial = np.std(diffDuringPreTrial)

degreesOfFreedom = 7
t_critic = 1.895
t_calc = (percentageDiff.loc[percentageDiff.month_year.between('2019-02', '2019-04'),'percentageDiffSales'] - 0) / stdDev_pretrial

t_calc
t_calc > t_critic
scalingFactorForControlnCustomers  = preTrialMeasures[preTrialMeasures.STORE_NBR == 77].nCustomers.sum() / preTrialMeasures[preTrialMeasures.STORE_NBR == 233].nCustomers.sum()

scalingFactorForControlnCustomers



measureOverTimenCustomers = measures_over_time.copy()

measureOverTimenCustomers.loc[ measureOverTimenCustomers.STORE_NBR == 233, 'nCustomers'] = measureOverTimenCustomers.loc[ measureOverTimenCustomers.STORE_NBR == 233, 'nCustomers'] * scalingFactorForControlnCustomers

scaledControlnCustomers = measureOverTimenCustomers[ measureOverTimenCustomers.STORE_NBR == 233]

scaledControlnCustomers
measureOverTimenCustomersStore77 = measureOverTimenCustomers.loc[ (measureOverTimenCustomers.STORE_NBR == 77)]

percentageDiff  = pd.merge(measureOverTimenCustomersStore77, scaledControlnCustomers, on = 'month_year', suffixes=('_trial', '_control'))

percentageDiff
percentageDiff['percentageDiffnCustomers'] = abs(percentageDiff.nCustomers_control - percentageDiff.nCustomers_trial)/ percentageDiff.nCustomers_control

diffDuringPreTrial = percentageDiff[percentageDiff.month_year < '2019-02'].percentageDiffnCustomers

diffDuringPreTrial
stdDev_pretrial = np.std(diffDuringPreTrial)

degreesOfFreedom = 7
t_calc = (percentageDiff.loc[percentageDiff.month_year.between('2019-02', '2019-04'),'percentageDiffnCustomers'] - 0) / stdDev_pretrial

t_calc
t_calc > t_critic
factor  = preTrialMeasures[preTrialMeasures.STORE_NBR == 86].totSales.sum() / preTrialMeasures[preTrialMeasures.STORE_NBR == 155].totSales.sum()

factor



measuresOverTimeSales = measures_over_time.copy()
measuresOverTimeSales.loc[ measuresOverTimeSales.STORE_NBR == 155, 'totSales'] = measuresOverTimeSales.loc[ measuresOverTimeSales.STORE_NBR == 155, 'totSales'] * factor

scaledControlSales = measuresOverTimeSales[ measuresOverTimeSales.STORE_NBR == 155]

scaledControlSales
measureOverTimeSalesStore86 = measuresOverTimeSales.loc[ (measuresOverTimeSales.STORE_NBR == 86)]

perceDiff  = pd.merge(measureOverTimeSalesStore86, scaledControlSales, on = 'month_year', suffixes=('_trial', '_control'))

perceDiff
perceDiff['percentageDiffSales'] = abs(perceDiff.totSales_control - perceDiff.totSales_trial)/ perceDiff.totSales_control

diffDuringPreTrial = perceDiff[perceDiff.month_year < '2019-02'].percentageDiffSales

diffDuringPreTrial
stdDev_pretrial = np.std(diffDuringPreTrial)

degreesOfFreedom = 7
t_critic = 1.895
t_calc = (perceDiff.loc[perceDiff.month_year.between('2019-02', '2019-04'),'percentageDiffSales'] - 0) / stdDev_pretrial

t_calc
t_calc > t_critic
scalingFactorForControlnCustomers = preTrialMeasures[preTrialMeasures.STORE_NBR == 86].nCustomers.sum() / preTrialMeasures[preTrialMeasures.STORE_NBR == 155].nCustomers.sum()

scalingFactorForControlnCustomers



measureOverTimenCustomers = measures_over_time.copy()

measureOverTimenCustomers.loc[ measureOverTimenCustomers.STORE_NBR == 155, 'nCustomers'] = measureOverTimenCustomers.loc[ measureOverTimenCustomers.STORE_NBR == 155, 'nCustomers'] * scalingFactorForControlnCustomers

scaledControlnCustomers = measureOverTimenCustomers[ measureOverTimenCustomers.STORE_NBR == 155]

scaledControlnCustomers
measureOverTimenCustomersStore86 = measureOverTimenCustomers.loc[ (measureOverTimenCustomers.STORE_NBR == 86)]

percentageDiff  = pd.merge(measureOverTimenCustomersStore86, scaledControlnCustomers, on = 'month_year', suffixes=('_trial', '_control'))

percentageDiff
percentageDiff['percentageDiffnCustomers'] = abs(percentageDiff.nCustomers_control - percentageDiff.nCustomers_trial)/ percentageDiff.nCustomers_control

diffDuringPreTrial = percentageDiff[percentageDiff.month_year < '2019-02'].percentageDiffnCustomers

diffDuringPreTrial
stdDev_pretrial = np.std(diffDuringPreTrial)

degreesOfFreedom = 7
t_calc = (percentageDiff.loc[percentageDiff.month_year.between('2019-02', '2019-04'),'percentageDiffnCustomers'] - 0) / stdDev_pretrial

t_calc
t_calc > t_critic