import pandas as pd

import numpy as np

import seaborn as sns

import pdb

import os



%matplotlib inline

import matplotlib.pyplot as plt



import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv('../input/inpatientCharges.csv')

df.head(2)
df.describe()
df.info()
df['DRG Definition'].unique().shape

df['Provider Id'].unique().shape

df['Provider City'].unique().shape
df.columns = [column.strip() for column in df.columns]
for column in ['Average Covered Charges', 'Average Total Payments', 'Average Medicare Payments']:

    df[column] = df[column].map(lambda x: x[1:])

    df[column] = pd.to_numeric(df[column])
agg_columns = ['mean', 'median', 'var', 'std', 'count', 'min', 'max']

groupby_drg = df[['DRG Definition', 'Average Total Payments']].groupby(by='DRG Definition').agg(agg_columns)

groupby_drg.columns = [header + '-' + agg_column 

                       for header, agg_column in zip(groupby_drg.columns.get_level_values(0), agg_columns)]

groupby_drg.columns = groupby_drg.columns.get_level_values(0)
groupby_drg.reset_index(inplace=True)

groupby_drg['Average Total Payments-range'] = groupby_drg['Average Total Payments-max'] - groupby_drg['Average Total Payments-min']

groupby_drg.head()
def plt_setup(_plt):

    _plt.tick_params(

    axis='x',          # changes apply to the x-axis

    which='both',      # both major and minor ticks are affected

    bottom='off',      # ticks along the bottom edge are off

    top='off',         # ticks along the top edge are off

    labelbottom='off')
plt.figure(figsize=(20,5))

sns.barplot(x='DRG Definition', y='Average Total Payments-mean', 

            data=groupby_drg.sort_values('Average Total Payments-mean'))

plt_setup(plt)

plt.title('Mean Average Total Payments by DRG', fontsize=16)

plt.ylabel('Mean of Average Total Payments', fontsize=16)
plt.figure(figsize=(20,5))

sns.barplot(x='DRG Definition', y='Average Total Payments-var', 

            data=groupby_drg.sort_values('Average Total Payments-mean'))

plt_setup(plt)

plt.title('Mean Average Total Payments Variance by DRG', fontsize=16)

plt.ylabel('Mean of Average Total Payments Variance', fontsize=16)
plt.figure(figsize=(20,5))

sns.barplot(x='DRG Definition', y='Average Total Payments-range', 

            data=groupby_drg.sort_values('Average Total Payments-var'))

plt_setup(plt)

plt.title('Mean Average Total Payments Range by DRG', fontsize=16)

plt.ylabel('Mean of Average Total Payments Range', fontsize=16)
import csv, sqlite3

os.listdir('.')
def unlock_db(db_filename):

    """Replace db_filename with the name of the SQLite database."""

    connection = sqlite3.connect(db_filename)

    connection.commit()

    connection.close()

    

unlock_db('hospital_charges.db')
conn = sqlite3.connect('hospital_charges.db')

_df = pd.read_csv('../input/inpatientCharges.csv')

_df.columns = [column.strip() for column in _df.columns]

for column in ['Average Covered Charges', 'Average Total Payments', 'Average Medicare Payments']:

    _df[column] = _df[column].map(lambda x: x[1:])

    _df[column] = pd.to_numeric(_df[column])

_df.to_sql('hospital_charges', conn, if_exists='replace', index=False)

conn.close()
conn = sqlite3.connect('hospital_charges.db')



query = '''

SELECT `DRG Definition`, `Provider State` as providerState, t.maxAvgPaymentPerDRG

FROM hospital_charges hc

INNER JOIN (

    SELECT `DRG Definition` as drg, MAX(`Average Total Payments`) as maxAvgPaymentPerDRG

    FROM hospital_charges

    GROUP BY `DRG Definition`

) t

ON hc.`DRG Definition` == t.drg AND hc.`Average Total Payments` == t.maxAvgPaymentPerDRG

ORDER BY providerState ASC

'''



cursor = conn.execute(query)

results = [record for record in cursor]

conn.close()
_df = pd.DataFrame(results, columns=['DRG', 'ProviderState', 'Payment'])
conn = sqlite3.connect('hospital_charges.db')



ranking_by_drg = dict()

for drg in df['DRG Definition'].unique():



    query = """

        SELECT `Provider State` as providerState

        FROM hospital_charges

        WHERE `DRG Definition` = '{drg}' 

        GROUP BY `Provider State`

        ORDER BY AVG(`Average Total Payments`) ASC

    """.format(drg=drg)

    

    cursor = conn.execute(query)

    ranking_by_drg[drg] = [record[0] for record in cursor]

    

conn.close()
for k,v in ranking_by_drg.items():

    while True:

        if len(v) >= 51:

            break

        v.append(None)
drg_by_id = {key.split(' - ')[0]: key for key in ranking_by_drg.keys()}

df_rank = pd.DataFrame(ranking_by_drg)
from collections import defaultdict



_df = pd.DataFrame(index=df['Provider State'].unique())

for column in df_rank.columns:

    rankings_by_state = []



    if column in ['rank']:

        continue

        

    for rank, curr_state in zip(df.index, df_rank[column]):

        rankings_by_state.append(curr_state)

    

    t = pd.DataFrame(rankings_by_state, columns=['State'])

    t['Rank'] = t.index

    t.set_index(['State'], inplace=True)

    

    _df = pd.merge(left=_df, right=t, how='left', left_index=True, right_index=True)

    

_df.columns = drg_by_id.keys()
_df.fillna(-100, inplace=True)

_df.head() #wherre -100 is not a rank but only a placeholder - shown as dark blue in the heatmap
plt.figure(figsize=(20,20))

sns.heatmap(_df, square=True, vmin=-100, cbar=False, linewidths=0.1)

plt.title("Cost Rankings Per DRG Definition (Darker Shade = More Expensive)", fontsize=20)

plt.xlabel("DRG Definition Id", fontsize=20)
conn = sqlite3.connect('hospital_charges.db')



total_discharges_by_drg = dict()

for drg in df['DRG Definition'].unique():



    query = """

        SELECT `Provider State` as providerState, SUM(`Total Discharges`) as numDischarges

        FROM hospital_charges

        WHERE `DRG Definition` = '{drg}' 

        GROUP BY `Provider State`

    """.format(drg=drg)

    

    cursor = conn.execute(query)

    total_discharges_by_drg[drg] = [(record[0], record[1]) for record in cursor]

    

conn.close()
_df = pd.DataFrame(index=df['Provider State'].unique())

for k,v in total_discharges_by_drg.items():

    t = pd.DataFrame(v, columns=['State', 'Total_Discharges'])

    t.set_index(['State'], inplace=True)

    _df = pd.merge(left=_df, right=t, how='left', left_index=True, right_index=True)

_df.columns = drg_by_id.keys()

_df.fillna(0, inplace=True)
normalized_df = _df.divide(_df.sum(axis=1), axis=0)

normalized_df.head()
normalized_df = _df.divide(_df.sum(axis=1), axis=0)



plt.figure(figsize=(20,20))

sns.heatmap(normalized_df, square=True, vmin=0, cbar=False, linewidths=0.1)

plt.title("Normalized Pct of Total Discharges Per DRG Definition (Darker Shade = Higher Proportion of Population)", fontsize=20)

plt.xlabel("DRG Definition Id", fontsize=20)
drg_by_id['563'], drg_by_id['203'], drg_by_id['177']
SELECTED_DRG =  drg_by_id['203']

SELECTED_PROVIDER_ID = None

SELECTED_DRG
_df = df[df['DRG Definition'] == SELECTED_DRG]

_df.loc[:, 'Provider State'] = _df.loc[:, 'Provider State'].astype('category')

groupby_state = _df.groupby(by='Provider State').agg(['mean', 'min', 'max'])

groupby_state.reset_index(inplace=True)
groupby_state.head(3)
plt.figure(figsize=(20,5))

sns.barplot(x='Provider State', y='Average Total Payments', data=_df)

plt.title('Avg. Cost and 95% CI for DRG = {}'.format(SELECTED_DRG), fontsize=20)

plt.xlabel('Average Cost ($)', fontsize=20)

plt.xlabel('Provider State', fontsize=20)
plt.figure(figsize=(20,5))

sns.swarmplot(x="Provider State", y='Average Total Payments', data=_df)

plt.title('Cost At Each Provider for DRG = {}'.format(SELECTED_DRG), fontsize=20)

plt.xlabel('Cost ($)', fontsize=20)

plt.xlabel('Provider State', fontsize=20)
plt.figure(figsize=(20,5))

plt.plot(range(len(groupby_state['Provider State'])), 

         groupby_state['Average Total Payments']['max'], 'r.', markersize=14, alpha=0.6)

plt.plot(range(len(groupby_state['Provider State'])), 

         groupby_state['Average Total Payments']['min'], 'b.', markersize=14, alpha=0.6)

plt.title('Min./Max. for DRG = {}'.format(SELECTED_DRG), fontsize=20)

plt.xlabel('Cost ($)', fontsize=20)

plt.xlabel('Provider State', fontsize=20)

plt.show()