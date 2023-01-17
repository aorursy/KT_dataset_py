# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

import os 

import glob 

%matplotlib inline

%config InlineBackend.figure_formats = ['svg']

sns.set_style("white")
full_zika_df = pd.read_csv(os.path.join('..', 'input', 'cdc_zika.csv'),low_memory=False)
full_zika_df.info()
full_zika_df['location'].head(2)
full_zika_df.columns.tolist()
full_zika_df['country'] = full_zika_df['location']

full_zika_df['country'] = full_zika_df['country'].astype(str)

full_zika_df['country'] = full_zika_df['country'].apply(lambda x: pd.Series(x.split('-')))

full_zika_df.country.unique()
## Brazil Data Frame

mask_brazil = full_zika_df['country'] == "Brazil"

brazil_frame = full_zika_df[mask_brazil]

brazil_frame.country.unique()
## Removing colums I dont need

## Brazil  reports locations at the state level

brazil_frame = brazil_frame.drop(['location_type', 'data_field_code', 'time_period_type', 'time_period', 'unit'], axis=1)
## Creating state/city column 

foo = lambda x: pd.Series([i for i in reversed(x.split('-'))])

brazil_frame['location'] = brazil_frame.location.apply(foo)

brazil_frame.rename(columns={'location':'state_city'}, inplace=True)
## Cleaning the state city column

brazil_frame.state_city = brazil_frame.state_city.map(lambda x: x.replace('_',' '))

brazil_frame.state_city.value_counts()
brazil_frame = brazil_frame[brazil_frame.state_city != 'Brazil']

brazil_frame.state_city.value_counts()
## Changing the order of the columns

def order(frame,var):

    varlist =[w for w in frame.columns if w not in var]

    frame = frame[var+varlist]

    return frame



brazil_frame = order(brazil_frame,['report_date','country', 'state_city'])
## Keeping microcephaly_confirmed and microcephaly_fatal_confirmed

mask_conf = (brazil_frame['data_field'] == "microcephaly_confirmed") | (brazil_frame['data_field'] == "microcephaly_fatal_confirmed")

brazil_frame = brazil_frame[mask_conf]

print (brazil_frame.data_field.value_counts())

print ("++++++++++++++")
def int_replace(x):

    try:

        return int(x)

    except:

        return np.nan_to_num(x)



brazil_frame.value = brazil_frame.value.map(int_replace)

brazil_frame.value = brazil_frame.value.astype(int)
print ("Shape of data:", brazil_frame.shape)

print ("++++++++++++++\n")

print ("Missing values:\n")

print (brazil_frame.isnull().sum())

print ("++++++++++++++\n")

print (brazil_frame.info())

print ("++++++++++++++\n")
## Reshaping the Data

brazil_zika = pd.pivot_table(brazil_frame,

                             index=['country','state_city','report_date'],

                             columns=['data_field'],values=['value'],

                             aggfunc=sum)

brazil_zika = brazil_zika['value'].reset_index()
## Making Report date the index

print (brazil_zika.report_date.dtype )

print ("++++++++++++++")

brazil_zika.sort_values("report_date", inplace=True)

brazil_zika.set_index("report_date", inplace=True)

brazil_zika.index = brazil_zika.index.to_datetime()
## Now creating a year, month and day column

brazil_zika['year'] = brazil_zika.index.year

brazil_zika['month'] = brazil_zika.index.month

brazil_zika['day'] = brazil_zika.index.day
brazil_zika['microcephaly_fatal_confirmed'].fillna(0, inplace=True)

brazil_zika['microcephaly_confirmed'].fillna(0, inplace=True)

brazil_zika['microcephaly_confirmed'] = brazil_zika.microcephaly_confirmed.astype(int)

brazil_zika['microcephaly_fatal_confirmed'] = brazil_zika.microcephaly_fatal_confirmed.astype(int)
print (brazil_zika.info())

print ("++++++++++++++\n")

print (brazil_zika.shape)
## Creting Data Frame for graphs by state and date

brazil_zika_ = brazil_zika.reset_index()

brazil_zika_ = brazil_zika_.rename(columns={'index':'date'})

brazil_gb = brazil_zika_.groupby(['state_city','date']).sum()
brazil_gb = brazil_gb.reset_index()
print (brazil_gb.columns)

print (brazil_gb.head(3))
def plot_num_ts(df, seq_col, seq):

    cat = df[seq_col].unique()

    fig = plt.figure(figsize=(12,8))

    ax = fig.gca()

    for i in cat:

        df_ = df[df[seq_col] == i]

        df_ = df_.set_index('date')

        df_ = df_.resample(seq).sum()

        ax.plot(df_['microcephaly_confirmed'], lw=1.5, linestyle='dashed', marker='o',

                markerfacecolor='red', markersize=5,

                label="Confirmed cases in %s"%i)

        

    plt.legend(loc='best', fontsize=10, bbox_to_anchor=(0., 1.02, 1., .102),ncol=3)

    plt.xlabel('\nTime progression in %s'%seq)

    plt.ylabel("Total Microcephaly Cases\n")

    plt.show()
plot_num_ts(brazil_gb, 'state_city', 'W')
def plot_num_ts(num, seq):

    

    num = num.sort_index(ascending = False)

    num = num.resample(seq).sum()

    mask = np.isfinite(num['microcephaly_confirmed'])

    fig = plt.figure(figsize=(12,5))

    ax = fig.gca()

    ax.plot(num['microcephaly_confirmed'][mask], c='y', lw=3, linestyle='dashed', marker='o',

                markerfacecolor='red', markersize=5,

            label="Brazil - number of microcephaly confirmed cases (2016) per %s"%seq)

    plt.legend(loc='best')

    

    plt.xlabel('\nTime progression in %s'%seq)

    plt.ylabel("Microcephaly confirmed cases\n")

    plt.show()
plot_num_ts(brazil_zika, "W")
grouped = brazil_zika.groupby(['state_city'])

##

grouped_val_1 = grouped['microcephaly_confirmed']

category_group_1=grouped_val_1.sum()

##

grouped_val_2 = grouped['microcephaly_fatal_confirmed']

category_group_2=grouped_val_2.sum()

##





plt.figure(figsize=(10,7))

braz_microc_conf_1 = category_group_1.plot(kind='bar', color='green', 

                                           alpha=0.7, 

                                           title= "Microcephaly Confirmed Cases 2016\n")

braz_microc_conf_1.set_xlabel("States\n")

braz_microc_conf_1.set_ylabel("Sum of Cases\n")

plt.show()
print (category_group_2.tail(4))

print ("++++++++")



plt.figure(figsize=(10,7))

braz_microc_conf_2 = category_group_2.plot(kind='bar', color='orange', alpha=0.4, 

                                           title= "Microcephaly Confirmed Fatal Cases 2016\n")

braz_microc_conf_2.set_xlabel("States\n")

braz_microc_conf_2.set_ylabel("Sum of Cases\n")

plt.show()
grouped = brazil_zika.groupby(['month'])

##

grouped_val = grouped['microcephaly_confirmed']

category_group=grouped_val.sum()

#

grouped_val_1 = grouped['microcephaly_fatal_confirmed']

category_group_1=grouped_val_1.sum()





plt.figure(figsize=(10,6))



braz_graph= category_group.plot(kind='bar', color='green', alpha=0.7, 

                                title= "Microcephaly Cases 2016\n", 

                                label="microcephaly_confirmed")

braz_graph_1= category_group_1.plot(kind='bar', color='red', alpha=0.7)



braz_graph.set_xlabel("Months\n")

braz_graph.set_ylabel("Total Cases\n")



plt.tick_params(labelsize=14)

plt.legend(loc='upper left')

plt.show()