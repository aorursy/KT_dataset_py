# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



import datetime



patient = pd.read_csv('../input/coronavirusdataset/patient.csv')

trend = pd.read_csv('../input/coronavirusdataset/trend.csv')

time = pd.read_csv('../input/coronavirusdataset/time.csv')

route = pd.read_csv('../input/coronavirusdataset/route.csv')

patient = patient.query('confirmed_date < "2020-02-14"')# and infection_reason.str.con not like "%visit"')

for i in ['visit', 'residence']:

    patient = patient[~patient['infection_reason'].str.contains(i)]

patient['age'] = 2020 - patient['birth_year'] + 1

patient['age_group'] = patient['age'] // 10

patient['age_group'] = [str(a).replace('.','') for a in patient['age_group']]



print('Female 20, 40, 50 age rank top with significant difference.')

print('What happens to them? It might be a next topic of my analysis')

plt.figure(figsize = (15,8))

ax = sns.countplot(patient['age_group'], order = patient['age_group'].value_counts().sort_index().index)

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

plt.show()

## floating population data(only January)

fp_01 = pd.read_csv("../input/seoul-floating-population-2020/fp_2020_01_english.csv")
# trim data

fp_01['date'] = fp_01['date'].apply(lambda x: datetime.datetime.strptime(x, "%Y.%m.%d").date()).astype('str')

fp_01['date'] = fp_01['date'].apply(lambda x: x[8:]) ## only use day



fp_01 = fp_01.sort_values(['date', 'hour', 'birth_year', 'sex'])  ## this data is not sorted.

fp_01.reset_index(drop= True, inplace = True)

fp_01.head()
def make_brith_hour_plot(date1, date2, city):

    

    if city == 'all district':        

        gan17 = fp_01[(fp_01['date'] == date1)]

        gan31 = fp_01[(fp_01['date'] == date2)]

    else:

        gan17 = fp_01[(fp_01['date'] == date1) & (fp_01['city'] == city)]

        gan31 = fp_01[(fp_01['date'] == date2) & (fp_01['city'] == city)]



    gan17 = pd.DataFrame(gan17.groupby(['hour', 'birth_year'])['fp_num'].sum())

    gan17.reset_index(inplace = True)



    gan31 = pd.DataFrame(gan31.groupby(['hour', 'birth_year'])['fp_num'].sum())

    gan31.reset_index(inplace = True)



    fig, ax = plt.subplots(1,2, figsize = (18,8),  gridspec_kw={'wspace': 0.2})

    

    fig.suptitle('{} : Changes by age group and hour'.format(city), fontsize = 18)

    fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, wspace=0.05)

    

    t = ax[0].scatter(x=gan17['hour'], y=gan17['birth_year'], c=gan17['fp_num'], s= 100, cmap=plt.cm.RdYlBu_r)

    ax[0].set_title("{}day".format(date1), fontsize=15)

    ax[0].set_xlabel('hour', fontsize=13)

    ax[0].set_ylabel('birth_year', fontsize=13)    

    plt.colorbar(t, ax = ax[0])



    t2 = ax[1].scatter(x=gan31['hour'], y=gan31['birth_year'], c=gan31['fp_num'], s= 100, cmap=plt.cm.RdYlBu_r)

    ax[1].set_title("{}day".format(date2), fontsize=15)

    ax[1].set_xlabel('hour', fontsize=13)

#     ax[1].set_ylabel('birth_year', fontsize=13)    

    cbar1  = plt.colorbar(t2, ax = ax[1])

    cbar1.set_label('fp_num', fontsize=13)        

    plt.show()



def make_brith_hour_diff_plot(date1, date2, city):

    

    if city == 'all district':        

        gan17 = fp_01[(fp_01['date'] == date1)]

        gan31 = fp_01[(fp_01['date'] == date2)]

    else:

        gan17 = fp_01[(fp_01['date'] == date1) & (fp_01['city'] == city)]

        gan31 = fp_01[(fp_01['date'] == date2) & (fp_01['city'] == city)]

    

    gan17_no_groupby = pd.DataFrame(gan17.groupby(['hour'])['fp_num'].sum())

    gan17_no_groupby.reset_index(inplace = True)



    gan31_no_groupby = pd.DataFrame(gan31.groupby(['hour'])['fp_num'].sum())

    gan31_no_groupby.reset_index(inplace = True)

    

    gan17_groupby = pd.DataFrame(gan17.groupby(['hour', 'birth_year'])['fp_num'].sum())

    gan17_groupby.reset_index(inplace = True)



    gan31_groupby = pd.DataFrame(gan31.groupby(['hour', 'birth_year'])['fp_num'].sum())

    gan31_groupby.reset_index(inplace = True)

    

    fp_num_diff = gan17_groupby.iloc[:,-1:] -  gan31_groupby.iloc[:,-1:]

    axix_=gan17_groupby.iloc[:,:-1]

    df = pd.concat([axix_, fp_num_diff ], axis = 1)



        

    fig, ax = plt.subplots(1,2, figsize = (18,8),  gridspec_kw={'wspace': 0.2, 'hspace': 0.4})

    fig.suptitle('{} : Diff between (before corona) and (after_corona)'.format(city), fontsize = 18)

    fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, wspace=0.05)

    t = ax[0].scatter(x=df['hour'], y=df['birth_year'], c=df['fp_num'], s= 200, cmap=plt.cm.RdYlBu_r, vmax =7000) #, vmax)

    ax[0].set_title("{}day-{}day diff, groupby hour and age_group".format(date1, date2), fontsize=15)

    ax[0].set_xlabel('hour', fontsize=13)

    ax[0].set_ylabel('birth_year', fontsize=13)    

    cbar1 = plt.colorbar(t, ax = ax[0])

    cbar1.set_label('fp_num', fontsize=13)



    

    

    

    sns.lineplot(data=gan17_no_groupby, x='hour', y='fp_num', color='green', ax=ax[1]).set_title('fp_num', fontsize=16)

    sns.lineplot(data=gan31_no_groupby, x='hour', y='fp_num', color='purple', ax=ax[1]).set_title('fp_num', fontsize=16)

    ax[1].set_title("{}day-{}day diff, total".format(date1, date2), fontsize=15)

    ax[1].set_xlabel('hour', fontsize=13)

    ax[1].set_ylabel('total fp_num', fontsize=13)    

    ax[1].legend([date1, date2])

    plt.show()

    

    

    plt.show()

    
make_brith_hour_plot('17', '31', 'all district')
make_brith_hour_diff_plot('17', '31', 'all district')
plt.figure(figsize = (15,8))

ax = sns.countplot(patient['age_group'], order = patient['age_group'].value_counts().sort_index().index)

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

plt.show()
route[(route['date'] < '2020-02-01') & ( route['province'] == 'Seoul')].groupby('city')['patient_id'].nunique()
make_brith_hour_plot('17', '31', 'Gangnam-gu')
make_brith_hour_diff_plot('17', '31', 'Gangnam-gu')
make_brith_hour_plot('17', '31', 'Jongno-gu')
make_brith_hour_diff_plot('17', '31', 'Jongno-gu')
make_brith_hour_plot('17', '31', 'Jung-gu')
make_brith_hour_diff_plot('17', '31', 'Jung-gu')
make_brith_hour_plot('17', '31', 'Jungnang-gu')
make_brith_hour_diff_plot('17', '31', 'Jungnang-gu')