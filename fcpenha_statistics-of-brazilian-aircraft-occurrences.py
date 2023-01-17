# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# from subprocess import check_output

# print(check_output(["ls", "../input/data"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



# Basic libraries

import numpy as np

import pandas as pd

from scipy import stats



# File related

import zipfile

from subprocess import check_output



# Machine Learning

# import sklearn

# from sklearn.model_selection import train_test_split

# from sklearn.linear_model import LinearRegression, Ridge

# import tensorflow as tf



# Plotting with matplotlib

import matplotlib

import matplotlib.pyplot as plt



from pandas.tools.plotting import parallel_coordinates

from pandas.tools.plotting import andrews_curves

from pandas.tools.plotting import radviz



plt.style.use('fivethirtyeight')



plt.rcParams['axes.labelsize'] = 20

plt.rcParams['axes.titlesize'] = 20

plt.rcParams['xtick.labelsize'] = 18

plt.rcParams['ytick.labelsize'] = 18

plt.rcParams['legend.fontsize'] = 14
from subprocess import check_output

print(check_output(['ls', '../input/']).decode('utf8'))
aircrafts = pd.read_csv('../input/aircrafts.csv',

                        na_values=['****', '***',''],

                        header=0,

                        encoding='latin-1'

                       )



# Exchange white spaces by underlines, in column names

aircrafts.columns = [c.replace(' ', '_') for c in aircrafts.columns]



# Scores for damage level

aircrafts.replace(

    to_replace=['UNKNOWN', 'NONE', 'LIGHT', 'SUBSTANTIAL', 'DESTROYED'],

    value=[np.nan, 0, 1, 2, 3],

    inplace=True

    )



# Drop all rows for which 'damage_level' & 'engines_amount' are NAN

aircrafts.dropna(axis=0,

                 how='any',

                 subset=['damage_level', 'engines_amount'],

                 inplace=True

                 )



aircrafts.head()
occurrences = pd.read_csv('../input/occurrences.csv',

                        na_values=['****', '***',''],

                        header=0,

                        encoding='latin-1'

                       )



# Exchange white spaces by underlines, in column names

occurrences.columns = [c.replace(' ', '_') for c in occurrences.columns]



# Read year from 'occurrence_day' column. 

# Create a new column called 'occurrence_year'

occurrences['occurrence_year'] = pd.to_datetime(

                                    occurrences['occurrence_day'],

                                    format='%Y/%m/%d'

                                    ).dt.year



occurrences.head()
# Classification



print(occurrences['classification'].unique())
print('Total: ' +

      str(occurrences['classification'].count())

     )



print('SERIOUS INCIDENT: ' +

      str(occurrences['classification'][occurrences['classification'] == 'SERIOUS INCIDENT'].count())

     )



print('ACCIDENT: ' +

      str(occurrences['classification'][occurrences['classification'] == 'ACCIDENT'].count())

     )
# Type of occurrences



print(occurrences['type_of_occurrence'].unique())
for occ in occurrences['type_of_occurrence'].unique():

    

    print(occ + ' :' +

          str(occurrences['type_of_occurrence'][

              occurrences['type_of_occurrence'] == occ

              ].count()

             )

         )
# %%% Normalization %%%



occurrences_freq_dict = {}



for occ in occurrences['type_of_occurrence'].unique():

    

    occurrences_freq_dict[occ] = occurrences['type_of_occurrence'][

                                occurrences['type_of_occurrence'] == occ

                                ].count() / occurrences['type_of_occurrence'].count()

    

occurrences_freq = pd.Series(occurrences_freq_dict)



occurrences_freq.head(10)
for occ in occurrences['fu'].unique():

    

    print(str(occ) + ' :' +

          str(occurrences['fu'][

              occurrences['fu'] == occ

              ].count()

             )

         )
# %%% Normalization %%%



states_freq_dict = {}



for occ in occurrences['fu'].unique():

    

    states_freq_dict[occ] = occurrences['fu'][

                                occurrences['fu'] == occ

                                ].count() / occurrences['fu'].count()

    

states_freq = pd.Series(states_freq_dict)



states_freq.head(10)
fig, axes = plt.subplots(figsize=(10.,6.))



states_freq.sort_values(ascending=False).plot(kind='bar')



axes.set_xlabel('State')

axes.set_ylabel('Flight occurrence rate')

                   

plt.show()

plt.close()
year_freq_dict = { }

year_freq_dict['all'] = {}



for s in occurrences['fu'].unique():

    year_freq_dict[s] = {}



for occ in occurrences['occurrence_year'].unique():

    

    year_freq_dict['all'][occ] = occurrences['occurrence_year'][

                                    occurrences['occurrence_year'] == occ

                                    ].count()



    for s in occurrences['fu'].unique():

        

        year_freq_dict[s][occ] = occurrences['occurrence_year'][

                                        (occurrences['occurrence_year'] == occ) &

                                        (occurrences['fu'] == s)

                                        ].count()



year_freq = pd.DataFrame(year_freq_dict)



year_freq.head()
fig, axes = plt.subplots(figsize=(10.,6.))



year_freq['all'].plot.area(label='total')

year_freq['SP'].plot.area(label='SP')

year_freq['RS'].plot.area(label='RS')

year_freq['EX'].plot.area(label='abroad')



axes.set_xlabel('State')

axes.set_ylabel('Occurrences in 10 years')

          

axes.legend(loc='upper left')



plt.show()

plt.close()
manuf_year_freq_dict = {}



manuf_year_lst = aircrafts['year_manufacture'].unique()



# Exclude 'nan' and '0'

manuf_year_lst = manuf_year_lst[~np.isnan(manuf_year_lst)]

manuf_year_lst[ manuf_year_lst != 0.]



for y in manuf_year_lst:

    

    manuf_year_freq_dict[y] = aircrafts['year_manufacture'][

                                        aircrafts['year_manufacture'] == y

                                        ].count()

    

manuf_year_freq = pd.Series(manuf_year_freq_dict)
fig, axes = plt.subplots(figsize=(10.,6.))



manuf_year_freq.plot.area(alpha=0.5)



axes.set_xlabel('Year of aircraft manufacture')

axes.set_ylabel('Occurrences in 10 years')

axes.set_xlim(xmin=1936) 



plt.show()

plt.close()
label_lst = ['damage_level',

             'engines_amount',

             'seatings_amount',

             'takeoff_max_weight_(Lbs)',

             'year_manufacture'

            ]



fig, axes = plt.subplots(nrows=4, sharex=True, figsize=(10.,10.))

plt.subplots_adjust(hspace=0.4)



for j in range(1, len(label_lst)):

  

    axes[j-1].scatter(

                aircrafts[label_lst[0]],

                aircrafts[label_lst[j]],

                color='red',

                alpha=0.5

                )

    

    axes[j-1].set_ylabel(label_lst[j])



axes[0].set_xlim([-1.,4.])

axes[3].set_ylim([1935, 2016])



axes[3].set_xlabel(label_lst[0])



plt.show()

plt.close()
def plot_heatmap(df):

    

    import seaborn as sns

    

    fig, axes = plt.subplots()



    sns.heatmap(df, annot=True)



    plt.xticks(rotation=90)

    

    plt.show()

    plt.close()

    

plot_heatmap(aircrafts[label_lst].corr(method='pearson'))
# Feature scaling, or normalization

def z_score_norm(df, feat):

    

    dff = df.copy(deep=True)

    

    dff[feat] = (

                df[feat] - df[feat].mean()

                ) / (

                    df[feat].max() - df[feat].min()

                    )

    

    return dff



aircrafts_norm = z_score_norm(aircrafts[label_lst], label_lst[1:])



fig, axes = plt.subplots(figsize=(10.,6.))



parallel_coordinates(aircrafts_norm, 'damage_level')



plt.show()

plt.close()
andrews_curves(aircrafts_norm, 'damage_level')
radviz(aircrafts_norm, 'damage_level')