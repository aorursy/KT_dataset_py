#!/usr/bin/env python3

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline



def plot(df, f):

    mfg = df.groupby('Gender')

    mean_income = mfg['Income'].mean()

    print("Plotting for:", f)

    print(mean_income)



    # Let's now do some ethnic minortity plotting.

    mfg_ethnic = df.groupby(['Gender', 'IsEthnicMinority'])

    mfg_ethnic_im = mfg_ethnic['Income'].mean()

    print(mfg_ethnic['ID.x'].count())  # Debug

    mfg_ethnic_im.sort_values(ascending=True, inplace=True)

    print(mfg_ethnic_im)

    ax = mfg_ethnic_im.plot(kind='barh')

    ax.set_xlabel('Average Income')

    ax.get_figure().savefig(('/tmp/'+f+'.pdf'), bbox_inches='tight')

    plt.show()







df = pd.read_csv('../input/2016-FCC-New-Coders-Survey-Data.csv')

# What control variables should we use?

df['IsEthnicMinority'] = df['IsEthnicMinority'].map(lambda x:

                                                    'Yes'

                                                    if (x == 1.0)

                                                    else 'No')

# Just the USA for now and small and large cities

df = df[df['CountryLive'] == 'United States of America']

df = df[df['SchoolMajor'] == 'Computer Science']

df = df[df['SchoolDegree'] == 'bachelor\'s degree']

# Plot for the larger cities

plot(df[df['CityPopulation'] == 'more than 1 million'], 'large_cities')
# Plot for smaller cities

plot(df[df['CityPopulation'] == 'between 100,000 and 1 million'],

         'smaller_cities')