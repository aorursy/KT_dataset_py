# Imports

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from collections import defaultdict

sns.set_context('notebook')

from collections import Counter



import copy



from sklearn.preprocessing import MinMaxScaler
# Reading data

df = pd.read_csv('../input/NationalNames.csv')
df.head()
print ('Data year ranges from {} to {}'.format(min(df['Year']),max(df['Year'])))
# Assign decades to Year

df['Decade'] = df['Year'].apply(lambda x: x - (x%10))

df.tail()
df_pivot = df.pivot_table(values='Count',index=['Decade','Name','Gender'],aggfunc='sum')

new_df = pd.DataFrame()

new_df['Decade'] = df_pivot.index.get_level_values('Decade')

new_df['Name'] = df_pivot.index.get_level_values('Name')

new_df['Gender'] = df_pivot.index.get_level_values('Gender')

new_df['Count'] = df_pivot.values
# This new dataframe contains data aggregated into decades

new_df.head()
decadeList = list(new_df['Decade'].unique())

boys_percentileList = []

girls_percentileList = []



# Making temp copies so I can add a column to the dataframe

boys_df = new_df[new_df['Gender']=='M'].copy()

girls_df = new_df[new_df['Gender']=='F'].copy()



for i in decadeList:

    scaler = MinMaxScaler()

    boys_percentileList.extend(scaler.fit_transform(boys_df[boys_df['Decade']==i][['Count']]))

    girls_percentileList.extend(scaler.fit_transform(girls_df[girls_df['Decade']==i][['Count']]))



boys_df['decade_percentile'] = boys_percentileList

girls_df['decade_percentile'] = girls_percentileList



new_df = boys_df.append(girls_df)



# MinMaxScaler() returns a list of np.arrays, let's change that into a float 

new_df['decade_percentile'] = new_df['decade_percentile'].apply(lambda x: float(x) * 100)

new_df.sort_index(inplace=True)

new_df.head()



# Just cleaning some memory

del boys_df

del girls_df
plt.plot(new_df[(new_df['Name']=='John')&(new_df['Gender']=='M')]['Decade'],

         new_df[(new_df['Name']=='John')&(new_df['Gender']=='M')]['decade_percentile'])
# Listing out the most popular names through the past century

new_df[new_df['decade_percentile']>=99.0]
# Showing all names with less than 1% popularity

new_df[new_df['decade_percentile'] < 1]
plt.figure()

sns.distplot(new_df[(new_df['Gender']=='M')]['decade_percentile'], bins=100)

plt.xlim(xmin=0,xmax=100)

plt.title('Boys Name Popularity Distribution')



plt.figure()

sns.distplot(new_df[(new_df['Gender']=='F')]['decade_percentile'], bins=100)

plt.xlim(xmin=0,xmax=100)

plt.title('Girls Name Popularity Distribution')          



plt.show()
def nameFilter(decade,gender,lowerBound,upperBound,startsWith=None):

    '''

        This function helps you find rare/common baby names!

        Inputs:

            decade : integer = Decade as a 4 digit number, e.g. 1980.

            gender : string = Gender as a single letter string, e.g. 'M' for Male

            lowerBound: float = Lower percentage of the names you want to query, e.g. 25 for 25%, NOT 0.25

            upperBound: float = Upper percentage of the names you want to query

            startsWith: str = (Optional) Single letter representing the starting letter of a name

        Returns:

            A dataframe slice fitting your parameters.

    '''

    if upperBound < lowerBound:

        raise ValueError('lowerBound needs to be less than upperBound')

    

    if startsWith != None:

        result_df = new_df[(new_df['Decade'] == decade) &

                           (new_df['Gender'] == gender) &

                           (new_df['decade_percentile'] >= lowerBound) &

                           (new_df['decade_percentile'] <= upperBound) &

                           (new_df['Name'].str[0]==startsWith.upper())

                          ]

    else:

        result_df = new_df[(new_df['Decade'] == decade) &

                           (new_df['Gender'] == gender) &

                           (new_df['decade_percentile'] >= lowerBound) &

                           (new_df['decade_percentile'] <= upperBound) 

                          ]

    return result_df
nameFilter(decade=1980, gender='M', lowerBound=50, upperBound=100, startsWith='C')