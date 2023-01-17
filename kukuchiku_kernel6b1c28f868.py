# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Understanding marginal emission factor

#Average emissions refers to the amount of emissions generated over a given time, divided by the amount of energy produced in that time.

#So for specific source, according to different types of primary_fuel calculate emission_factor 

#total_emission_factor = amount of emission generation over a given time/ amount of energy produced in that time 





df = pd.read_csv('/kaggle/input/ds4g-environmental-insights-explorer/eie_data/gppd/gppd_120_pr.csv')

df.column = ['capacity_mw', 'estimated_generation_gwh', 'source', 'primary_fuel']

df1 = df[df.column]

print(df1)



df1_source = df1.source.unique()

df1 = df1.values.tolist()





def avg_emission_factor(df1, a=0, b=0, c=0, d=0):

    for i in range(len(df1_source)):

        for j in range(len(df1)):

            if df1[j][3] == df1_source[i]:

                a += int(df1[j][0])        # emission_generated_per_source

                b += int(df1[j][1])        # energy_produced_per_source

        c += a                             # total_emission_generated

        d += b                             # total_energy_produced

    avg_emission_factor = c/d

    return avg_emission_factor

    

avg_marginal_emission = avg_emission_factor

print(avg_marginal_emission)