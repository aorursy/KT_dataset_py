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
df = pd.read_csv('/kaggle/input/perseus-galaxies-cluster-in-sdss-dr15/Skyserver_SQL11_25_2019 9_13_47 AM.csv')
df.head()
import seaborn as sb



sb.distplot(df['z']).set_title('Redshift')

print('mean: ',df['z'].mean())
df = df.loc[df['z'] < 0.050]

df = df.loc[df['z'] > 0.010]



df.describe()
sb.distplot(df['z'], bins = 5, kde=False).set_title('Redshift')

print('mean distance: ',df['z'].mean())



df['z'].size
df['r_deg'] = np.sqrt((df['ra'] - 49.9467)**2 + (df['dec'] - 41.5131)**2)

df
sb.scatterplot(x="r_deg", y="z", data=df).set_title('Redshift and distance from centre')
df['u-r'] = df['psfMag_u']-df['psfMag_r']
blue = df[df['u-r'] >= 2.2]

red = df[df['u-r'] < 2.2]



sb.scatterplot(x="psfMag_r", y="u-r", data=red).set_title('CMD')

sb.scatterplot(x="psfMag_r", y="u-r", data=blue)
sb.distplot(df['psfMag_r']);
import matplotlib.pyplot as plt 



alpha=-1.35

log_M0=19

phi=5.96E-11

def schechter_fit(logM):

    schechter = phi*(10**((alpha+1)*(logM-log_M0)))*(np.e**(-pow(10,logM-log_M0)))

    return schechter
y = schechter_fit(df['psfMag_r'].sort_values(ascending = True))

y_i = schechter_fit(df['psfMag_r'])



plt.plot(df['psfMag_r'].sort_values(ascending = True), np.log(y.real), 'b', label="Re Part")

plt.scatter(df['psfMag_r'], np.log(y_i.real))

plt.title('Schechter Luminosity Function')

plt.xlabel('psfMag_r')

plt.ylabel('log $\phi *$')