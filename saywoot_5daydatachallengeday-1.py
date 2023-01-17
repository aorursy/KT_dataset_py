# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 







import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import mpl_toolkits



%matplotlib inline 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))





bar = pd.read_csv("../input/bar_locations.csv")

party = pd.read_csv("../input/party_in_nyc.csv")



#Latitude Longitude



# <class 'pandas.core.frame.DataFrame'>

# RangeIndex: 2440 entries, 0 to 2439

# Data columns (total 7 columns):

# Location Type    2440 non-null object

# Incident Zip     2440 non-null float64

# City             2440 non-null object

# Borough          2440 non-null object

# Latitude         2440 non-null float64

# Longitude        2440 non-null float64

# num_calls        2440 non-null int64

bar.info()



# <class 'pandas.core.frame.DataFrame'>

# RangeIndex: 225414 entries, 0 to 225413

# Data columns (total 8 columns):

# Created Date     225414 non-null object

# Closed Date      224619 non-null object

# Location Type    225414 non-null object

# Incident Zip     224424 non-null float64

# City             224424 non-null object

# Borough          225414 non-null object

# Latitude         223946 non-null float64

# Longitude        223946 non-null float64

party.info()



ax1 = party.plot(kind='scatter', x='Latitude', y='Longitude',alpha = 0.2,color = '#89C4F4', label = 'party')

ax1.set_facecolor('#F2F1EF')

ax2 = bar.plot(kind='scatter', x='Latitude', y='Longitude',alpha = 0.2,color = '#013243', ax=ax1, label = 'bar')



plt.legend()

plt.savefig('scatter.png', format='png', dpi=1200)



plt.show()
