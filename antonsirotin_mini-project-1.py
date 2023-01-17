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
#importing additional modules to help plot data and read data from pandas

import pandas_datareader.data as pdr

import matplotlib.pyplot as plt



#importing modules to help calculate the coefficient

from numpy import cov

from scipy.stats import pearsonr



#loading the csv file into a var

nba = pd.read_csv("../input/ppg-stats/NBA Stats - PPG.csv")
#let's see how the data set looks

nba
#defining variable to hold columns

pts = nba['PTS']

mins = nba['MIN']



#defines function that calculates PPM

def ppmCalc(pts, mins):

    ppm = pts/mins

    return ppm



#loads PPM column with PPM calculations

nba['PPM'] = ppmCalc(pts, mins)



#quantitatively describes the distribution of PPM

nba['PPM'].describe()
#re-sorts dataframe by PPM, descending

nba.sort_values(by=['PPM'], ascending=False)
#function to calculate categories

def scorerQual(ppm):

    if ppm >= 0.6:

        return 'Elite'

    elif ppm >= 0.46 and ppm < 0.6:

        return 'Middle of the Pack'

    else:

        return 'Over rated'

    

#applies the scorerQual function through the values in the PPM collumn, puts them into categories, then plots a simple pie graph

nba['PPM'].apply(scorerQual).value_counts().plot(kind='bar')
nba[['MIN', 'PPM']].plot(kind='scatter', x='MIN', y='PPM', alpha=1)
#defining variables to help

x = nba['MIN']

y = nba['PPM']



#calculating covarience using numpy functions imported above

covarience = cov(x, y)

print('The covarience matrix of MPG and PPG is:', covarience)



#calculates the correlation coefficient

corr, _ = pearsonr(x, y)

print('\n\nThe correlation Coefficient is: %.3f' % corr)