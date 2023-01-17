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
PGA = pd.read_csv('../input/pga-tour-20102018-data/PGA_Data_Historical.csv')
PGA_RORS = PGA[PGA['Player Name']=='Rory McIlroy']

PGA_DJ = PGA[PGA['Player Name']=='Dustin Johnson'] 

PGA_JasonDay = PGA[PGA['Player Name']=='Jason Day']

PGA_ZachJohnson = PGA[PGA['Player Name']=='Zach Johnson']
PGA_RORS = PGA_RORS.drop(columns = 'Player Name')

PGA_DJ = PGA_DJ.drop(columns = 'Player Name')

PGA_JasonDay = PGA_JasonDay.drop(columns = 'Player Name')

PGA_ZachJohnson = PGA_ZachJohnson.drop(columns = 'Player Name')
PGA_RORS_driving = PGA_RORS[PGA_RORS['Variable']=='Driving Distance - (AVG.)']

PGA_DJ_driving = PGA_DJ[PGA_DJ['Variable']=='Driving Distance - (AVG.)']

PGA_JasonDay_driving = PGA_JasonDay[PGA_JasonDay['Variable']=='Driving Distance - (AVG.)']

PGA_ZachJohnson_driving = PGA_ZachJohnson[PGA_ZachJohnson['Variable']=='Driving Distance - (AVG.)']
PGA_RORS_driving
PGA_ZachJohnson_driving
PGA_JasonDay_driving
PGA_DJ_driving
PGA_DJ_driving['Value']= PGA_DJ_driving['Value'].astype(float)

PGA_RORS_driving['Value']= PGA_RORS_driving['Value'].astype(float)

PGA_JasonDay_driving['Value']= PGA_JasonDay_driving['Value'].astype(float)

PGA_ZachJohnson_driving['Value']= PGA_ZachJohnson_driving['Value'].astype(float)
import matplotlib.pyplot as plt

%matplotlib inline
plt.plot(PGA_RORS_driving['Season'],PGA_RORS_driving['Value'],color='red',marker='o',label='Rors')

plt.plot(PGA_DJ_driving['Season'],PGA_DJ_driving['Value'], color='blue', marker='o', label='DJ')

plt.plot(PGA_JasonDay_driving['Season'],PGA_JasonDay_driving['Value'], color='green', marker='o', label='JDay')

plt.plot(PGA_ZachJohnson_driving['Season'],PGA_ZachJohnson_driving['Value'],color='black', marker='o',label='ZachJ' )

plt.title('Driving (Avg/Season) trend')

plt.xlabel('Season')

plt.ylabel('Driving (Avg) - yds')

plt.legend()