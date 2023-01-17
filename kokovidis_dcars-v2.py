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

        

        

pd.set_option('display.max_columns', 500) # show all columns

pd.options.display.max_rows = 200 # show 200 rows

        

import warnings

warnings.filterwarnings("ignore") #remove warning messages during csv import



# Any results you write to the current directory are saved as output.
fullspecs = pd.read_csv("/kaggle/input/thecarconnection32000/fullspecs (1).csv", warn_bad_lines=False)

print(fullspecs.shape) #dimensions

fullspecs_T = fullspecs.T
fullspecs_T.columns = fullspecs_T.iloc[0]

fullspecs_T.head()
fullspecs_T = fullspecs_T.drop(fullspecs_T.index[0])

fullspecs_T.head()
fullspecs_T = fullspecs_T.reset_index()

fullspecs_T=fullspecs_T.rename(columns = {'index':'Full Title'})

fullspecs_T.index.name = None

fullspecs_T.head()
fullspecs_T['Year']=fullspecs_T['Full Title'].str[0:4]

fullspecs_T.head()
fullspecs_T['Company-Model'] = fullspecs_T['Full Title'].str[5:].str.partition('Specs:')[0]

fullspecs_T['Specs'] = fullspecs_T['Full Title'].str[5:].str.partition('Specs:')[2]

fullspecs_T['MSRP'] = fullspecs_T['MSRP'].str.partition('$')[2]

fullspecs_T["MSRP"] = fullspecs_T["MSRP"].str.replace(",","").astype(float)

fullspecs_T.set_index('Full Title', inplace=True)

fullspecs_T.head()
fullspecs_T['mpg City'] = fullspecs_T['Gas Mileage'].str[:2]

fullspecs_T['mpg Hwy'] = fullspecs_T['Gas Mileage'].str.partition('/')[2].str[:2]



del fullspecs_T['Gas Mileage']



#fullspecs_T['mpg City'].value_counts(dropna=False)

#fullspecs_T['mpg Hwy'].value_counts(dropna=False)



#Remove columns with more than 75% missing values

fullspecs_T = fullspecs_T[fullspecs_T.columns[fullspecs_T.isnull().mean() < 0.25]]
# CHECK DISTINCT VALUES FOR ENGINE

#print(fullspecs_T['Engine'].value_counts().to_frame().to_string())
fullspecs_T['Engine Liters'] = fullspecs_T['Engine'].str.partition(',')[2].str.partition('L')[0]

fullspecs_T['Engine Name'] = fullspecs_T['Engine'].str.partition(',')[0]

del fullspecs_T['Engine']



fullspecs_T['SAE Net Torque'] = fullspecs_T['SAE Net Torque @ RPM'].str.partition('@')[0]

fullspecs_T['RPM of SAE Net Torque'] = fullspecs_T['SAE Net Torque @ RPM'].str.partition('@')[2]

del fullspecs_T['SAE Net Torque @ RPM']



fullspecs_T['SAE Net Horsepower'] = fullspecs_T['SAE Net Horsepower @ RPM'].str.partition('@')[0]

fullspecs_T['RPM of SAE Net Horsepower']= fullspecs_T['SAE Net Horsepower @ RPM'].str.partition('@')[2]

del fullspecs_T['SAE Net Horsepower @ RPM']

# Convert Yes/No to 1/0 



list_of_bool = [

 'Disc - Rear (Yes or   )',

 'Disc - Front (Yes or   )',

 'Air Bag-Frontal-Driver',

 'Air Bag-Frontal-Passenger',

 'Air Bag-Passenger Switch (On/Off)',

 'Air Bag-Side Body-Front',

 'Air Bag-Side Body-Rear',

 'Air Bag-Side Head-Front',

 'Air Bag-Side Head-Rear',

 'Brakes-ABS',

 'Child Safety Rear Door Locks',

 'Daytime Running Lights',

 'Traction Control',

 'Night Vision',

 'Rollover Protection Bars',

 'Fog Lamps',

 'Parking Aid',

 'Tire Pressure Monitor',

 'Back-Up Camera',

 'Stability Control',

 ] # needs to be double checked



for index, name in enumerate(list_of_bool):

    fullspecs_T[name] = fullspecs_T[name].map(dict(Yes=1, No=0))
fullspecs_T.head()



######################## TO DO ########################

# Rear Tire Size, Front Wheel Size (in), Rear Wheel Size (in)

# Check more the NaN values
fullspecs_T.to_csv('fullspecs.csv')