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
file_path = '../input/world-development-indicators/Indicators.csv'

world_indicators = pd.read_csv(file_path)

world_indicators
len(world_indicators.CountryCode.unique())
len(world_indicators.Year.unique())
for col in world_indicators.columns:

    print(col, len(world_indicators[col].unique()))
world_indicators.IndicatorName 
world_indicators.IndicatorName.iloc[3]
world_indicators.IndicatorName.iloc[1]
age_arb = world_indicators.loc[(world_indicators.CountryCode == 'ARB') & (world_indicators.IndicatorCode == 'SP.POP.DPND.YG')]

age_arb
sns.lineplot(x=age_arb['Year'], y=age_arb['Value'])
world_indicators.head()
world_indicators.tail()
world_indicators.columns
import seaborn as sns

# Create the default pairplot

sns.pairplot(world_indicators)
#sns.heatmap(data=world_indicators, annot=True)

sns.scatterplot(x=world_indicators['Year'], y=world_indicators['Value'])
sns.regplot(x=world_indicators['Year'], y=world_indicators['Value'])

import matplotlib.pyplot as plt

plt.figure(figsize=(14,7))



sns.scatterplot(x=world_indicators['Year'], y=world_indicators['Value'], hue=world_indicators['CountryCode'])
sns.kdeplot(data=world_indicators['Value'], shade=True)

sns.jointplot(x=world_indicators['Year'], y=world_indicators['Value'], kind="kde")

# Histograms for each species

sns.kdeplot(data=world_indicators['Year'], label="Year", shade=True)

sns.kdeplot(data=world_indicators['Value'], label="Value", shade=True)



# Add title

plt.title("Histogram of data")



# Force legend to appear

plt.legend()
y = world_indicators.IndicatorName
features=['Year','Value']

   

X=world_indicators[features]