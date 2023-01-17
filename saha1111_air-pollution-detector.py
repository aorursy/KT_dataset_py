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
Air_Quality = pd.read_excel("/kaggle/input/airquality/AirQuality.xlsx")
Air_Quality.head()
print(type(Air_Quality))

print(Air_Quality.columns)
Air_Quality
print(Air_Quality.index)
print(Air_Quality['Country'])
print(Air_Quality['State'])
print(Air_Quality.State.unique())
print(Air_Quality.Pollutants.unique())
print(Air_Quality.head())

print()

print()

print(Air_Quality.describe())
group_of_States = Air_Quality.groupby('State')

group_of_States.head()
Air_Quality.head(5)

group_of_States.head([5])
states_pollution = Air_Quality.groupby(['State','Pollutants'])
states_pollution.head()
state_quality = Air_Quality.groupby(['State', 'Pollutants'])
state_quality.head(10)
Mean_Pollution = group_of_States.mean()

Mean_Pollution
Mean_Pollution.sort_values(by=['Avg'])
type(Mean_Pollution)



type(Air_Quality)
type(group_of_States)
type(state_quality)
group_of_States_df = pd.DataFrame(group_of_States)

Air_Quality_df = pd.DataFrame(Air_Quality)
group_of_States_df
Air_Quality_df
import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns
plt.hist(Air_Quality_df.Avg, histtype='bar', rwidth=0.7)

plt.xlabel('AVERAGE')

plt.ylabel('COUNT')

plt.title('AVERAGE OF AIR QUALITY IN INDIA')

plt.show()
plt.figure(figsize=(17,7), dpi=100)

sns.countplot(x='State',data=Air_Quality)

plt.xlabel('State')

plt.tight_layout()
plt.hist(Air_Quality_df.Max, histtype ='bar', rwidth=0.7)

plt.xlabel("MAX")

plt.ylabel("COUNT")

plt.title("MAX OF AIR QUALITY OF INDIA")

plt.show()
plt.hist(Air_Quality_df.Min, histtype ='bar', rwidth=0.7)

plt.xlabel("Min")

plt.ylabel("COUNT")

plt.title("Min OF AIR QUALITY OF INDIA")

plt.show()
Air_Quality['Pollutants'].value_counts().plot()

plt.xlabel("Pollutants")

plt.ylabel("COUNT")

plt.title("Pollutants OF AIR QUALITY OF INDIA")

print(plt.show())

Air_Quality['Pollutants'].value_counts().plot('bar')

plt.xlabel("Pollutants")

plt.ylabel("COUNT")

plt.title("Pollutants OF AIR QUALITY OF INDIA")

print(plt.show())

import seaborn as sns

pollutant = list(Air_Quality['Pollutants'].unique())

for poll in pollutant:

    plt.figure(figsize=(18,8), dpi = 100)

    sns.countplot(Air_Quality[Air_Quality['Pollutants'] == poll]['State'], data = Air_Quality)

    plt.tight_layout()

    plt.title(poll)