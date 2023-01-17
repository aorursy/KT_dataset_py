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
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('/kaggle/input/daily-temperature-of-major-cities/city_temperature.csv')
df[df['Country']=='Brazil']
# Well, a lot of work ahead, getting start with a NaN values and Temperature in F...
# Lets see now a plot of our data
plt.figure(figsize=(10,6))

sns.distplot(df['AvgTemperature'],bins=30)
#We can see a uncomun values about -100 F? a bit strange...Lets rip them out.
df[(df['AvgTemperature']<=0) & (df['Country']=='Brazil')]
df.drop('State',axis=1,inplace=True)
df['AvgTemperature'].replace(-99.0,np.nan,inplace=True)
# What I did in that last lines was replace that -99 by nan values.
plt.figure(figsize=(10,6))

sns.distplot(df['AvgTemperature'],bins=30)
# the plot above we can no more see that values but the NaN are still there.
df[(df['AvgTemperature']<0)&(df['Country']=='Brazil')].count()
#Lets drop then.
#Obs: I choose this method because there was few records with uncomun values. I considered whimsy for any real effect in the data. 
df.dropna(inplace=True)
# Lets see if there is some null values
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#Right, lets see whats going on around the world.
regioes = df['Region']
plt.figure(figsize=(10,6))

sns.lineplot(x=df['Year'], y=df['AvgTemperature'],hue=regioes,data = df)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# We can see above the Africa and Middle East are picking a little fight for the hotter continent of the world
# Down bellow, I'll convert the temperature to Celsius.
def temp(F):

    C = (F-32)/1.8

    return round(C,2)
df['AvgTemperature'] = temp(df['AvgTemperature'])
df.loc[df['AvgTemperature'].idxmax()]
# Gee!!! Kuwait has the hotter average temperature in the world
# Lets see whats going on here in Brazil.
df[df['Country']=='Brazil']
df[(df['Year']==2020)&(df['Country']=='Brazil')].count()
df[df['City']=='Brasilia']['AvgTemperature'].max()
df[df['City']=='Sao Paulo']['AvgTemperature'].max()
df[df['City']=='Rio de Janeiro']['AvgTemperature'].max()
# Yes Bro, Rio is the hotter city in Brazil, lets go to the beach.
br = df[df['Country']=='Brazil']
plt.figure(figsize=(10,6))

sns.barplot(x='Month',y='AvgTemperature',data=br,color='b')
#Above we can see a similar average temperature of all those years along the months...

# and just in the winter(may, june, july,august) a low averare of the temperature.
plt.figure(figsize=(10,6))

sns.lineplot(x='Month',y='AvgTemperature',hue='City',data=br)
# Now, the other view of the data, a much more simple one to the best visualize the data.

# Sao Paulo is the 'colder' city, its a charming place to visit.

# Brasilia is kind the average of the other two, its good place there.
plt.figure(figsize=(10,6))

sns.boxplot(x = br['City'],y = br['AvgTemperature'])
# That's a intersting plot. We can see a low variance in Brasilia, a average one in Rio and higher in Sao Paulo.



# Thats it for now. Maybe I go on with some predictions for this data but I didn't see nothing relevant to bring them

# to some predictions, just for a simple analysis.