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
import matplotlib.pyplot as plt

import seaborn as sns
fit = pd.read_csv('/kaggle/input/one-year-of-fitbit-chargehr-data/One_Year_of_FitBitChargeHR_Data.csv')

fit.head()
fit['Minutes_sitting'] = fit['Minutes_sitting'].astype(int)
fit['total_activity'] = fit[['Minutes_of_slow_activity','Minutes_of_moderate_activity','Minutes_of_intense_activity']].sum(axis=1)

fit['total_day_minutes'] = fit['Minutes_sitting'] + fit['total_activity']

fit.head()
fit.drop(columns = 'floors',inplace=True)
fit.info()
fit.describe()
#Converting the date from object to date format  

fit['Date'] = pd.to_datetime(fit['Date'],format='%d-%m-%Y')

#Extracting the month only

fit['Month_only'] = pd.to_datetime(fit['Date']).dt.month

fit
### Scatterplot of Calories Vs total activity



# figure size

plt.figure(figsize=(15,8))



# Simple scatterplot

ax = sns.scatterplot(x='total_activity', y='Calories', data=fit)



ax.set_title('Scatterplot of calories and total_activities')
#Bar plot with respect ot date and calories burned

plt.figure(figsize=(20,6))

sns.barplot(x="Date", y="Calories", data=fit)

plt.title('Calories with respect to date')

plt.xticks(rotation=90)

plt.show()
#Extracting the week day from the date

fit['day_of_week'] = fit['Date'].dt.dayofweek

plt.figure(figsize=(15,6))

data = fit.groupby('day_of_week').sum().reset_index()

sns.barplot(x='day_of_week',y='total_activity',data=fit)

plt.title('DAY OF THE WEEK')

plt.show()
plt.figure(figsize=(15,6))

sns.scatterplot(x='Date', y='total_activity', data=fit)

plt.title('Activity')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(15,6))

data = fit.groupby('Month_only').sum().reset_index()

sns.barplot(x='Month_only',y='total_activity',data=fit)

plt.title('Month vs activity')

plt.show()
#Distanc was in object and also it has ' , ' in it so here I am removing the comma and converting it as float

fit['Distance'] = fit.Distance.str.replace(',', '').astype(float)

fit['Distance']

#pd.to_numeric(cpy_fit, downcast='integer')

#cpy_fit['first'] = cpy_fit['first'].astype(int)



#cpy_fit["Last"] = cpy_fit["Last"].astype(str).astype(int)

#cpy_fit['Distance_meter'] = cpy_fit['first'] * 1000

#cpy_fit['Last']  = cpy_fit['Last'].astype(int)

#cpy_fit['Distance_meter'] = cpy_fit['Distance_meter'] + cpy_fit['Last']

#converting values into meters already it was in KM (already 2places of decimal was correct, multiplying by 10),in above cell when we convert into float it changed

fit['Distance'] = fit['Distance']*10

fit
fit['length'] = fit['Distance']/(fit['Steps']*1000)

fit['length']
fit['length'].median()
print(fit['total_day_minutes'].mean())

print(fit['Minutes_sitting'].mean())

print(fit['total_activity'].mean())
descending = fit.sort_values(by='Distance', ascending=False)

descending.head(10)


# figure size

plt.figure(figsize=(15,8))



# Simple scatterplot

ax = sns.scatterplot(x='Distance', y='total_activity', data=descending)



ax.set_title('Scatterplot of distance and total_activities')
ascending = fit.sort_values(by='Distance')

ascending.head(20)
# number of days the user walk less than 1 km in a day.

ascending[ascending['Distance'] < 1000].groupby('Distance').sum()
fit.head()


fit['speed_km'] = (fit['Distance']/1000)/(fit['total_activity']/60)

print(fit['speed_km'].median(), 'kilometer per hour')

cpy_fit =fit

cpy_fit = cpy_fit.drop(['Month_only', 'day_of_week', 'length','speed_km'], axis = 1)

f, ax = plt.subplots(figsize=(10, 8))

corr_temp = cpy_fit.corr()

ax = sns.heatmap(corr_temp, mask=np.zeros_like(corr_temp, dtype=np.bool), 

                 cmap=sns.diverging_palette(220, 10, as_cmap=True),

                 annot=True, square=True)



ax.set_title('Correlation between calories and different activities')