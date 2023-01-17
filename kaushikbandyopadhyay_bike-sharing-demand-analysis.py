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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('/kaggle/input/hour.csv')

df.head()
#Check for null values in the data

df.isnull().sum()
#Sanity checks

#Check if registered + casual = cnt for all the records.

#Month values should be 1-12 only

#Hour values should be 0-23

print('Month sanity: ',np.sum(df['registered'] + df['casual'] != df['cnt']))

print('Hour value: ',np.unique(df['hr']).tolist())
#drop redundant columns and created a new data frame:inp1

inp0 = df.drop(df[['casual','registered','dteday','instant']],axis=1,inplace=True)

df.head()

inp1 = df.copy()
#Created a density plot,this would give a sense of the centrality and the spread of the distribution.

inp1.temp.plot.density()
#Boxplot for atemp ,to check outliers

inp1.atemp.plot.box()
#histogram to detect abnormally high values 

inp1.hum.plot.hist()
#Density plot for windspeed

inp1.windspeed.plot.kde()
#boxplot for cnt columns, found outliers present

inp1.cnt.plot.box()

#Found out the following percentiles - 10, 25, 50, 75, 90, 95, 99

#Decide the cutoff percentile and drop records with values higher that the cutoff, named the new dataframe ‘inp2’.

inp1.cnt.quantile([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])

inp2 = inp1[inp1.cnt < 563].copy()

inp2.cnt.describe()

plt.figure(figsize=[12,6])

sns.boxplot(x='hr',y='cnt',data=inp2)

#It’s evident that the peak hours are 5PM – 7PM, the hours 7-8AM also have high upper quartile. 

#A hypothesis could be that a lot of people use the bikes for commute to workplace and back.
plt.figure(figsize=[12,6])

sns.boxplot('mnth','cnt',data=inp2)

#Looks like end of winter/ early spring months have the least bike riding instances.
plt.figure(figsize=[12,6])

sns.boxplot('season','cnt',data=inp2)
#Make a bar plot with the median value of cnt for each hr

res = inp2.groupby('hr').mean()

res.cnt.plot.bar()
corr = inp2[['atemp','temp','hum','windspeed']].corr()

sns.heatmap(corr, annot=True, cmap="Reds")
#Treating mnth column

#For values 5,6,7,8,9,10, replace with a single value 5. This is because these have very similar values for cnt.

#Get dummies for the updated 6 mnth values

inp3 = inp2.copy()

inp3.mnth[inp3.mnth.isin([5,6,7,8,9])] = 5

np.unique(inp3.mnth)

#Treating hr column

#Create new mapping: 0-5: 0, 11-15: 11; other values are untouched. Again, the bucketing is done in a way that hr values with similar levels of cnt are treated the same.



inp3.hr[inp3.hr.isin([0,1,2,3,4,5])] = 0

inp3.hr[inp3.hr.isin([11,12,13,14,15])] = 11
#Get dummy columns for season, weathersit, weekday, mnth, and hr. You needn’t club these further as the levels seem to have different values for the median cnt, when seen from the box plots.

cat_cols = ['season', 'weathersit', 'weekday', 'mnth', 'hr']

inp3 = pd.get_dummies(inp3, columns=cat_cols, drop_first=True)

inp3
#Train test split: Apply 70-30 split.

#Model building

#Use linear regression as the technique

#Report the R2 on the train set

#Make predictions on test set and report R2.

from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(inp3, train_size = 0.7, random_state = 100)

y_train = df_train.pop("cnt")

X_train = df_train



y_test = df_test.pop("cnt")

X_test = df_test

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X_train, y_train)



#Reporting r2 for the model 

from sklearn.metrics import r2_score

y_train_pred= lr.predict(X_train)

r2_score(y_train, y_train_pred)



y_test_pred= lr.predict(X_test)

r2_score(y_test, y_test_pred)
