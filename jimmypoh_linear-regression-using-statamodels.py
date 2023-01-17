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
import matplotlib.pyplot as plt

import seaborn as sns



wave = pd.read_csv("/kaggle/input/waves-measuring-buoys-data-mooloolaba/Coastal Data System - Waves (Mooloolaba) 01-2017 to 06 - 2019.csv")
wave.describe()

# we can notice that there is an abnormal value of -99.9 presents in every variables.
wave.info()

# we can change "Date/Time" from object format to datetime format 
wave["Date/Time"] = pd.to_datetime(wave["Date/Time"])
# I will remove rows with value -99.9 before I do any EDA on the data itself

Hs_index = wave[wave["Hs"] == -99.9].index

wave.drop(Hs_index, inplace=True)



Peak_index = wave[wave["Peak Direction"] == -99.9].index

wave.drop(Peak_index, inplace=True)



SST_index = wave[wave["SST"] == -99.9].index

wave.drop(SST_index, inplace=True)
wave.describe()

# just to make sure value of -99.9 removed from the data.
########################## Exploratory Data Analysis #####################################



plt.figure(figsize=(16,8))

sns.lineplot(x="Date/Time", y="Hs", data=wave)

plt.show()

# Hs shows highest readings at around Oct-2018 and Mar-2019
plt.figure(figsize=(16,8))

sns.lineplot(x="Date/Time", y="Tp", data=wave)

plt.show()

# Increase in peak energy wave period does not neccessary increase wave height

# It does not really explain why there were two sudden peaks of wave height at around Oct-2018 and Mar-2019.
plt.figure(figsize=(16,8))

sns.lineplot(x="Date/Time", y="Peak Direction", data=wave)

plt.show()

# There is a considerably high peak direction value at around Mar-2019 which could explain why there is a sudden peak of wave height.

# However it still doesn't explain sudden peak of wave height at around Oct-2018.
plt.figure(figsize=(16,8))

sns.lineplot(x="Date/Time", y="SST", data=wave)

plt.show()

# Sudden increase of SST at around Oct-2018 may be the reason of sudden peak of wave height at around that time.

# However it does not really explain sudden peak of wave height at around Mar-2019.
# Perhaps there is other factor that can explain the sudden peak of wave height around Oct-2018 and Mar-2019.

# Or it could be simply due to measurement error.
plt.figure(figsize=(16,8))

sns.scatterplot(x="Hmax", y="Hs", data=wave)

plt.show()

# Hmax and Hs has very strong correlation
plt.figure(figsize=(16,8))

sns.scatterplot(x="SST", y="Hs", data=wave)

plt.show()

# Hs and SST has weak correlation

# Perhaps change of sea surface temperature will not affect wave height
corrmat = wave.corr()

sns.heatmap(corrmat, annot=True)
features = corrmat["Hs"]

features.abs().sort_values(ascending=False)

# it seems like Hmax has very strong correlation with Hs

# Tp and Peak Direction has very weak correlation with Hs
################################ Feature Engineer #########################################



# check for missing values



wave.isnull().sum()

# no missing values
# check for outliers



sns.boxplot(wave["Hs"])

# got outliers at upper end
sns.boxplot(wave["Hmax"])

# got outliers at upper end
sns.boxplot(wave["Tz"])

# got outliers at both ends
sns.boxplot(wave["Peak Direction"])

# got outliers at both ends

# got one extreme outlier at upper end
sns.boxplot(wave["SST"])

# no outliers
# remove the extreme outlier

peak_index = wave[wave["Peak Direction"] > 300].index

wave.drop(peak_index, inplace=True)
# unsure of whether to retain or remove those outliers

# I decided to create another set of data with removed outliers



wave_removed = wave.copy()



hs_ind = wave_removed[wave_removed["Hs"] > 2.6015].index

wave_removed.drop(hs_ind, inplace=True)



hmax_ind = wave_removed[wave_removed["Hmax"] > 4.275].index

wave_removed.drop(hmax_ind, inplace=True)



tz_lower_ind = wave_removed[wave_removed["Tz"] < 3.2375].index

wave_removed.drop(tz_lower_ind, inplace=True)

tz_upper_ind = wave_removed[wave_removed["Tz"] > 7.8415].index

wave_removed.drop(tz_upper_ind, inplace=True)



peak_lower_ind = wave_removed[wave_removed["Peak Direction"] < 38.5].index

wave_removed.drop(peak_lower_ind, inplace=True)

peak_upper_ind = wave_removed[wave_removed["Peak Direction"] > 162.5].index

wave_removed.drop(peak_upper_ind, inplace=True)

# model building on wave data sets

# split the data into train and test sets



X = wave.drop(["Date/Time", "Hs"], axis=1)

Y = wave["Hs"]



from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=101)
from statsmodels.regression.linear_model import OLS



model = OLS(Y_train, X_train).fit()

model.summary()

# adjusted r-squared is 0.992 which is very good

# all variables have p-value less than 0.05 which show that all of them are significant to our model.
# evaluate the model

pred = model.predict(X_test)



from sklearn.metrics import mean_squared_error

mean_squared_error(Y_test, pred)
# model building on wave_removed data sets

# split the data into train and test sets



X_removed = wave_removed.drop(["Date/Time", "Hs"], axis=1)

Y_removed = wave_removed["Hs"]



X_train_removed, X_test_removed, Y_train_removed, Y_test_removed = train_test_split(X_removed, Y_removed, test_size=0.3, random_state=101)
model_removed = OLS(Y_train_removed, X_train_removed).fit()

model_removed.summary()

# adjusted r-squared is 0.993 which is slightly higher than model with outliers retained.

# all variables have p-value less than 0.05
# evaluate the model

pred_removed = model.predict(X_test_removed)

mean_squared_error(Y_test_removed, pred_removed)

# it shows that mean_squarer_error of model with outliers removed is lower.

# removal of outliers will slightly increase performance of the model.
# As a conclusion, we can choose to remove those outliers in order to improve performance of the model.

# Those variables of data are sufficient to predict Hs.