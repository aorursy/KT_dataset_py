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
df = pd.read_excel('/kaggle/input/covid19/dataset.xlsx')
df.head(5)
df['SARS-Cov-2 exam result'] = df['SARS-Cov-2 exam result'].map(lambda r: 1 if r == 'positive' else 0 )
df = df.rename(columns={'SARS-Cov-2 exam result': 'has_covid', 'Patient addmited to regular ward (1=yes, 0=no)': 'hospital',

                   'Patient addmited to semi-intensive unit (1=yes, 0=no)': 'semi_icu',

                   'Patient addmited to intensive care unit (1=yes, 0=no)': 'icu'})
corr = df.corr()

corr[['has_covid', 'hospital', 'semi_icu', 'icu']].sort_values('has_covid', ascending=False)
features = df.iloc[:, 6:]

#For task 1, we won't use the hospital/semi-icu/icu data
import missingno as msno

# checking null values

msno.bar(features, figsize=(16, 4),log = True)
features.head(5)
#Dropping all NanColumns

all_nas = []

for f in features.columns:

    if features[f].isna().all():

        all_nas += [f]

features = features.drop(columns=all_nas) #Dropping all NanColumns
features.head(5)

#It's important to show the head of the dataframe because we know now that 5 features are completely needless
msno.bar(features, figsize=(16, 4), log = True)
msno.dendrogram(features)
nan_series = pd.Series(index = list(features.columns), data = np.zeros(len(features.columns)))

#Creating an empty nan_series. We'll add the nan_values below

nan_series.head(5)
for label in features.columns:

    for i in features[label].isnull().index:

        if features[label].isnull()[i] == True:

            nan_series[label]+=1

#This cell has a very high computational cost, by the way

nan_series.head(5)
nan_series.nunique()

#The value below shows that there are 40 categories of NaN values (41, if we count the all-NaN values)
np.unique(nan_series.values)

#These are the values of NaN for each group
#In this cell, we'll show the features in each category

unique_nan_series = np.unique(nan_series.values) 

for i in range(0, len(unique_nan_series)):

    print("Category ", i)

    print("Number of NaN values: ", unique_nan_series[i] )

    print("Features in category ", i, " :")

    for element in list(nan_series[nan_series == unique_nan_series[i]].index):

        print(element)

    print('\n')