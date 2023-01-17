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
# Importing Essential Libraries

from matplotlib import pyplot as plt

import seaborn as sns, pystan, statsmodels.api as sm

from sklearn import linear_model
# Reading the dataset

train_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

sample_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')
# Understanding the training dataset

#train_data

#len(train_data)

#train_data.head(2)

#train_data.info()

#train_data.dtypes

#train_data.select_dtypes(include=int)

#train_data.YearBuilt

#train_data.select_dtypes(include=int)
# Histogram for the Year in which the house was built

n, bins, patches = plt.hist(x=train_data.YearBuilt, bins='auto', color='#b3b3ff', alpha=0.7, rwidth=0.85)

plt.grid(axis='y', alpha=0.75)

plt.xlabel('Years')

plt.ylabel('Number of houses built')

plt.title('Number of houses built every decade')

maxfreq = n.max()

plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10) # Set a clean upper y-axis limit.
# Histogram for the Year in which the house was remodeled

n, bins, patches = plt.hist(x=train_data.YearRemodAdd, bins='auto', color='#a2ffa2', alpha=0.7, rwidth=0.85)

plt.grid(axis='y', alpha=0.75)

plt.xlabel('Years')

plt.ylabel('Number of houses remodelled')

plt.title('Number of houses remodelled every decade')

maxfreq = n.max()

plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10) # Set a clean upper y-axis limit.
# Histogram for the Year in which the house was remodeled

age_of_remodel = train_data.YearRemodAdd - train_data.YearBuilt

n, bins, patches = plt.hist(x=age_of_remodel, bins=10, color = '#ffff0f', alpha=0.7, rwidth=0.85)

plt.grid(axis='y', alpha=0.75)

plt.xlim(5,120)

plt.ylim(0,150)

plt.xlabel('Years')

plt.ylabel('Number of houses remodelled')

plt.title('Years since the house was built to most recent remodel')

maxfreq = n.max()

#plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10) # Set a clean upper y-axis limit.
# Correlation between lot area and sale price

plt.scatter(train_data.LotArea,train_data.SalePrice, color = '#ffbbbb', marker = ">")

np.corrcoef(train_data.LotArea,train_data.SalePrice)
# Understanding the testing dataset

test_data.select_dtypes(include=int)
# Barplot for understnading the relationship between overall condition and sale price

sns.barplot(x='OverallQual', y='SalePrice', data=train_data)

np.corrcoef(train_data.OverallQual,train_data.SalePrice)
# Creating the predictor by combining a couple of variables

plt.subplots_adjust(left=0, bottom=0, right=1.5, top=1, wspace=0.5, hspace=1)

plt.subplot(121); sns.set_palette("Paired"); sns.barplot(x='OverallQual', y='YearRemodAdd', data=train_data); plt.ylim(1940,2015); plt.title('Train Data')

plt.subplot(122); sns.set_palette("pastel"); sns.barplot(x='OverallQual', y='YearRemodAdd', data=test_data); plt.ylim(1940,2015); plt.title('Test Data')
# Identifying the Distribution of Overall Quality 

plt.subplots_adjust(left=0, bottom=0, right=1.5, top=1, wspace=0.5, hspace=1)

plt.subplot(121); train_data.OverallQual.apply(lambda x: np.log(x+0.1)).hist(bins=20, color = '#99ff99')

plt.subplot(122); test_data.OverallQual.apply(lambda x: np.log(x+0.1)).hist(bins=20, color = '#ffbbbb')
# Getting essential summaries

train_data.select_dtypes(include=int).describe().loc[['min','max', 'mean','50%'],:]
# Creating the model based on Id

model = sm.OLS(train_data.SalePrice,train_data.Id).fit()

predictions = model.predict(train_data.SalePrice) 

model.summary()
# Comparing the predictions with the real values

plt.scatter(train_data.SalePrice,predictions, color = '#c1fffa', marker = "v")

np.corrcoef(train_data.SalePrice,predictions)

# Assigning the predictions to test data

test_data.SalePrice = predictions
# Creating the model based on Id

model = sm.OLS(train_data.SalePrice,train_data.OverallQual).fit()

predictions = model.predict(train_data.OverallQual) 

model.summary()
# Checking with the predictions

predictions