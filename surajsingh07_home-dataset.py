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
train_data = pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv')

test_data = pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv')
# There are 81 columns so taking only valuable columns



feature_names = ['LotArea','YearBuilt','YearRemodAdd','OverallCond','GarageArea','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr',

                 'TotRmsAbvGrd','SalePrice']

df = train_data[feature_names]
df.head(3)
df.describe()
remodified_homes = round((len(df.loc[df['YearBuilt']!=df['YearRemodAdd']])/len(df))*100)

print('Remodified house percentage: {}%'.format(remodified_homes))



double_story_house = round((len(df[df['2ndFlrSF']==0])/len(df))*100)

print('Double story house percentage: {}%'.format(double_story_house))
# Importing visiualization libraries

import matplotlib.pyplot as plt

import seaborn as sns
# finding how much percentage of data outliear takes

LotArea_outliers = (len(df[df['LotArea']>=40000])/len(df))*100

print('LotArea_outliers percentage: {}%'.format(LotArea_outliers))



# Removing outliers

df = df.loc[(df['LotArea']<40000) & (df['OverallCond']>1)]

print('Outliers Removed')
fig, ((ax0,ax1),(ax2,ax3)) = plt.subplots(ncols=2,nrows=2,figsize=(15,10))



ax0.hist(df['YearBuilt'],bins=50, edgecolor='black')

ax0.set_title('YearBuilt',fontsize=20)



ax1.hist(df['LotArea'],bins=50, edgecolor='black')

ax1.set_title('LotArea',fontsize=20)



ax2.hist(df['1stFlrSF'],bins=50, edgecolor='black')

ax2.set_title('1stFlrSF',fontsize=20)



ax3.hist(df['SalePrice'],bins=50, edgecolor='black')

ax3.set_title('SalePrice',fontsize=20)
fig, ((ax0,ax1),(ax2,ax3)) = plt.subplots(nrows=2, ncols=2, figsize=(20,20))



slices_1 = df['OverallCond'].value_counts().values

labels_1 = ['Avg','Above Avg','Good','Very Good','Below Avg','Fair','Excellent','Poor']

ax0.pie(slices_1,labels=labels_1, wedgeprops={'edgecolor':'Black'}, autopct='%1.1f%%',textprops={'fontsize':15}, pctdistance=0.9)

ax0.set_title('House Conditions',fontsize=20)



slices_2 = df['TotRmsAbvGrd'].value_counts().values

labels_2 = df['TotRmsAbvGrd'].value_counts().index

ax1.pie(slices_2,labels=labels_2, wedgeprops={'edgecolor':'Black'}, autopct='%1.1f%%', textprops={'fontsize':15})

ax1.set_title('Total Number of Room',fontsize=20)



slices_3 = df['FullBath'].value_counts().values

labels_3 = ['two','one','three','zero']

ax2.pie(slices_3,labels=labels_3, wedgeprops={'edgecolor':'Black'}, autopct='%1.1f%%', textprops={'fontsize':15})

ax2.set_title('Bathroom',fontsize=20)



slices_4 = df['BedroomAbvGr'].value_counts().values

labels_4 = df['BedroomAbvGr'].value_counts().index

ax3.pie(slices_4,labels=labels_4, wedgeprops={'edgecolor':'Black'}, autopct='%.2f%%', textprops={'fontsize':15}, pctdistance=0.9)

ax3.set_title('Bedroom',fontsize=20)



plt.show()
sns.pairplot(data=df, y_vars=['SalePrice'],x_vars=['LotArea','YearBuilt','GarageArea','1stFlrSF','2ndFlrSF'])

plt.show()
X = df.drop('SalePrice',axis=1)

y = df['SalePrice']



X_test = test_data[['LotArea','YearBuilt','YearRemodAdd','OverallCond','GarageArea','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr',

                 'TotRmsAbvGrd']]
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X,y)
X_test = X_test.fillna(X_test['GarageArea'].mean())

X_test.info()
predictions = model.predict(X_test)
output = pd.DataFrame({'Id':test_data.Id,'SalePrice':predictions})

output.to_csv('my_submission',index=False)

print('output saved')