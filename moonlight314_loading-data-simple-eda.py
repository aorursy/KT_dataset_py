#-*- coding: CP949 -*-

%matplotlib inline

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



import warnings

warnings.filterwarnings('ignore')



import matplotlib

from IPython.display import set_matplotlib_formats



matplotlib.rc('font', family='Malgun Gothic')



matplotlib.rc('axes', unicode_minus=False)



set_matplotlib_formats('retina')



from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"



%config IPCompleter.greedy=True
train = pd.read_csv("../input/ashrae-energy-prediction/train.csv")

train.shape

train.head()
print("Building ID Count : ",train['building_id'].nunique())

print("Building ID Min Value : ",train['building_id'].min())

print("Building ID Max Value : ",train['building_id'].max())
tmp = train['meter'].value_counts(dropna=False)

tmp.plot.bar()

del tmp
tmp = train.sort_values(by = 'timestamp' )

tmp.head()

tmp.tail()

del tmp
building_metadata = pd.read_csv("../input/ashrae-energy-prediction/building_metadata.csv")



building_metadata.shape

building_metadata.head()



print("SIte ID Count : ",building_metadata['site_id'].nunique())

print("SIte ID Min Value : ",building_metadata['site_id'].min())

print("SIte ID Max Value : ",building_metadata['site_id'].max())
print("Building ID Count : ",building_metadata['building_id'].nunique())

print("Building ID Min Value : ",building_metadata['building_id'].min())

print("Building ID Max Value : ",building_metadata['building_id'].max())
plt.rcParams["figure.figsize"] = (18,8)

tmp = building_metadata['primary_use'].value_counts(dropna=False)

tmp

tmp.plot.bar(rot = 45)

del tmp
pd.isnull( building_metadata["square_feet"] ).sum()

building_metadata["square_feet"].describe()



sns.distplot( building_metadata["square_feet"] , bins = 500 )

plt.show()
print( "No Data Row Count :" , pd.isnull( building_metadata["year_built"] ).sum() )

building_metadata["year_built"].describe()
tmp = building_metadata["year_built"].dropna()

sns.distplot( tmp , bins = 120 , kde=False)

plt.show()
print( "No Data Row Count :" , pd.isnull( building_metadata["floor_count"] ).sum() )

building_metadata["floor_count"].describe()



building_metadata["floor_count"].value_counts()
tmp = building_metadata["floor_count"].dropna()

sns.distplot( tmp  , kde=False)

plt.show()
weather_test = pd.read_csv("../input/ashrae-energy-prediction/weather_test.csv")

weather_train = pd.read_csv("../input/ashrae-energy-prediction/weather_train.csv")



weather_train.shape

weather_train.head()



weather_test.shape

weather_test.head()
print("Site ID Count : " , weather_train['site_id'].nunique() )

weather_train['site_id'].value_counts(dropna=False)



print("Site ID Count : " , weather_test['site_id'].nunique() )

weather_test['site_id'].value_counts(dropna=False)
sample_submission = pd.read_csv("../input/ashrae-energy-prediction/sample_submission.csv")

test = pd.read_csv("../input/ashrae-energy-prediction/test.csv")
test.shape

test.head()
tmp = test.sort_values(by = 'timestamp')

tmp.head()

tmp.tail()

del tmp
sample_submission.shape

sample_submission.head()
print("Test Data Building ID Count : ",test['building_id'].nunique())

print("Test Data Building ID Min Value : ",test['building_id'].min())

print("Test Data Building ID Max Value : ",test['building_id'].max())