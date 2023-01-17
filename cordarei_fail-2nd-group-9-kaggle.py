import numpy as np

import pandas as pd # for data processing-read

%matplotlib inline

import seaborn as sns # for visualization

import matplotlib.pyplot as plt

from matplotlib import pyplot as plt

import matplotlib.colors as mcolors
prop2016 = pd.read_csv ("../input/2016prop.csv", low_memory = False)

prop2017 = pd.read_csv ("../input/2017prop.csv", low_memory = False)

trans2016 = pd.read_csv ("../input/train_2016_v2.csv", low_memory=False)

trans2017 = pd.read_csv ("../input/train_2017.csv", low_memory=False)

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)
prop2016.head()
prop2017.head()
prop2016.tail()
prop2017.tail()
#https://stackoverflow.com/questions/34682828/extracting-specific-selected-columns-to-new-dataframe-as-a-copy

clean2016 = prop2016[['parcelid', 'calculatedfinishedsquarefeet', 'landtaxvaluedollarcnt', 'propertyzoningdesc', 'regionidcounty', 'regionidzip', 'structuretaxvaluedollarcnt', 'taxamount', 'taxvaluedollarcnt', 'yearbuilt']].copy()
clean2017 = prop2017[['parcelid', 'calculatedfinishedsquarefeet', 'landtaxvaluedollarcnt', 'propertyzoningdesc', 'regionidcounty', 'regionidzip', 'structuretaxvaluedollarcnt', 'taxamount', 'taxvaluedollarcnt', 'yearbuilt']].copy()
clean2016.head()
clean2017.head()
#https://www.geeksforgeeks.org/adding-new-column-to-existing-dataframe-in-pandas/

clean2016['propertytype'] = 'NR'

#https://stackoverflow.com/questions/36701689/assign-value-to-a-pandas-dataframe-column-based-on-string-condition/36701728#36701728

#https://stackoverflow.com/questions/28311655/ignoring-nans-with-str-contains

clean2016.loc[prop2016['propertyzoningdesc'].str.contains('r', na=False, case=False), 'propertytype'] = 'R'
#https://www.geeksforgeeks.org/adding-new-column-to-existing-dataframe-in-pandas/

clean2017['propertytype'] = 'NR'

#https://stackoverflow.com/questions/36701689/assign-value-to-a-pandas-dataframe-column-based-on-string-condition/36701728#36701728

#https://stackoverflow.com/questions/28311655/ignoring-nans-with-str-contains

clean2017.loc[prop2016['propertyzoningdesc'].str.contains('r', na=False, case=False), 'propertytype'] = 'R'
#https://stackoverflow.com/questions/44593284/python-pandas-dataframe-merge-and-pick-only-few-columns

clean2016=clean2016.merge(trans2016[['parcelid', 'logerror']], on = 'parcelid', how='outer')
clean2017=clean2017.merge(trans2017[['parcelid', 'logerror']], on = 'parcelid', how='outer')
clean2016.head()
clean2017.head()
clean2016.index = clean2016['parcelid']

del clean2016['parcelid']
clean2017.index = clean2017['parcelid']

del clean2017['parcelid']
clean2016.head()
clean2017.head()
#https://stackoverflow.com/questions/13413590/how-to-drop-rows-of-pandas-dataframe-whose-value-in-a-certain-column-is-nan

clean2016.dropna(subset = ['propertyzoningdesc'], inplace = True)

clean2016.dropna(subset = ['yearbuilt'], inplace = True)

clean2016.dropna(subset = ['calculatedfinishedsquarefeet'], inplace = True)

clean2016.dropna(subset = ['landtaxvaluedollarcnt'], inplace = True)

clean2016.dropna(subset = ['regionidcounty'], inplace = True)

clean2016.dropna(subset = ['regionidzip'], inplace = True)

clean2016.dropna(subset = ['structuretaxvaluedollarcnt'], inplace = True)

clean2016.dropna(subset = ['taxamount'], inplace = True)

clean2016.dropna(subset = ['taxvaluedollarcnt'], inplace = True)
clean2017.dropna(subset = ['propertyzoningdesc'], inplace = True)

clean2017.dropna(subset = ['propertyzoningdesc'], inplace = True)

clean2017.dropna(subset = ['yearbuilt'], inplace = True)

clean2017.dropna(subset = ['calculatedfinishedsquarefeet'], inplace = True)

clean2017.dropna(subset = ['landtaxvaluedollarcnt'], inplace = True)

clean2017.dropna(subset = ['regionidcounty'], inplace = True)

clean2017.dropna(subset = ['regionidzip'], inplace = True)

clean2017.dropna(subset = ['structuretaxvaluedollarcnt'], inplace = True)

clean2017.dropna(subset = ['taxamount'], inplace = True)

clean2017.dropna(subset = ['taxvaluedollarcnt'], inplace = True)
clean2016.head()
clean2017.head()
#https://thispointer.com/pandas-count-rows-in-a-dataframe-all-or-those-only-that-satisfy-a-condition/

TotalNumOfRows = len(clean2016.index)

seriesObj = clean2016.apply(lambda x: True if x['propertytype'] == 'R' else False, axis=1)

numOfRows = len(seriesObj[seriesObj == True].index)

print('Number of Residential Properties: ', numOfRows)

print ("Number of Non-Residential Properties: ", TotalNumOfRows - numOfRows)
TotalNumOfRows2 = len(clean2017.index)

seriesObj2 = clean2017.apply(lambda x: True if x['propertytype'] == 'R' else False, axis=1)

numOfRows2 = len(seriesObj2[seriesObj2 == True].index)

print('Number of Residential Properties: ', numOfRows2)

print ("Number of Non-Residential Properties: ", TotalNumOfRows2 - numOfRows2)
plt.pie

clean2016.propertytype.value_counts().plot(kind='pie', autopct='%1.0f%%')

plt.axis('equal')

plt.title('Residential (R) and NonResidential (NR)')

clean2017.propertytype.value_counts().plot(kind='pie', autopct='%1.0f%%')

plt.axis('equal')

plt.title('Residential (R) and NonResidential (NR)')