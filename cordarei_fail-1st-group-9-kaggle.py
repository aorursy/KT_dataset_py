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
prop2016.head()
clean2016 = prop2016[['parcelid', 'calculatedfinishedsquarefeet', 'landtaxvaluedollarcnt', 'propertyzoningdesc', 'regionidcounty', 'regionidzip', 'structuretaxvaluedollarcnt', 'taxamount', 'taxvaluedollarcnt', 'yearbuilt']].copy()
clean2016.head()
prop2017.tail()
prop2016.index = prop2016['parcelid']

del prop2016['parcelid']
prop2016.head()
prop2017.index = prop2017['parcelid']

del prop2017['parcelid']
prop2017.head()
#https://stackoverflow.com/questions/13413590/how-to-drop-rows-of-pandas-dataframe-whose-value-in-a-certain-column-is-nan

prop2016.dropna(subset = ['propertyzoningdesc'], inplace = True)

prop2016.dropna(subset = ['taxamount'], inplace = True)

prop2016.dropna(subset = ['landtaxvaluedollarcnt'], inplace = True)

prop2016.dropna(subset = ['taxvaluedollarcnt'], inplace = True)

prop2016.dropna(subset = ['structuretaxvaluedollarcnt'], inplace = True)

prop2016.dropna(subset = ['yearbuilt'], inplace = True)

prop2016.dropna(subset = ['regionidcounty'], inplace = True)

prop2016.dropna(subset = ['regionidzip'], inplace = True)

prop2016.dropna(subset = ['calculatedfinishedsquarefeet'], inplace = True)

#taxamount, landtaxvaluedollarcnt, taxvaluedollarcnt,structuretaxvaluedollarcnt, yearbuilt,regioncounty,regionzip,propertytype, calculatedfinishedsquarefeet

#bool_series=pd.notnull(prop2016["propertyzoningdesc"])

#prop2016=prop2016[bool_series]
prop2016.head()
prop2017.dropna(subset = ['propertyzoningdesc'], inplace = True)
prop2016.isnull().mean()
prop2017.isnull().mean()
#https://www.listendata.com/2019/06/pandas-drop-columns-from-dataframe.html

cols = prop2016.columns[prop2016.isnull().mean() > 0.7]

prop2016.drop(cols, axis=1, inplace = True)
cols = prop2017.columns[prop2017.isnull().mean() > 0.7]

prop2017.drop(cols, axis=1, inplace = True)
prop2016.head()
prop2017.head()
#https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.mean.html

prop2016.isnull().mean(axis=1)
prop2016.dropna(thresh=prop2016.shape[1]-3, axis=0, inplace = True)
prop2016.isnull().mean(axis=1)
rows = prop2016[prop2016.isnull().mean(axis=1) > 0.7]

prop2016.drop(rows, axis=0, inplace = True)
prop2016.head()
#https://www.geeksforgeeks.org/adding-new-column-to-existing-dataframe-in-pandas/

clean2016['propertytype'] = 'NR'

#https://stackoverflow.com/questions/36701689/assign-value-to-a-pandas-dataframe-column-based-on-string-condition/36701728#36701728

#https://stackoverflow.com/questions/28311655/ignoring-nans-with-str-contains

clean2016.loc[prop2016['propertyzoningdesc'].str.contains('r', na=False, case=False), 'propertytype'] = 'R'
prop2016.head(100)
#https://www.geeksforgeeks.org/adding-new-column-to-existing-dataframe-in-pandas/

prop2017['propertytype'] = 'NR'

#https://stackoverflow.com/questions/36701689/assign-value-to-a-pandas-dataframe-column-based-on-string-condition/36701728#36701728

#https://stackoverflow.com/questions/28311655/ignoring-nans-with-str-contains

prop2017.loc[prop2017['propertyzoningdesc'].str.contains('r', na=False, case=False), 'propertytype'] = 'R'

#https://thispointer.com/pandas-count-rows-in-a-dataframe-all-or-those-only-that-satisfy-a-condition/

TotalNumOfRows = len(clean2016.index)

seriesObj = clean2016.apply(lambda x: True if x['propertytype'] == 'R' else False, axis=1)

numOfRows = len(seriesObj[seriesObj == True].index)

print('Number of Residential Properties: ', numOfRows)

print ("Number of Non-Residential Properties: ", TotalNumOfRows - numOfRows)
plt.pie

prop2016.propertytype.value_counts().plot(kind='pie', autopct='%1.0f%%')

plt.axis('equal')

plt.title('Residential (R) and NonResidential (NR)')
first2016=prop2016.merge(trans2016, on = 'parcelid', how='outer')
total2016.head(100)
#https://thispointer.com/pandas-count-rows-in-a-dataframe-all-or-those-only-that-satisfy-a-condition/

TotalNumOfRows = len(prop2016.index)

seriesObj = prop2016.apply(lambda x: True if x['propertytype'] == 'R' else False, axis=1)

numOfRows = len(seriesObj[seriesObj == True].index)

print('Number of Residential Properties: ', numOfRows)

print ("Number of Non-Residential Properties: ", TotalNumOfRows - numOfRows)
total2016.ax = plt.subplots(figsize=(18,18)) #correlation map

hmcols = total2016.taxamount, total2016.landtaxvaluedollarcnt, total2016.taxvaluedollarcnt, total2016.structuretaxvaluedollarcnt, total2016.yearbuilt,total2016.regionidcounty, total2016.regionidzip, total2016.propertytype, total2016.calculatedfinishedsquarefeet, total2016.logerror

sns.heatmap(hmcols, annot=True, linewidths=5, ax=None, na=False)

plt.show()