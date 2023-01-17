import xlrd
xls = xlrd.open_workbook(r'../input/DataDownload.xls')
print (xls.sheet_names()) #show the tab names in the workbook
import pandas as pd
variables = pd.read_excel('../input/DataDownload.xls',sheet_name='Variable List') #take a look at the variables in the workbook
variables
#let's look at the access tab
access = pd.read_excel('../input/DataDownload.xls',sheet_name='ACCESS')
access.head()
#find out more about PCT_LACCESS_POP15, which is the 2015 county population with low access to store as a percentage
access['PCT_LACCESS_POP15'].dropna().describe()
#mean population percentage with low access to grocery store by state
access.groupby('State').PCT_LACCESS_POP15.mean().plot(kind='bar',figsize=(12,4))
#median population percentage with low access to grocery store by state
access.groupby('State').PCT_LACCESS_POP15.median().plot(kind='bar',figsize=(12,4))
access.boxplot(column='PCT_LACCESS_POP15',by='State',figsize=(15,5))
import numpy as np
access['not_food_desert'] = np.where(access['PCT_LACCESS_POP15']<=67, 1, 0) #if more than 2/3 of population has access then not a food desert
fd = pd.crosstab(access['State'],access['not_food_desert']).apply(lambda r: r/r.sum(), axis=1)
fd
#let's visualize the above table where the height of the red bar is the percentage of counties in the state that are food deserts
fd.div(fd.sum(1),axis=0).plot(kind='bar',stacked=True, figsize=(15,5), color=('red','lightgrey'), legend=False)
#let's see how many counties there are per state in this dataset
access.groupby(['State'])['County'].count()