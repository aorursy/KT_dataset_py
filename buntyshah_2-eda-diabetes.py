# Using Counties.csv received from Feature analysis of USDA Data.
# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import scipy.stats as stats
sns.set_style('darkgrid')
sns.set_context('talk')
sns.set_palette('OrRd')

counties=pd.read_csv('../input/counties.csv')
counties.head()
# Confirm no missing values
counties.isnull().sum()
# Confirm all columns are numeric
counties.dtypes
# Confirm no extreme minimums# Confi 
counties.min()
# Confirm no extreme maximums# Confi 
counties.max()
# Is the target imbalanced?
print('Above average diabetes:', '\n', counties['hi_diabetes'].value_counts())
print('\n')
print('Above average obesity:', '\n', counties['hi_obesity'].value_counts())
# Are they correlated with the target variable?
corr=counties[['hi_diabetes', 'hi_obesity', 'GROCPTH14',  'WICSPTH12','SNAPSPTH16','SPECSPTH14','CONVSPTH14']].corr()
print(corr)
new=abs(corr)
sns.heatmap(new, cmap='Blues',annot=True);
# Are they correlated with the target variable?
corr=counties[['hi_diabetes', 'hi_obesity', 'FFRPTH14', 'FSRPTH14', 'PC_FFRSALES12', 'PC_FSRSALES12']].corr()
print(corr)
new=abs(corr)
sns.heatmap(new, cmap='Greens',annot=True);
# Are they correlated with the target variable?
corr=counties[['hi_diabetes', 'hi_obesity', 'PCT_NHWHITE10', 'PCT_65OLDER10', 'PCT_18YOUNGER10', 'PERPOV10', 'METRO13', 'POPLOSS10','MEDHHINC15']].corr()
print(corr)
new=abs(corr)
sns.heatmap(new, cmap='Reds',annot=True);
# Are they correlated with the target variable?
corr=counties[['hi_diabetes', 'hi_obesity','RECFACPTH14','PCT_LACCESS_POP15']].corr()
print(corr)
new=abs(corr)
sns.heatmap(new, cmap='BuPu',annot=True);
# Are they correlated with the target variable?
corr=counties[['hi_diabetes', 'hi_obesity', 'SNAP_PART_RATE13', 'PCT_NSLP15', 'PCT_WIC15', 'PCT_CACFP15']].corr()
print(corr)
new=abs(corr)
sns.heatmap(new, cmap='OrRd',annot=True);
# There is no significant difference in the proposed target variable, between groups.
sns.barplot(y='PCT_WIC15', x='hi_obesity', data=counties);
stats.ttest_ind(counties.loc[counties['hi_obesity']==1, 'PCT_WIC15'], counties.loc[counties['hi_obesity']==0, 'PCT_WIC15'])

# There is no significant difference in the proposed target variable, between groups.# There  
sns.barplot(y='PCT_WIC15', x='hi_diabetes', data=counties);
stats.ttest_ind(counties.loc[counties['hi_diabetes']==1, 'PCT_WIC15'], counties.loc[counties['hi_diabetes']==0, 'PCT_WIC15'])
