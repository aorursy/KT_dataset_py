import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Load all files 

# UK paygap data
gap = pd.read_csv('../input/UK Gender Pay Gap Data_05_06.csv')

# SIC codes data
co_data = pd.read_csv('../input/all_uk_data.csv')


gap.sample(5)
gap.info()
co_data.sample(5)
co_data.info()
# Remove spaces from beginning of column names of co_data df

co_data.columns = [c.replace(' ', '_') for c in co_data.columns]

co_data.info()
# Rename the column CompanyNumber

co_data.rename(columns={'_CompanyNumber': 'CompanyNumber'}, inplace=True)

# Join the two df's on CompanyNumber

gap_all = pd.merge(gap, co_data, how='left', on='CompanyNumber')
# Confirm the length of the the joined dataset

len(gap_all)
gap_all.head()
gap_all.info()
# Define what columns we need

gap_col = ['EmployerName', 'DiffMedianHourlyPercent', 'DiffMedianBonusPercent', 
      'MaleBonusPercent', 'FemaleBonusPercent', 'MaleLowerQuartile', 'FemaleLowerQuartile', 'MaleTopQuartile', 
      'FemaleTopQuartile', 'CompanyLinkToGPGInfo', 'EmployerSize', 'RegAddress.PostTown', 'CompanyCategory',
           'SICCode.SicText_1'] 


gap_all = gap_all[gap_col]


gap_all.info()
gap_all.describe()
gap_all['DiffMedianHourlyPercent'].plot(kind='hist', bins=50, figsize=[12,6], alpha=.6, legend=True, color = 'green')

gap_all['DiffMedianBonusPercent'].plot(kind='hist', bins=50, figsize=[12,6], alpha=.6, legend=True, color = 'green',
                                       range=(-250,250)) #setting up a range to ignore ourliers

# 10 Companies with highest paygap in hourly payment favouring men
    
gap_all.nlargest(10, 'DiffMedianHourlyPercent')
# 10 Companies with highest paygap in hourly payment favouring women
    
gap_all.nsmallest(10, 'DiffMedianHourlyPercent')
# 10 Companies with highest paygap in bonus payment favouring men
    
gap_all.nlargest(10, 'DiffMedianBonusPercent')
# 10 Companies with highest paygap in bonus payment favouring women
    
gap_all.nsmallest(10, 'DiffMedianBonusPercent')
# Explore the frequency of the various SIC codes

sic = gap_all['SICCode.SicText_1'].value_counts()
sic[sic > 50]
# Convert the categorical to ordinal variables, by choosing the mid-point of each range.

mapping = {
    'Less than 250': 125,
    '250 to 499': 350,
    '500 to 999': 750,
    '1000 to 4999': 3500,
    '5000 to 19,999': 12000,
    '20,000 or more': 35000,
    'Not Provided': np.nan
}

gap_all['EmployerSizeCenter'] = gap_all['EmployerSize'].map(mapping.get)
#Check for collinear variables

corr_matrix = gap_all.corr()
sns.heatmap(corr_matrix)
# Plot the DiffMedianHourlyPercent based on EmployerSizeCenter

with sns.axes_style("darkgrid"):
    ax = sns.factorplot(kind='box', y='DiffMedianHourlyPercent', x='EmployerSizeCenter',
                   data=gap_all, size=8, aspect=1.5, legend_out=False)
    ax.set(ylim=(-70, 70))
# Plot the DiffMedianBonusPercent based on EmployerSizeCenter

with sns.axes_style("darkgrid"):
    ax = sns.factorplot(kind='box', y='DiffMedianBonusPercent', x='EmployerSizeCenter',
                   data=gap_all, size=8, aspect=1.5, legend_out=False)
    ax.set(ylim=(-200, 200))
# Plot the FemaleTopQuartile based on EmployerSizeCenter

with sns.axes_style("darkgrid"):
    ax = sns.factorplot(kind='box', y='FemaleTopQuartile', x='EmployerSizeCenter',
                   data=gap_all, size=7, aspect=3, legend_out=False)
    ax.set(ylim=(-0, 100))
# See https://en.wikipedia.org/wiki/Standard_Industrial_Classification

mappings = [
    (100, 999, 'Agriculture'),
    (1000, 1499, 'Mining'),
    (1500, 1799, 'Construction'),
    (1800, 1999, 'not used'),
    (2000, 3999, 'Manufacturing'),
    (4000, 4999, 'Utility Services'),
    (5000, 5199, 'Wholesale Trade'),
    (5200, 5999, 'Retail Trade'),
    (6000, 6920, 'Financials'),
    (7000, 9004, 'Services'),
    (9100, 9729, 'Public Administration'),
    (9800, 9999, 'Nonclassifiable'),
]

errors = set()
def to_code_range(i):
    if type(i) != str:
        return np.nan
    if i == "None Supplied":
        return np.nan
    code = int(i[0:4])
    for code_from, code_to, name in mappings:
        if code >= code_from and code <= code_to:
            return name
    #print("ERROR", code)
    errors.add(code)
    return np.nan

gap_all['SIC_SECTOR'] = gap_all['SICCode.SicText_1'].map(to_code_range)

errors
#create a factor plot for exploring whether the combination of size & industry affect the hourly paygap.

with sns.axes_style("darkgrid"):
    ax = sns.factorplot(kind='box', y='DiffMedianHourlyPercent', hue='EmployerSizeCenter', x='SIC_SECTOR',
                   data=gap_all, size=7, aspect=3, legend_out=False) 
    ax.set(ylim=(-50, 50))
    ax.set_xticklabels(rotation=90)
#create a factor plot for exploring whether the combination of size & industry affect the bonus payment.

with sns.axes_style("darkgrid"):
    ax = sns.factorplot(kind='box', y='DiffMedianBonusPercent', hue='EmployerSizeCenter', x='SIC_SECTOR',
                   data=gap_all, size=7, aspect=3, legend_out=False) 
    ax.set(ylim=(-100, 100))
    ax.set_xticklabels(rotation=90)
# Create a factor plot for exploring whether the combination of size & industry affect the female representation 
# in the top quartile.

with sns.axes_style("darkgrid"):
    ax = sns.factorplot(kind='box', y='FemaleTopQuartile', hue='EmployerSizeCenter', x='SIC_SECTOR',
                   data=gap_all, size=7, aspect=3, legend_out=False) 
    ax.set(ylim=(0, 100))
    ax.set_xticklabels(rotation=90)