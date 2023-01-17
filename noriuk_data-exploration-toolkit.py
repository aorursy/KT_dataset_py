# Install a dependency not included in Kaggle's defaults
!pip install us
# Imports
import matplotlib.pyplot as plt # plotting
import seaborn as sns # plot formatting
from ipywidgets import * # interactive plots
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import us # list of state names
# List local files
print(os.listdir('../input'))
# Load data
nRowsRead = None # specify 'None' if want to read whole file
base_df = pd.read_csv('../input/states_all.csv', delimiter=',', nrows = nRowsRead)


base_df.dataframeName = 'states_all.csv'
nRow, nCol = base_df.shape
print(f'There are {nRow} rows and {nCol} columns')
# Subset of states-only data
'''
Data for U.S. Territories is fairly erratic; limiting records to states simplifies preliminary analysis.
'''
STATES = [str(x).upper() for x in us.STATES]
states_df = base_df[base_df['STATE'].isin(STATES)].copy()
print('Initial Count: {}'.format(len(base_df)))
print('States-Only Count: {}'.format(len(states_df)))
# List nulls
for col in states_df.columns:
    nulls = states_df[col].isnull().sum()
    print(col + ': ' + str(nulls))
# Sample data
display(states_df.head(5))
YEAR_RANGE=states_df['YEAR'].unique()
display(YEAR_RANGE)
finance_keys = ['PRIMARY_KEY', 'STATE', 'YEAR',
                      'ENROLL', 'TOTAL_REVENUE', 'FEDERAL_REVENUE', 'STATE_REVENUE', 'LOCAL_REVENUE',
                      'TOTAL_EXPENDITURE', 'INSTRUCTION_EXPENDITURE', 'SUPPORT_SERVICES_EXPENDITURE',
                      'CAPITAL_OUTLAY_EXPENDITURE', 'OTHER_EXPENDITURE']
finance_keys_mappings = ['ENROLL', 'TOTAL_REVENUE', 'FEDERAL_REVENUE', 'STATE_REVENUE', 'LOCAL_REVENUE',
                      'TOTAL_EXPENDITURE', 'INSTRUCTION_EXPENDITURE', 'SUPPORT_SERVICES_EXPENDITURE',
                      'CAPITAL_OUTLAY_EXPENDITURE', 'OTHER_EXPENDITURE']
finance_df = states_df[finance_keys]
# Interactive plot
# Metrics by state, by year
%matplotlib notebook

sort_by_keys_mappings = ['Numerical', 'Alphabetical']

@interact(year=(YEAR_RANGE[0],YEAR_RANGE[-1],1), metric=finance_keys_mappings, sort_by=sort_by_keys_mappings)
def metric_explorer(year, metric, sort_by):
    # Clear any old figures
    plt.close()
    plt.style.use('ggplot')
    
    # Take a snapshot of the data for the given year
    snapshot = finance_df[finance_df['YEAR'] == year].copy()
    if sort_by == 'Alphabetical':
        snapshot.sort_values('STATE', ascending=False, inplace=True)
    else:
        snapshot.sort_values(metric, ascending=True, inplace=True)
    y_pos = np.arange(len(snapshot[metric]))
    
    # Make a plot to match states to the chosen metric
    plt.figure(figsize=(8, 10), num='Financial Metric Explorer Tool')
    plt.barh(y_pos, snapshot[metric])
    plt.yticks(y_pos, snapshot['STATE'])
    plt.title('{0}: {1}'.format(metric, year))

    
interactive_plot = interactive(metric_explorer, year=2005, metric=finance_keys_mappings[0], sort_by=sort_by_keys_mappings[0])
# Interactive plot
# Metrics across years, by state
%matplotlib notebook

@interact(state=STATES, metric=finance_keys_mappings)
def state_explorer(state, metric):
    # Clear any old figures
    plt.close()
    plt.style.use('ggplot')
    
    # Take a snapshot of the data for the given year
    snapshot = finance_df[finance_df['STATE'] == state].copy()
    
    # Make a plot to match states to the chosen metric
    plt.figure(figsize=(8, 10), num='Financial State Explorer Tool')
    plt.plot(YEAR_RANGE, snapshot[metric], '-o')
    plt.title('{0}: {1}'.format(metric, state))

    
interactive_plot = interactive(state_explorer, state='ALABAMA', metric=finance_keys_mappings[0])
enrollment_keys = ['PRIMARY_KEY', 'STATE', 'YEAR',
                      'GRADES_PK_G', 'GRADES_KG_G', 'GRADES_4_G', 'GRADES_8_G', 'GRADES_12_G',
                      'GRADES_1_8_G', 'GRADES_9_12_G', 'GRADES_ALL_G', 'ENROLL']
enrollment_keys_mappings = ['GRADES_PK_G', 'GRADES_KG_G', 'GRADES_4_G', 'GRADES_8_G', 'GRADES_12_G',
                      'GRADES_1_8_G', 'GRADES_9_12_G', 'GRADES_ALL_G', 'ENROLL']
sort_by_keys_mappings = ['Numerical', 'Alphabetical']
enroll_df = states_df[enrollment_keys]
# Interactive plot
# Metrics by state, by year
%matplotlib notebook

@interact(year=(YEAR_RANGE[0],YEAR_RANGE[-1],1), metric=enrollment_keys_mappings, sort_by=sort_by_keys_mappings)
def metric_explorer(year, metric, sort_by):
    # Clear any old figures
    plt.close()
    plt.style.use('ggplot')
    
    # Take a snapshot of the data for the given year
    snapshot = enroll_df[finance_df['YEAR'] == year].copy()
    if sort_by == 'Alphabetical':
        snapshot.sort_values('STATE', ascending=False, inplace=True)
    else:
        snapshot.sort_values(metric, ascending=True, inplace=True)
    y_pos = np.arange(len(snapshot[metric]))
    
    # Make a plot to match states to the chosen metric
    plt.figure(figsize=(8, 10), num='Enrollment Metric Explorer Tool')
    plt.barh(y_pos, snapshot[metric], color='blue')
    plt.yticks(y_pos, snapshot['STATE'])
    plt.title('{0}: {1}'.format(metric, year))

    
interactive_plot = interactive(metric_explorer, year=2005, metric=enrollment_keys_mappings[0], sort_by=sort_by_keys_mappings[0])
# Interactive plot
# Metrics across years, by state
%matplotlib notebook

@interact(state=STATES, metric=enrollment_keys_mappings)
def state_explorer(state, metric):
    # Clear any old figures
    plt.close()
    plt.style.use('ggplot')
    
    # Take a snapshot of the data for the given year
    snapshot = enroll_df[enroll_df['STATE'] == state].copy()
    
    # Make a plot to match states to the chosen metric
    plt.figure(figsize=(8, 10), num='Enrollment State Explorer Tool')
    plt.plot(YEAR_RANGE, snapshot[metric], '-bo')
    plt.title('{0}: {1}'.format(metric, state))

    
interactive_plot = interactive(state_explorer, state='ALABAMA', metric=enrollment_keys_mappings[0])
acad_keys = ['PRIMARY_KEY', 'STATE', 'YEAR',
                      'AVG_MATH_4_SCORE', 'AVG_MATH_8_SCORE', 'AVG_READING_4_SCORE', 'AVG_READING_8_SCORE']
acad_keys_mappings = ['AVG_MATH_4_SCORE', 'AVG_MATH_8_SCORE', 'AVG_READING_4_SCORE', 'AVG_READING_8_SCORE']

sort_by_keys_mappings = ['Numerical', 'Alphabetical']
acad_df = states_df[acad_keys]
# Interactive plot
# Metrics by state, by year
%matplotlib notebook

@interact(year=(YEAR_RANGE[0],YEAR_RANGE[-1],1), metric=acad_keys_mappings, sort_by=sort_by_keys_mappings)
def metric_explorer(year, metric, sort_by):
    # Clear any old figures
    plt.close()
    plt.style.use('ggplot')
    
    # Take a snapshot of the data for the given year
    snapshot = acad_df[acad_df['YEAR'] == year].copy()
    if sort_by == 'Alphabetical':
        snapshot.sort_values('STATE', ascending=False, inplace=True)
    else:
        snapshot.sort_values(metric, ascending=True, inplace=True)
    y_pos = np.arange(len(snapshot[metric]))
    
    # Make a plot to match states to the chosen metric
    plt.figure(figsize=(8, 10), num='Academic Metric Explorer Tool')
    plt.barh(y_pos, snapshot[metric], color='green')
    plt.yticks(y_pos, snapshot['STATE'])
    plt.title('{0}: {1}'.format(metric, year))

    
interactive_plot = interactive(metric_explorer, year=2009, metric=acad_keys_mappings[0], sort_by=sort_by_keys_mappings[0])
# Interactive plot
# Metrics across years, by state
%matplotlib notebook

@interact(state=STATES, metric=acad_keys_mappings)
def state_explorer(state, metric):
    # Clear any old figures
    plt.close()
    plt.style.use('ggplot')
    
    # Take a snapshot of the data for the given year
    snapshot = acad_df[acad_df['STATE'] == state].copy()
    
    # Make a plot to match states to the chosen metric
    plt.figure(figsize=(8, 10), num='Academics State Explorer Tool')
    # Mask missing values to allow for a line plot
    null_mask = np.isfinite(snapshot[metric])
    plt.plot(YEAR_RANGE[null_mask], snapshot[metric][null_mask], '-go')
    plt.title('{0}: {1}'.format(metric, state))

    
interactive_plot = interactive(state_explorer, state='ALABAMA', metric=acad_keys_mappings[0])
# Compute the correlation matrix
corr = states_df.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
