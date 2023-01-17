# zipfile module to work on zip file
from zipfile import ZipFile

import pandas as pd
from pprint import pprint
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from bokeh.io import output_notebook, show
from bokeh.models import ColumnDataSource, FactorRange
from bokeh.plotting import figure
from bokeh.transform import factor_cmap
#with ZipFile('human-resources-data-set.zip', 'r') as zipf:
    # print all contents of zip file
    #zipf.printdir()
    # extracting all the files
    #zipf.extractall()
df = pd.read_csv('../input/human-resources-data-set/HRDataset_v13.csv')
df.head()
df.info()
df.dropna(how = 'all', axis = 0, inplace=True)
df['DateofTermination'].unique()
print('Number of null values in TermReason column:')
print(df['TermReason'].isnull().sum())
    
df['TermReason'].value_counts()
df[['DateofTermination', 'TermReason']].head(10)
df[df['DateofTermination'].notnull()]['TermReason'].unique().tolist()
print('Number of job leaves without any reason:')
df[df['DateofTermination'].notnull()]['TermReason'].isnull().sum()
print(df[df['DateofTermination'].isnull()]['TermReason'].value_counts())
print('\n Number of employees whose DateofTermination and TermReason are both unknown:')
df[df['DateofTermination'].isnull()]['TermReason'].isnull().sum()
df.ManagerID.unique()
df.ManagerName.unique()
mng = df[['ManagerID', 'ManagerName']]
mng_id = list(df.ManagerID.unique())
def check_mng(dataframe):
    for i in mng_id:
        print('ManagerID: ' + str(i) + '...' + str(dataframe[dataframe['ManagerID'] == i]['ManagerName'].unique().tolist()))
    # since there is NaN value, let's write a code to this this case specifically
    print('ManagerID: ' + 'nan' + '...' + str(dataframe[dataframe['ManagerID'].isnull()]['ManagerName'].unique().tolist()))
check_mng(mng)
# replace NaN value in ManagerID by 39
df.ManagerID.fillna(39.0, inplace = True)
df[df.ManagerName == 'Brandon R. LeBlanc'][['ManagerID', 'Department']]
df.ManagerID.replace(3.0, 1.0, inplace = True)
df[df.ManagerName == 'Michael Albert'][['ManagerID', 'Department']]
df.ManagerID.replace(30.0, 22.0, inplace = True)
df.ManagerID.unique()
df[['LastPerformanceReview_Date', 'DaysLateLast30']].head(10)
df[df.LastPerformanceReview_Date.isnull()].DaysLateLast30.unique()
print('Number of employees with no LastPerformanceReview_Date:')
df.LastPerformanceReview_Date.isnull().sum()
df[df.LastPerformanceReview_Date.isnull()].DateofTermination.unique()
df.select_dtypes('float').head(10)
df.select_dtypes('float').columns
# select the columns that I want to convert float values into integer values
cols = ['EmpID', 'MarriedID', 'MaritalStatusID', 'GenderID', 'EmpStatusID',
       'DeptID', 'PerfScoreID', 'FromDiversityJobFairID', 'Termd',
       'PositionID', 'Zip', 'ManagerID', 'SpecialProjectsCount', 'DaysLateLast30']
for col in cols:
    df[col] = df[col].astype('Int32')
df.head()
# get rid of space in Department and Sex values
df.Sex = df.Sex.apply(lambda x: x.strip())
df.Department = df.Department.apply(lambda x: x.strip())
emp = df[df.DateofTermination.isnull()]
emp.shape
print('PayRate count based on Sex and Department')
pd.pivot_table(emp[['Sex', 'Department', 'PayRate']], index=['Sex'],
                    columns=['Department'], values = ['PayRate'], aggfunc=lambda x: int(len(x)))
print('PayRate mean based on Sex and Department')
pr = pd.pivot_table(emp[['Sex', 'Department', 'PayRate']], index=['Sex'],
                    columns=['Department'], values = ['PayRate'], aggfunc=np.mean)
pr
# PLOT
plt.figure(figsize=(10,5))
bplot=sns.stripplot(y='PayRate', x='Department', data=emp, jitter=True, dodge=True, marker='o', alpha=0.8, hue='Sex')
bplot.legend(loc='upper left')

plt.xticks(rotation=60, horizontalalignment='right');
print('PerfScoreID mean based on Sex and Department')
pfm = pd.pivot_table(emp[['Sex', 'Department', 'PerfScoreID']], index=['Sex'],
                    columns=['Department'], values = ['PerfScoreID'], aggfunc=np.mean)
pfm
# PLOT
plt.figure(figsize=(10,4))
swarm=sns.swarmplot(y='PerfScoreID', x='Department', data=emp, dodge=True, marker='o', alpha=0.8, hue='Sex')
swarm.legend(loc='lower right', ncol = 2)

plt.xticks(rotation=60, horizontalalignment='right');
df.PerfScoreID.unique()
df.PerformanceScore.unique()
perf = df[['PerfScoreID', 'PerformanceScore']]
perf.head(10)
def check_perf(dataframe):
    for i in range(1,5):
        print(dataframe[dataframe['PerfScoreID'] == i]['PerformanceScore'].unique().tolist())
check_perf(perf)
perf4 = emp[emp.PerfScoreID == 4].groupby('Department').count()['EmpID']
perf4
per_dep = emp.groupby('Department').count()['EmpID']
percent = pd.merge(perf4, per_dep, on='Department', how='left', suffixes=('_count_perf4', '_count'))
percent['Percentage'] = percent.apply(lambda row: row.EmpID_count_perf4/row.EmpID_count*100, axis = 1) 
percent
print('EngagementSurvey mean based on Sex and Department')
eng_s = pd.pivot_table(emp[['Sex', 'Department', 'EngagementSurvey']], index=['Sex'],
                    columns=['Department'], values = ['EngagementSurvey'], aggfunc=np.mean)
eng_s
# PLOT
plt.figure(figsize=(10,5))
sns.set(style="ticks", palette="pastel")

box = sns.boxplot(x="Department", y="EngagementSurvey",
            hue="Sex", palette=["m", "g"],
            data=emp)
sns.despine(offset=10, trim=True)
box.legend(loc='lower right', ncol = 2)

plt.xticks(rotation=60, horizontalalignment='right');
print('EmpSatisfaction mean based on Sex and Department')
emp_s = pd.pivot_table(emp[['Sex', 'Department', 'EmpSatisfaction']], index=['Sex'],
                    columns=['Department'], values = ['EmpSatisfaction'], aggfunc=np.mean)
emp_s
df.EmpSatisfaction.unique()
# PLOT
plt.figure(figsize=(10,4))
sns.set(style="whitegrid", palette="pastel", color_codes=True)
violin = sns.violinplot(x="Department", y="EmpSatisfaction", hue="Sex",
               split=True,
               palette={"F": "y", "M": "b"},
               data=emp, scale = 'count')
sns.despine(left=True)
violin.legend(loc='upper right', ncol = 2)
violin.set_xticklabels(violin.get_xticklabels(), rotation=45, horizontalalignment='right');
corr = df[['PayRate', 'PerfScoreID', 'EngagementSurvey', 'EmpSatisfaction']].corr()
corr
f, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(corr, cmap='Accent');
emp.RaceDesc.value_counts(normalize = True)
print('PayRate count based on Race and Department')
pd.pivot_table(emp[['RaceDesc', 'Department', 'PayRate']], index=['RaceDesc'],
                    columns=['Department'], values = ['PayRate'], aggfunc=lambda x: int(len(x)))
