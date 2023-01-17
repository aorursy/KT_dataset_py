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
# Suppressing Warnings

import warnings

warnings.filterwarnings('ignore')
# Importing Pandas and NumPy

import pandas as pd, numpy as np, seaborn as sns,matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
in_time = pd.read_csv('/kaggle/input/hr-analytics-case-study/in_time.csv')

manager_survey_data = pd.read_csv('/kaggle/input/hr-analytics-case-study/manager_survey_data.csv')

employee_survey_data = pd.read_csv('/kaggle/input/hr-analytics-case-study/employee_survey_data.csv')

data_dictionary= pd.read_excel('/kaggle/input/hr-analytics-case-study/data_dictionary.xlsx')

out_time = pd.read_csv('/kaggle/input/hr-analytics-case-study/out_time.csv')

general_data = pd.read_csv('/kaggle/input/hr-analytics-case-study/general_data.csv')

in_time.head()
in_time.shape
in_time=in_time.replace(np.nan,0)

in_time.head()


in_time.iloc[:, 1:] = in_time.iloc[:, 1:].apply(pd.to_datetime, errors='coerce')
in_time.head()
out_time=out_time.replace(np.nan,0)

out_time.head()
out_time.iloc[:, 1:] = out_time.iloc[:, 1:].apply(pd.to_datetime, errors='coerce')
out_time.head()
in_time=in_time.append(out_time)
in_time.head()
in_time=in_time.diff(periods=4410)

in_time=in_time.iloc[4410:]

in_time.reset_index(inplace=True)

in_time.head()
in_time.drop(columns=['index','Unnamed: 0'],axis=1,inplace=True)

in_time.head()
in_time.shape
in_time.drop(['2015-01-01', '2015-01-14','2015-01-26','2015-03-05',

             '2015-05-01','2015-07-17','2015-09-17','2015-10-02',

              '2015-11-09','2015-11-10','2015-11-11','2015-12-25'

             ], axis = 1,inplace=True) 
in_time.head()
in_time['Actual Time']=in_time.mean(axis=1)
in_time['Actual Time'].head()
in_time['hrs']=in_time['Actual Time']/np.timedelta64(1, 'h')

in_time.head()
in_time.reset_index(inplace=True)

in_time.head()
in_time.drop(in_time.columns.difference(['index','hrs']), 1, inplace=True)
in_time.rename(columns={'index': 'EmployeeID'},inplace=True)

in_time.head()
general_data.head()
employee_survey_data.head()
df_1 = pd.merge(employee_survey_data, general_data, how='inner', on='EmployeeID')

hr = pd.merge(manager_survey_data, df_1, how='inner', on='EmployeeID')

hr = pd.merge(in_time, hr, how='inner', on='EmployeeID')

hr.head()
hr.describe()
# Correcting Datatype for the variable
hr['JobLevel']=hr['JobLevel'].astype('object')
hr['Education'] = hr['Education'].replace({ 1 : 'Below College', 2: 'College',3: 'Bachelor',4: 'Master',5 : 'Doctor'})

hr['EnvironmentSatisfaction'] = hr['EnvironmentSatisfaction'].replace({ 1 : 'Low', 2: 'Medium',3: 'High',4: 'Very High'})

hr['JobInvolvement'] = hr['JobInvolvement'].replace({ 1 : 'Low', 2: 'Medium',3: 'High',4: 'Very High'})

hr['JobSatisfaction'] = hr['JobSatisfaction'].replace({ 1 : 'Low', 2: 'Medium',3: 'High',4: 'Very High'})

#hr['RelationshipSatisfaction'] = hr['RelationshipSatisfaction'].replace({ 1 : 'Low', 2: 'Medium',

#                   3: 'High',4: 'Very High'})

hr['PerformanceRating'] = hr['PerformanceRating'].replace({ 1 : 'Low', 2: 'Good',3: 'Excellent',4: 'Outstanding'})

hr['WorkLifeBalance'] = hr['WorkLifeBalance'].replace({ 1 : 'Bad', 2: 'Good',3: 'Better',4: 'Best'})
hr.head()
hr['EmployeeCount'].value_counts(ascending=False)
hr['Over18'].value_counts(ascending=False)
hr['StandardHours'].value_counts(ascending=False)
hr.drop(['EmployeeID', 'EmployeeCount','StandardHours','Over18'], axis = 1,inplace=True) 
# Let's see the head of our master dataset

hr.head()
# Let's check the dimensions of the dataframe

hr.shape
# let's look at the statistical aspects of the dataframe

hr.describe()
# Let's see the type of each column

hr.info()
hr['EnvironmentSatisfaction'].value_counts(ascending=False)
sns.countplot(x='EnvironmentSatisfaction',data=hr);
hr['EnvironmentSatisfaction'] = hr['EnvironmentSatisfaction'].fillna('High')

hr['EnvironmentSatisfaction'].isnull().sum()
hr['JobSatisfaction'].value_counts(ascending=False)
sns.countplot(x='JobSatisfaction',data=hr);
hr['JobSatisfaction'] = hr['JobSatisfaction'].fillna('High')

hr['JobSatisfaction'].isnull().sum()
hr['WorkLifeBalance'].value_counts(ascending=False)
sns.countplot(x='WorkLifeBalance',data=hr);
hr['WorkLifeBalance'] = hr['WorkLifeBalance'].fillna('Better')

hr['WorkLifeBalance'].isnull().sum()
hr['NumCompaniesWorked'].value_counts(ascending=False)
sns.countplot(x='NumCompaniesWorked',data=hr);
sns.boxplot(x='NumCompaniesWorked',data=hr);
hr['NumCompaniesWorked'] = hr['NumCompaniesWorked'].fillna(2)

hr['NumCompaniesWorked'].isnull().sum()
hr['TotalWorkingYears'].value_counts(ascending=False)
plt.figure(figsize=(8,8))

ax = sns.distplot(hr['TotalWorkingYears'], hist=True, kde=False, 

             bins=int(180/5), color = 'darkblue', 

             hist_kws={'edgecolor':'black'},

             kde_kws={'linewidth': 4})

ax.set_ylabel('# of Employees')

ax.set_xlabel('TotalWorkingYears');

sns.boxplot(x='TotalWorkingYears',data=hr);
hr['TotalWorkingYears'] = hr['TotalWorkingYears'].fillna(2)

hr['TotalWorkingYears'].isnull().sum()
hr.info()
plt.figure(figsize=(8,8))

ax = sns.countplot(x='WorkLifeBalance',data=hr,hue="Attrition")

bars = ax.patches

half = int(len(bars)/2)

left_bars = bars[:half]

right_bars = bars[half:]



for left, right in zip(left_bars, right_bars):

    height_l = left.get_height()

    height_r = right.get_height()

    total = height_l + height_r



    ax.text(left.get_x() + left.get_width()/2., height_l + 40, '{0:.0%}'.format(height_l/total), ha="center")

    ax.text(right.get_x() + right.get_width()/2., height_r + 40, '{0:.0%}'.format(height_r/total), ha="center")
plt.figure(figsize=(8,8))

ax = sns.countplot(x='PerformanceRating', data=hr, hue="Attrition")

ax.set_ylabel('# of Employee')

bars = ax.patches

half = int(len(bars)/2)

left_bars = bars[:half]

right_bars = bars[half:]



for left, right in zip(left_bars, right_bars):

    height_l = left.get_height()

    height_r = right.get_height()

    total = height_l + height_r



    ax.text(left.get_x() + left.get_width()/2., height_l + 40, '{0:.0%}'.format(height_l/total), ha="center")

    ax.text(right.get_x() + right.get_width()/2., height_r + 40, '{0:.0%}'.format(height_r/total), ha="center")
plt.figure(figsize=(8,10))

ax = sns.countplot(x='EnvironmentSatisfaction', data=hr, hue="Attrition")

ax.set_ylabel('# of Employee')

bars = ax.patches

half = int(len(bars)/2)

left_bars = bars[:half]

right_bars = bars[half:]



for left, right in zip(left_bars, right_bars):

    height_l = left.get_height()

    height_r = right.get_height()

    total = height_l + height_r



    ax.text(left.get_x() + left.get_width()/2., height_l + 30, '{0:.0%}'.format(height_l/total), ha="center")

    ax.text(right.get_x() + right.get_width()/2., height_r + 40, '{0:.0%}'.format(height_r/total), ha="center")
plt.figure(figsize=(8,8))

ax = sns.countplot(x='JobSatisfaction', data=hr, hue="Attrition")

ax.set_ylabel('# of Employee')

bars = ax.patches

half = int(len(bars)/2)

left_bars = bars[:half]

right_bars = bars[half:]



for left, right in zip(left_bars, right_bars):

    height_l = left.get_height()

    height_r = right.get_height()

    total = height_l + height_r



    ax.text(left.get_x() + left.get_width()/2., height_l + 40, '{0:.0%}'.format(height_l/total), ha="center")

    ax.text(right.get_x() + right.get_width()/2., height_r + 40, '{0:.0%}'.format(height_r/total), ha="center")
plt.figure(figsize=(8,8))

ax = sns.countplot(x='WorkLifeBalance', data=hr, hue="Attrition")

ax.set_ylabel('# of Employee')

bars = ax.patches

half = int(len(bars)/2)

left_bars = bars[:half]

right_bars = bars[half:]



for left, right in zip(left_bars, right_bars):

    height_l = left.get_height()

    height_r = right.get_height()

    total = height_l + height_r



    ax.text(left.get_x() + left.get_width()/2., height_l + 40, '{0:.0%}'.format(height_l/total), ha="center")

    ax.text(right.get_x() + right.get_width()/2., height_r + 40, '{0:.0%}'.format(height_r/total), ha="center")
plt.figure(figsize=(8,8))

sns.violinplot(y='Age',x='Attrition',data=hr)

plt.show()
plt.figure(figsize=(8,8))

ax = sns.countplot(x='BusinessTravel', data=hr, hue="Attrition")

ax.set_ylabel('# of Employee')

bars = ax.patches

half = int(len(bars)/2)

left_bars = bars[:half]

right_bars = bars[half:]



for left, right in zip(left_bars, right_bars):

    height_l = left.get_height()

    height_r = right.get_height()

    total = height_l + height_r



    ax.text(left.get_x() + left.get_width()/2., height_l + 40, '{0:.0%}'.format(height_l/total), ha="center")

    ax.text(right.get_x() + right.get_width()/2., height_r + 40, '{0:.0%}'.format(height_r/total), ha="center")
plt.figure(figsize=(8,8))

ax = sns.countplot(x='Department', data=hr, hue="Attrition")

ax.set_ylabel('# of Employee')

bars = ax.patches

half = int(len(bars)/2)

left_bars = bars[:half]

right_bars = bars[half:]



for left, right in zip(left_bars, right_bars):

    height_l = left.get_height()

    height_r = right.get_height()

    total = height_l + height_r



    ax.text(left.get_x() + left.get_width()/2., height_l + 40, '{0:.0%}'.format(height_l/total), ha="center")

    ax.text(right.get_x() + right.get_width()/2., height_r + 40, '{0:.0%}'.format(height_r/total), ha="center")
plt.figure(figsize=(8,8))

sns.violinplot(y='DistanceFromHome',x='Attrition',data=hr)



plt.show()
plt.figure(figsize=(8,8))

ax = sns.countplot(x='Education', data=hr, hue="Attrition")

ax.set_ylabel('# of Employee')

bars = ax.patches

half = int(len(bars)/2)

left_bars = bars[:half]

right_bars = bars[half:]



for left, right in zip(left_bars, right_bars):

    height_l = left.get_height()

    height_r = right.get_height()

    total = height_l + height_r



    ax.text(left.get_x() + left.get_width()/2., height_l + 40, '{0:.0%}'.format(height_l/total), ha="center")

    ax.text(right.get_x() + right.get_width()/2., height_r + 40, '{0:.0%}'.format(height_r/total), ha="center")
plt.figure(figsize=(15,8))

ax = sns.countplot(x='EducationField', data=hr, hue="Attrition")

ax.set_ylabel('# of Employee')

bars = ax.patches

half = int(len(bars)/2)

left_bars = bars[:half]

right_bars = bars[half:]



for left, right in zip(left_bars, right_bars):

    height_l = left.get_height()

    height_r = right.get_height()

    total = height_l + height_r



    ax.text(left.get_x() + left.get_width()/2., height_l + 40, '{0:.0%}'.format(height_l/total), ha="center")

    ax.text(right.get_x() + right.get_width()/2., height_r + 40, '{0:.0%}'.format(height_r/total), ha="center")
plt.figure(figsize=(8,8))

ax = sns.countplot(x='Gender', data=hr, hue="Attrition")

ax.set_ylabel('# of Employee')

bars = ax.patches

half = int(len(bars)/2)

left_bars = bars[:half]

right_bars = bars[half:]



for left, right in zip(left_bars, right_bars):

    height_l = left.get_height()

    height_r = right.get_height()

    total = height_l + height_r



    ax.text(left.get_x() + left.get_width()/2., height_l + 40, '{0:.0%}'.format(height_l/total), ha="center")

    ax.text(right.get_x() + right.get_width()/2., height_r + 40, '{0:.0%}'.format(height_r/total), ha="center")
plt.figure(figsize=(8,8))

ax = sns.countplot(x='JobLevel', data=hr, hue="Attrition")

ax.set_ylabel('# of Employee')

bars = ax.patches

half = int(len(bars)/2)

left_bars = bars[:half]

right_bars = bars[half:]



for left, right in zip(left_bars, right_bars):

    height_l = left.get_height()

    height_r = right.get_height()

    total = height_l + height_r



    ax.text(left.get_x() + left.get_width()/2., height_l + 40, '{0:.0%}'.format(height_l/total), ha="center")

    ax.text(right.get_x() + right.get_width()/2., height_r + 40, '{0:.0%}'.format(height_r/total), ha="center")
plt.figure(figsize=(20,8))

ax = sns.countplot(x='JobRole', data=hr, hue="Attrition")

ax.set_ylabel('# of Employee')

bars = ax.patches

half = int(len(bars)/2)

left_bars = bars[:half]

right_bars = bars[half:]



for left, right in zip(left_bars, right_bars):

    height_l = left.get_height()

    height_r = right.get_height()

    total = height_l + height_r



    ax.text(left.get_x() + left.get_width()/2., height_l + 40, '{0:.0%}'.format(height_l/total), ha="center")

    ax.text(right.get_x() + right.get_width()/2., height_r + 40, '{0:.0%}'.format(height_r/total), ha="center")
plt.figure(figsize=(8,8))

ax = sns.countplot(x='MaritalStatus', data=hr, hue="Attrition")

ax.set_ylabel('# of Employee')

bars = ax.patches

half = int(len(bars)/2)

left_bars = bars[:half]

right_bars = bars[half:]



for left, right in zip(left_bars, right_bars):

    height_l = left.get_height()

    height_r = right.get_height()

    total = height_l + height_r



    ax.text(left.get_x() + left.get_width()/2., height_l + 40, '{0:.0%}'.format(height_l/total), ha="center")

    ax.text(right.get_x() + right.get_width()/2., height_r + 40, '{0:.0%}'.format(height_r/total), ha="center")
plt.figure(figsize=(8,8))

sns.violinplot(y='MonthlyIncome',x='Attrition',data=hr)



plt.show()
plt.figure(figsize=(8,8))

sns.violinplot(y='PercentSalaryHike',x='Attrition',data=hr)



plt.show()
plt.figure(figsize=(8,8))

ax = sns.countplot(x='StockOptionLevel', data=hr, hue="Attrition")

ax.set_ylabel('# of Employee')

bars = ax.patches

half = int(len(bars)/2)

left_bars = bars[:half]

right_bars = bars[half:]



for left, right in zip(left_bars, right_bars):

    height_l = left.get_height()

    height_r = right.get_height()

    total = height_l + height_r



    ax.text(left.get_x() + left.get_width()/2., height_l + 40, '{0:.0%}'.format(height_l/total), ha="center")

    ax.text(right.get_x() + right.get_width()/2., height_r + 40, '{0:.0%}'.format(height_r/total), ha="center")


plt.figure(figsize=(8,8))

sns.violinplot(y='TotalWorkingYears',x='Attrition',data=hr)



plt.show()
plt.figure(figsize=(8,8))

sns.violinplot(y='TrainingTimesLastYear',x='Attrition',data=hr)



plt.show()
plt.figure(figsize=(8,8))

sns.violinplot(y='YearsAtCompany',x='Attrition',data=hr)



plt.show()
plt.figure(figsize=(8,8))

sns.violinplot(y='YearsSinceLastPromotion',x='Attrition',data=hr)



plt.show()
plt.figure(figsize=(8,8))

sns.violinplot(y='YearsWithCurrManager',x='Attrition',data=hr)



plt.show()
plt.figure(figsize=(8,8))

sns.violinplot(y='hrs',x='Attrition',data=hr)



plt.show()
hr.columns
plt.figure(figsize=(20,18))

sns.heatmap(hr.corr(), annot = True, cmap="Accent");
hr_num=hr[[ 'MonthlyIncome', 'NumCompaniesWorked', 'PercentSalaryHike',

       'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',

       'YearsAtCompany', 'YearsSinceLastPromotion', 'YearsWithCurrManager',

           'DistanceFromHome','Age','hrs']]



sns.pairplot(hr_num, diag_kind='kde')

plt.show()