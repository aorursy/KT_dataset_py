import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

%matplotlib inline

import seaborn as sns
sns.set() 
df_emp = pd.read_csv('/kaggle/input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv') 
df_emp.head() 
df_emp.tail() 
df_emp.info() 
df_emp.describe(include='all') 
df_emp.isna().sum()
df_emp.EnvironmentSatisfaction = df_emp.EnvironmentSatisfaction.astype('object')

df_emp.JobSatisfaction = df_emp.JobSatisfaction.astype('object')

df_emp.PerformanceRating = df_emp.PerformanceRating.astype('object')

df_emp.WorkLifeBalance = df_emp.WorkLifeBalance.astype('object')

df_emp.info()
df_emp.isnull().sum()
df_emp.PerformanceRating.value_counts()
df_emp.PerformanceRating.mode()
df_emp.PerformanceRating.mode()[0]
df_emp['PerformanceRating'].fillna(df_emp['PerformanceRating'].mode(), inplace=True) 
df_emp['PerformanceRating'].isna().sum()
df_emp['HourlyRate'].describe() 
sns.boxplot(x='HourlyRate',data=df_emp) 
df_emp['HourlyRate'].fillna(df_emp['HourlyRate'].mean(), inplace=True)
df_emp.info() 
df_emp.PerformanceRating = df_emp.PerformanceRating.astype('object') 

df_emp.drop(['EmployeeNumber'], axis = 1, inplace=True) 
df_emp.shape
df_emp.Department.value_counts()
df_emp.Department.value_counts(normalize=True)
df_emp.Attrition.value_counts()
df_emp.Attrition.value_counts(normalize=True)
df_emp['Age'].describe() 
fig_dims = (10, 5) 

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=fig_dims) 

sns.distplot(df_emp.Age, kde=True, ax=axs[1]) 

sns.boxplot(x= 'Age', data=df_emp, ax=axs[0]) 
df_emp['MonthlyIncome'].describe()
fig_dims = (10, 5)

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=fig_dims)

sns.distplot(df_emp.MonthlyIncome, rug= True, kde=False, ax=axs[0]) 

sns.boxplot(x= 'MonthlyIncome', data=df_emp, color = 'm',ax=axs[1]) 
df_emp['YearsAtCompany'].describe()
fig_dims = (10, 5)

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=fig_dims)

sns.distplot(df_emp.YearsAtCompany, kde=True, ax=axs[0]) 

sns.boxplot(x= 'YearsAtCompany', data=df_emp, ax=axs[1])
df_emp['PercentSalaryHike'].describe()
fig_dims = (10, 5)

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=fig_dims)

sns.distplot(df_emp.PercentSalaryHike, kde=False, ax=axs[0])

sns.boxplot(x= 'PercentSalaryHike', data=df_emp, ax=axs[1], orient = 'v') 
def univariateAnalysis_category(cat_column):

    print("Details of " + cat_column)

    print("----------------------------------------------------------------")

    print(df_emp[cat_column].value_counts())

    sns.countplot(x=cat_column, data=df_emp, palette='pastel')

    plt.show()

    print("       ")
df_emp_object = df_emp.select_dtypes(include = ['object']) 

lstcatcolumns = list(df_emp_object.columns.values)

lstcatcolumns
for x in lstcatcolumns:

    univariateAnalysis_category(x)
ax = sns.catplot(y='JobRole', kind='count', aspect=2, data=df_emp)

# aspect signifies the width of each bar
ax = sns.countplot(x="EducationField", data=df_emp)



ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

df_emp_numeric = df_emp.select_dtypes(include = ['int64','float64'])

df_emp_numeric.shape
sns.pairplot(df_emp_numeric)
fig_dims = (10, 5)

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=fig_dims)

sns.scatterplot(x='Age', y='MonthlyIncome', data=df_emp,ax= axs[0])

sns.scatterplot(x='TotalWorkingYears', y='MonthlyIncome', data=df_emp,ax= axs[1])
corr = df_emp_numeric.corr()

round(corr,2)
fig_dims = (10,6)

#fig = plt.subplots(figsize=fig_dims)

#mask = np.triu(np.ones_like(corr, dtype=np.bool)) 

sns.heatmap(round(corr,2), annot=True,fmt='.2f', mask=(np.triu(corr,+1)))
sns.countplot(x='Attrition', hue='Gender', data=df_emp)
sns.countplot(y='OverTime', hue='Attrition', data=df_emp)
#pd.crosstab(df_emp.Attrition, df_emp.EducationField, margins=True, normalize='columns')

pd.crosstab(df_emp.Attrition, df_emp.EducationField)
pd.crosstab(df_emp.Attrition, df_emp.JobSatisfaction, margins=True)
pd.crosstab(df_emp.Attrition, df_emp.JobSatisfaction, normalize='columns')
pd.crosstab(df_emp.Attrition, df_emp.JobSatisfaction, margins=True, normalize=True)
fig_dims = (12, 5)

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=fig_dims)

sns.boxplot(x='Attrition', y='MonthlyIncome', data=df_emp, ax=axs[0])

sns.boxplot(x='Attrition', y='Age', data=df_emp, ax=axs[1])

sns.boxplot(x='Attrition', y='DistanceFromHome', data=df_emp, ax=axs[2])
fig_dims = (12, 6)

fig = plt.subplots(figsize=fig_dims)

sns.scatterplot(x='NumCompaniesWorked', y='TotalWorkingYears', hue='Attrition', data=df_emp)
fig_dims = (12, 6)

fig = plt.subplots(figsize=fig_dims)

sns.boxplot(x='Attrition', y='YearsAtCompany', hue='BusinessTravel',data=df_emp)
g = sns.FacetGrid(df_emp, col="JobRole", hue='Attrition',col_wrap=3, height=3)

g = g.map(plt.scatter, "YearsSinceLastPromotion", 'YearsAtCompany')