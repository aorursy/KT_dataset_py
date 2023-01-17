import pandas as pd
temperature = [34, 56, 15, -9, -121, -5, 39]
days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']

# create series 
series_from_list = pd.Series(temperature, index=days)
series_from_list
temperature = [34, 56, 'a', -9, -121, -5, 39]
days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
series_from_list = pd.Series(temperature, index=days)
series_from_list
my_dict = {'Mon': 33, 'Tue': 19, 'Wed': 15, 'Thu': 89, 'Fri': 11, 'Sat': -5, 'Sun': 9}
my_dict
series_from_dict = pd.Series(my_dict)
series_from_dict
import numpy as np
my_array = np.linspace(0,10,15)
my_array
series_from_ndarray = pd.Series(my_array)
series_from_ndarray
data = pd.read_excel(io="../input/WA_Fn-UseC_-HR-Employee-Attrition.xlsx", sheetname=0, index_col='EmployeeNumber')
data.head()
data.columns
data['Attrition'].head()
data[['Age', 'Gender','YearsAtCompany']].head()
data['AgeInMonths'] = 12*data['Age']
data['AgeInMonths'].head()
del data['AgeInMonths']
data.columns
data['BusinessTravel'][10:15]
data[10:15]
selected_EmployeeNumbers = [15, 94, 337, 1120]
data['YearsAtCompany'].loc[selected_EmployeeNumbers]
data.loc[selected_EmployeeNumbers]
data.loc[94,'YearsAtCompany']
data['Department'].value_counts()
data['Department'].value_counts().plot(kind='barh', title='Department')
data['Department'].value_counts().plot(kind='pie', title='Department')
data['Attrition'].value_counts()
data['Attrition'].value_counts(normalize=True)
data['HourlyRate'].mean()
data['JobSatisfaction'].head()
JobSatisfaction_cat = {
    1: 'Low',
    2: 'Medium',
    3: 'High',
    4: 'Very High'
}
data['JobSatisfaction'] = data['JobSatisfaction'].map(JobSatisfaction_cat)
data['JobSatisfaction'].head()
data['JobSatisfaction'].value_counts()
100*data['JobSatisfaction'].value_counts(normalize=True)
data['JobSatisfaction'].value_counts(normalize=True).plot(kind='pie', title='Department')
data['JobSatisfaction'] = data['JobSatisfaction'].astype(dtype='category', 
                               categories=['Low', 'Medium', 'High', 'Very High'],
                               ordered=True)
data['JobSatisfaction'].head()
data['JobSatisfaction'].value_counts().plot(kind='barh', title='Department')
data['JobSatisfaction'].value_counts(sort=False).plot(kind='barh', title='Department')
data['JobSatisfaction'] == 'Low'
data.loc[data['JobSatisfaction'] == 'Low'].index
data['JobInvolvement'].head()
subset_of_interest = data.loc[(data['JobSatisfaction'] == "Low") | (data['JobSatisfaction'] == "Very High")]
subset_of_interest.shape
subset_of_interest['JobSatisfaction'].value_counts()
subset_of_interest['JobSatisfaction'].cat.remove_unused_categories(inplace=True)
grouped = subset_of_interest.groupby('JobSatisfaction')
grouped.groups
grouped.get_group('Low').head()
grouped.get_group('Very High').head()
grouped['Age']
grouped['Age'].mean()
grouped['Age'].describe()
grouped['Age'].describe().unstack()
grouped['Age'].plot(kind='density', title='Age')
grouped['Department'].value_counts().unstack()
grouped['Department'].value_counts(normalize=True).unstack()
grouped['Department'].value_counts().unstack().plot(kind="barh")
grouped['Department'].value_counts(normalize=True).unstack().plot(kind="barh")
data['Department'].value_counts(normalize=True,sort=False).plot(kind="barh")
data['Department'] = data['Department'].astype(dtype='category', 
                               categories=['Human Resources', 'Research & Development', 'Sales'],
                               ordered=True)
data['Department'].value_counts(normalize=True,sort=False).plot(kind="barh")
grouped['DistanceFromHome'].describe().unstack()
grouped['DistanceFromHome'].plot(kind='density', title='Distance From Home',legend=True)
grouped['HourlyRate'].describe()
grouped['HourlyRate'].plot(kind='density', title='Hourly Rate',legend=True)
grouped['MonthlyIncome'].describe()
grouped['HourlyRate'].plot(kind='density', title='Hourly Rate',legend=True)