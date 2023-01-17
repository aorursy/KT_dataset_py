import pandas as pd, numpy as np

import matplotlib.pyplot as plt, seaborn as sns

import warnings

warnings.filterwarnings('ignore')
hr = pd.read_csv('../input/WA_Fn-UseC_-HR-Employee-Attrition.csv')
hr.shape
hr.head()
hr.info()
hr.describe()
# Number of unique values in each feature

for col in hr.columns:

    print(col, hr[col].nunique())
# Dropping column 'Over18', 'EmployeeCount' as it consists of only one value

hr.drop(['Over18', 'EmployeeCount', 'StandardHours'] , axis = 1, inplace = True)
# creating a list of the categorical variables

cat_var = ['Attrition',

'BusinessTravel',

'Department',

'Education',

'EducationField',

'EnvironmentSatisfaction',

'Gender',

'JobInvolvement',

'JobLevel',

'JobRole',

'JobSatisfaction',

'MaritalStatus',

'OverTime',

'PerformanceRating',

'RelationshipSatisfaction',

'StockOptionLevel',

'WorkLifeBalance']
# count of values of each categorical variable

for col in cat_var:

    print(hr.groupby(col)[col].count(), '\n')
# changing the dtype to 'category'

# some of the features here are ordered categorical, so handling them accordingly

hr['Attrition'] = hr['Attrition'].astype('category')

hr['BusinessTravel'] = hr['BusinessTravel'].astype('category')

hr['Department'] = hr['Department'].astype('category')

hr['Education'] = hr['Education'].astype('category', ordered = True)

hr['EducationField'] = hr['EducationField'].astype('category')

hr['EnvironmentSatisfaction'] = hr['EnvironmentSatisfaction'].astype('category', ordered = True)

hr['JobInvolvement'] = hr['JobInvolvement'].astype('category', ordered = True)

hr['JobLevel'] = hr['JobLevel'].astype('category', ordered = True)

hr['JobRole'] = hr['JobRole'].astype('category')

hr['JobSatisfaction'] = hr['JobSatisfaction'].astype('category', ordered = True)

hr['MaritalStatus'] = hr['MaritalStatus'].astype('category')

hr['OverTime'] = hr['OverTime'].astype('category')

hr['PerformanceRating'] = hr['PerformanceRating'].astype('category', ordered = True)

hr['RelationshipSatisfaction'] = hr['RelationshipSatisfaction'].astype('category', ordered = True)

hr['StockOptionLevel'] = hr['StockOptionLevel'].astype('category', ordered = True)

hr['WorkLifeBalance'] = hr['WorkLifeBalance'].astype('category', ordered = True)

hr['NumCompaniesWorked'] = hr['NumCompaniesWorked'].astype('category', ordered = True)

hr['PercentSalaryHike'] = hr['PercentSalaryHike'].astype('category', ordered = True)

hr['TrainingTimesLastYear'] = hr['TrainingTimesLastYear'].astype('category', ordered = True)
sns.catplot(x = 'Attrition', y = 'Age', kind = 'swarm', data = hr)

sns.catplot(x = 'Attrition', y = 'Age', kind = 'boxen', data = hr)

plt.show()
sns.catplot(x = 'Attrition', hue = 'BusinessTravel', data = hr, kind = 'count')

plt.show()
sns.catplot(x = 'Attrition', y = 'DailyRate', kind = 'swarm', data = hr)

sns.catplot(x = 'Attrition', y = 'DailyRate', kind = 'boxen', data = hr)

plt.show()
sns.catplot(x = 'Attrition', hue = 'Department', data = hr, kind = 'count')

plt.show()
sns.catplot(x = 'Attrition', y = 'DistanceFromHome', kind = 'swarm', data = hr)

sns.catplot(x = 'Attrition', y = 'DistanceFromHome', kind = 'boxen', data = hr)

plt.show()
p = sns.catplot(x = 'Attrition', hue = 'Education', data = hr,kind = 'count', legend_out = True)



# editing the labels

new_labels = ['1 - Below College', '2 - College', '3 - Bachelor', '4 - Master', '5 - Doctor']

for t, l in zip(p._legend.texts, new_labels): t.set_text(l)

plt.show()
sns.catplot(x = 'Attrition', hue = 'EducationField', data = hr, kind = 'count')

plt.show()
p = sns.catplot(x = 'Attrition', hue = 'EnvironmentSatisfaction', data = hr, kind = 'count', legend_out = True)



# editing the labels

new_labels = ['1 - Low', '2 - Medium', '3 - High', '4 - Very High']

for t, l in zip(p._legend.texts, new_labels): t.set_text(l)

plt.show()
p = sns.catplot(x = 'Attrition', hue = 'Gender', data = hr, kind = 'count')

plt.show()
sns.catplot(x = 'Attrition', y = 'HourlyRate', kind = 'swarm', data = hr)

sns.catplot(x = 'Attrition', y = 'HourlyRate', kind = 'boxen', data = hr)

plt.show()
p = sns.catplot(x = 'Attrition', hue = 'JobInvolvement', data = hr, kind = 'count', legend_out = True)



# editing the labels



new_labels = ['1 - Low', '2 - Medium', '3 - High', '4 - Very High']

for t, l in zip(p._legend.texts, new_labels): t.set_text(l)

plt.show()
p = sns.catplot(x = 'Attrition', hue = 'JobLevel', data = hr, kind = 'count')

plt.show()
p = sns.catplot(x = 'Attrition', hue = 'JobRole', data = hr, kind = 'count')

plt.show()
p = sns.catplot(x = 'Attrition', hue = 'JobSatisfaction', data = hr, kind = 'count', legend_out = True)



# editing the labels



new_labels = ['1 - Low', '2 - Medium', '3 - High', '4 - Very High']

for t, l in zip(p._legend.texts, new_labels): t.set_text(l)

plt.show()
p = sns.catplot(x = 'Attrition', hue = 'MaritalStatus', data = hr, kind = 'count')
sns.catplot(x = 'Attrition', y = 'MonthlyIncome', kind = 'swarm', data = hr)

sns.catplot(x = 'Attrition', y = 'MonthlyIncome', kind = 'boxen', data = hr)

plt.show()
sns.catplot(x = 'Attrition', y = 'MonthlyRate', kind = 'swarm', data = hr)

sns.catplot(x = 'Attrition', y = 'MonthlyRate', kind = 'boxen', data = hr)

plt.show()
p = sns.catplot(x = 'Attrition', hue = 'NumCompaniesWorked', data = hr, kind = 'count')
p = sns.catplot(x = 'Attrition', hue = 'OverTime', data = hr, kind = 'count')
p = sns.catplot(x = 'Attrition', hue = 'PercentSalaryHike', data = hr, kind = 'count')
p = sns.catplot(x = 'Attrition', hue = 'PerformanceRating', data = hr, kind = 'count', legend_out = True)



# editing the labels



new_labels = ['1 - Low', '2 - Good', '3 - Excellent', '4 - Outstanding']

for t, l in zip(p._legend.texts, new_labels): t.set_text(l)

plt.show()
p = sns.catplot(x = 'Attrition', hue = 'RelationshipSatisfaction', data = hr, kind = 'count', legend_out = True)



# editing the labels



new_labels = ['1 - Low', '2 - Medium', '3 - High', '4 - Very High']

for t, l in zip(p._legend.texts, new_labels): t.set_text(l)

plt.show()
sns.catplot(x = 'Attrition', hue = 'StockOptionLevel', data = hr, kind = 'count')

plt.show()
sns.catplot(x = 'Attrition', y = 'TotalWorkingYears', kind = 'swarm', data = hr)

sns.catplot(x = 'Attrition', y = 'TotalWorkingYears', kind = 'boxen', data = hr)

plt.show()
sns.catplot(x = 'Attrition', hue = 'TrainingTimesLastYear', data = hr, kind = 'count')

plt.show()
p = sns.catplot(x = 'Attrition', hue = 'WorkLifeBalance', data = hr, kind = 'count', legend_out = True)



# editing the labels



new_labels = ['1 - Bad', '2 - Good', '3 - Better', '4 - Best']

for t, l in zip(p._legend.texts, new_labels): t.set_text(l)

plt.show()
sns.catplot(x = 'Attrition', y = 'YearsAtCompany', kind = 'swarm', data = hr)

sns.catplot(x = 'Attrition', y = 'YearsAtCompany', kind = 'boxen', data = hr)

plt.show()
sns.catplot(x = 'Attrition', y = 'YearsInCurrentRole', kind = 'swarm', data = hr)

sns.catplot(x = 'Attrition', y = 'YearsInCurrentRole', kind = 'boxen', data = hr)

plt.show()
sns.catplot(x = 'Attrition', y = 'YearsSinceLastPromotion', kind = 'swarm', data = hr)

sns.catplot(x = 'Attrition', y = 'YearsSinceLastPromotion', kind = 'boxen', data = hr)

plt.show()
sns.catplot(x = 'Attrition', y = 'YearsWithCurrManager', kind = 'swarm', data = hr)

sns.catplot(x = 'Attrition', y = 'YearsWithCurrManager', kind = 'boxen', data = hr)

plt.show()