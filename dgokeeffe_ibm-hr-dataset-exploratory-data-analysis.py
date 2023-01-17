# REQUIRES PYTHON 3.6

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import re

from scipy import stats
from functools import reduce

# Some matplotlib options
%matplotlib inline
matplotlib.style.use("ggplot")

# General pandas options
pd.set_option('display.max_colwidth', -1)  # Show the entire column 
pd.options.display.max_columns = 100 
pd.options.display.max_rows = 10000 

# Seaborn options
sns.set_style("whitegrid")
# Load the data in
hr = pd.read_csv("../input/ibm-hr-wmore-rows/IBM HR Data.csv")

# Lets see what it looks like
print(hr.shape)
before_dedup = hr.shape[0]
hr.describe(include='all')
hr.head()
# Check for missings
print(np.count_nonzero(hr.isnull().values))
print(hr.isnull().any())

# Check for duplicates
print(hr[hr.duplicated(keep=False)].shape)

# Strip whitespaces
hr = hr.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Check for conflicting types
hr.dtypes
# Hate to throw away data but it's only 353 values out of over 23 thousand
hr.dropna(axis=0, inplace=True)

# Get rid of all the duplicates
hr.drop_duplicates(inplace=True)

# Lets see what it looks like now
print("Duplicates Removed: " + str(before_dedup - hr.shape[0]))
hr.describe()
hr['JobSatisfaction'].value_counts()
hr['JobSatisfaction'].unique()
# Half the rows in JobSatisfaction seem to be strings. 
# It's the same for the other columns. Let's cast them to floats.
cols = ['JobSatisfaction', 'HourlyRate', 'MonthlyIncome', 'PercentSalaryHike']
hr[cols] = hr[cols].applymap(np.float64)
# I know from looking in Excel that certain fields are useless so lets get rid of them
hr = hr.drop(['EmployeeCount', 'Over18', "StandardHours", "EmployeeNumber"], 1)
# Lets try find some more funky rows
for col in hr:
    print(col)
    print(hr[col].unique())
hr.to_csv("hr-clean.csv")
# Subset the dataset into all the numerical values
numeric_hr = hr.select_dtypes(include=[np.number])

# Compete the correlation matrix
corr = numeric_hr._get_numeric_data().corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
heatmap = sns.heatmap(corr, mask=mask, cmap=cmap, center=0.0,
                      vmax = 1, square=True, linewidths=.5, ax=ax)
plt.savefig('corr-heat.png')
plt.show()
# Lets drop the rates from the numerics dataframe
numeric_hr = numeric_hr.drop(["HourlyRate","DailyRate", "MonthlyRate"], 1)
hr.describe()
print(hr.Attrition.value_counts())

# Easier to join all leaver together for my analyses considering there are very few terminations
hr['Attrition'] = hr['Attrition'].replace("Termination", "Voluntary Resignation")
hr['Attrition'] = hr['Attrition'].replace("Voluntary Resignation", "Former Employees")
hr['Attrition'] = hr['Attrition'].replace("Current employee", "Current Employees")

plt.figure(figsize=(12,8))
plt.title('Number of Former/Current Employees at TechCo')
sns.countplot(x="Attrition", data=hr)
hr['Attrition'].value_counts()/hr['Attrition'].count()*100
temp3 = pd.crosstab([hr.Department,hr.Gender,hr.MaritalStatus,hr.WorkLifeBalance], hr['Attrition'])
print(temp3)
income_pivot = hr.pivot_table(values=["MonthlyIncome"], index=["Gender","MaritalStatus","WorkLifeBalance"], aggfunc=[np.mean, np.std])
print(income_pivot)
# Plot the distribution of age by Attrition Factor
plt.figure(figsize=(12,8))
plt.title('Age distribution of Employees at Telco by Attrition')
sns.distplot(hr.Age[hr.Attrition == 'Former Employees'], bins = np.linspace(1,70,35))
sns.distplot(hr.Age[hr.Attrition == 'Current Employees'], bins = np.linspace(1,70,35))
#sns.distplot(hr.Age[hr.Attrition == 'Termination'], bins = np.linspace(0,70,35))
plt.legend(['Former Emploees','Current Employees'])
# Plot the distribution of Years at Company by Attrition
plt.figure(figsize=(12,8))
plt.title('Distribution of the Number of Years Employees Spend at Telco by Attrition')
#sns.distplot(hr.YearsAtCompany, bins = np.linspace(0,40,40))
sns.distplot(hr.YearsAtCompany[hr.Attrition == 'Former Employees'], bins = np.linspace(0,40,40))
sns.distplot(hr.YearsAtCompany[hr.Attrition == 'Current Employees'], bins = np.linspace(0,40,40))
plt.legend(['Former Emploees','Current Employees'])
# Plot out the counts of OverTime
sns.factorplot("Attrition", col="OverTime", data=hr, kind="count", col_wrap=2, size=5)
plt.subplots_adjust(top=.85)
plt.suptitle('Attrition Counts by whether an Employee worked Over Time')

# Chi squared test of independence
# H0: Overtime and Attrition are independent of each other
res_1 = hr.OverTime[hr.Attrition == 'Current Employees'].value_counts()
res_2 = hr.OverTime[hr.Attrition == 'Former Employees'].value_counts()
obs = np.array([res_1, res_2])
stats.chi2_contingency(obs)
# Plot the distribution of Years at Company by Attrition
plt.figure(figsize=(12,8))
plt.title('Age Distribution of Employees who have worked Over Time')
#sns.distplot(hr.YearsAtCompany, bins = np.linspace(0,40,40))
sns.distplot(hr.Age[hr.OverTime == 'Yes'], bins = np.linspace(0,70,35))
plt.figure(figsize=(12,8))
sns.countplot(x="Gender", data=hr)
# Proportion of males
plt.title('Frequency of Gender at TechCo')
hr['Gender'].value_counts().Male/hr['Gender'].count()*100
# First lets cast these string columns into categories
cats = ['Attrition', 'BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']
for col in cats:
    hr[col] = hr[col].astype('category')

group_hr = hr.groupby(cats)

# Plot the distribution of females in this workplace
plt.figure(figsize=(12,8))
#sns.countplot(x="Gender", hue="Attrition", data=hr[hr['Attrition'].isin(['Voluntary Resignation', 'Termination'])])

attrition_counts = (hr.groupby(['Gender'])['Attrition']
                     .value_counts(normalize=True)
                     .rename('percentage')
                     .mul(100)
                     .reset_index()
                     .sort_values('Attrition'))
plt.title('Percent Distribution of Gender by Attrition at TechCo')
sns.barplot(x="Gender", y="percentage", hue="Attrition", data=attrition_counts)

# It's the same, looks suss
print(attrition_counts)

# Nope my code is alright
# Plot the distribution of females in this workplace
plt.figure(figsize=(12,8))
#sns.countplot(x="Gender", hue="Attrition", data=hr[hr['Attrition'].isin(['Voluntary Resignation', 'Termination'])])

attrition_counts = (hr.groupby(['Gender'])['BusinessTravel']
                     .value_counts(normalize=True)
                     .rename('percentage')
                     .mul(100)
                     .reset_index()
                     .sort_values('BusinessTravel'))
plt.title('Percent Distribution of Gender by Business Travel Frequency')
sns.barplot(x="Gender", y="percentage", hue="BusinessTravel", data=attrition_counts) 

#sns.countplot(x="Gender", data=hr, palette="Greens_d")
# Plot the distribution of females in this workplace
plt.figure(figsize=(12,8))
#sns.countplot(x="Gender", hue="Attrition", data=hr[hr['Attrition'].isin(['Voluntary Resignation', 'Termination'])])

attrition_counts = (hr.groupby(['Gender'])['Department']
                     .value_counts(normalize=True)
                     .rename('percentage')
                     .mul(100)
                     .reset_index()
                     .sort_values('Department'))
plt.title('Distribution of Gender across Departments')
sns.barplot(x="Gender", y="percentage", hue="Department", data=attrition_counts) 
# Plot the distribution of age by gender
plt.figure(figsize=(12,8))
sns.distplot(hr.Age[hr.Gender == 'Male'], bins = np.linspace(0,70,35))
sns.distplot(hr.Age[hr.Gender == 'Female'], bins = np.linspace(0,70,35))
plt.title('Distribution of Age by Gender')
plt.legend(['Males','Females'])
# It appears women are under-represented at this company. Let's see if they get paid less.
plt.figure(figsize=(15,10))
plt.title('Average Monthly Income by Gender')
sns.barplot(x="Gender", y="MonthlyIncome", data=hr)

# T-Test
cat1 = hr[hr['Gender']=='Male']
cat2 = hr[hr['Gender']=='Female']

print(stats.ttest_ind(cat1['MonthlyIncome'], cat2['MonthlyIncome']))
plt.figure(figsize=(15,10))
plt.title('Distribution of Monthly Income by Gender')
sns.distplot(hr.MonthlyIncome[hr.Gender == 'Male'], bins = np.linspace(0,20000,20))
sns.distplot(hr.MonthlyIncome[hr.Gender == 'Female'], bins = np.linspace(0,20000,20))
plt.legend(['Males','Females'])
# What about all the single ladies?
plt.figure(figsize=(15,10))
plt.title('Average Monthly Income by Gender and Maritial Status')
sns.barplot(x="MaritalStatus", y="MonthlyIncome", hue="Gender", data=hr)
# Age by Gender and Martial Status 
plt.figure(figsize=(15,15))
plt.title('Average Monthly Income by Gender and Maritial Status')
sns.boxplot(x="MaritalStatus", y="Age", hue="Gender", data=hr, width=.5) 
# Trying to get a binned distribution in of Age by MonthlyIncome in Seaborn
plt.figure(figsize=(15,15))
bins=[18, 25, 35, 50, 70]
out = hr.groupby(pd.cut(hr['Age'], bins=bins, include_lowest=True)).aggregate(np.mean)
print(out.head())
#out[['Age']] = out[['Age']].applymap(str)
out['Age Bracket'] = ['18-25', '26-35', '36-50', '51-70']

# Fixed X-axis labels currently looking awful!
plt.title('Average Monthly Income by Age Bracket')
sns.barplot('Age Bracket', 'MonthlyIncome', data=out, palette="muted")
out.head()
# Trying to get a binned distribution in of Age by MonthlyIncome in Seaborn
plt.figure(figsize=(15,15))
bins=[0, 10, 20, 30, 40]
out = hr.groupby(pd.cut(hr['YearsAtCompany'], bins=bins, include_lowest=True)).aggregate(np.mean)
out[['YearsAtCompany']] = out[['YearsAtCompany']].applymap(str)
out['Years at Company Bracket'] = ['0-10', '11-20', '21-30', '31-40']

# Fixed X-axis labels currently looking awful!
plt.title('Average Monthly Income by Years Worked at TechCo')
sns.barplot('Years at Company Bracket', 'MonthlyIncome', data=out, palette="muted")
out.head()
plt.figure(figsize=(15,15))
sns.lmplot("YearsAtCompany", "MonthlyIncome", data=hr, size=10) 
hr["Attrition"].value_counts() # Large class imbalance
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn_pandas import DataFrameMapper, gen_features, cross_val_score

# Encode the categorical variables so that scikit-learn can read them
cat_cols = ['Attrition', 'BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime', 'Employee Source']
feature_def = gen_features(
    columns= cat_cols,
    classes=[LabelEncoder]
)
mapper = DataFrameMapper(feature_def)
hr[cat_cols] = mapper.fit_transform(hr)
hr.head()
# Build a forest to predict attrition and compute the feature importances
rf = RandomForestClassifier(class_weight="balanced", n_estimators=500) 
rf.fit(hr.drop(['Attrition'],axis=1), hr.Attrition)
importances = rf.feature_importances_
names = hr.columns
importances, names = zip(*sorted(zip(importances, names)))

# Lets plot this
plt.figure(figsize=(12,8))
plt.barh(range(len(names)), importances, align = 'center')
plt.yticks(range(len(names)), names)
plt.xlabel('Importance of features')
plt.ylabel('Features')
plt.title('Importance of each feature')
plt.show()
# Make predictions using 10-K-Fold-CV

# Baseline:
print((hr.Attrition.value_counts()/(hr.shape[0]))*100)

# Accuracy
scores = cross_val_score(rf, hr.drop(['Attrition'],axis=1), hr.Attrition, cv=10, scoring='accuracy')
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# ROC
scores = cross_val_score(rf, hr.drop(['Attrition'],axis=1), hr.Attrition, cv=10, scoring='roc_auc')
print(scores)
print("ROC_AUC: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
from sklearn.decomposition import PCA

# Normalise PCA as it can have a large effect on the result then fit
std_clf = make_pipeline(StandardScaler(), PCA())
std_clf.fit(hr.drop(['Attrition'], axis=1))
existing_2d = std_clf.transform(hr.drop(['Attrition'],axis=1))
# Print out the ratio of explained variance for each principal component
pca_std = std_clf.named_steps['pca']
print(pca_std.explained_variance_ratio_.cumsum())
# Convert result to dataframe, add the labels
existing_hr_2d = pd.DataFrame(existing_2d)
existing_hr_2d = pd.concat([existing_hr_2d, hr[['Attrition']]], axis = 1)
existing_hr_2d.columns = ['PC' + str(i) for i in range(1, existing_hr_2d.shape[1])] + ['Attrition']
di = {0.0: "Current Employee", 1.0: "Former Employee"}
existing_hr_2d = existing_hr_2d.replace({"Attrition":di})
#ax = existing_hr_2d.plot(kind='scatter', x='PC1', y='PC2', figsize=(16,8))

# Plot with Seaborn
plt.figure(figsize=(16,8))
sns.lmplot("PC1", "PC2", data=existing_hr_2d, hue='Attrition', fit_reg=False, size=15)