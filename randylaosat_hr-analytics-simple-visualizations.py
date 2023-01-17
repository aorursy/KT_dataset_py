# Import the neccessary modules for data manipulation and visual representation

%matplotlib inline

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns
#Read the analytics csv file and store our dataset into a dataframe called "df"

df = pd.DataFrame.from_csv('../input/HR_comma_sep.csv', index_col=None)
# Check to see if there are any missing values in our data set

df.isnull().any()
# Get a quick overview of what we are dealing with in our dataset

df.head()
# Renaming certain columns for better readability

df = df.rename(columns={'satisfaction_level': 'satisfaction', 

                        'last_evaluation': 'evaluation',

                        'number_project': 'projectCount',

                        'average_montly_hours': 'averageMonthlyHours',

                        'time_spend_company': 'yearsAtCompany',

                        'Work_accident': 'workAccident',

                        'promotion_last_5years': 'promotion',

                        'sales' : 'department',

                        'left' : 'turnover'

                        })

# Convert "department" and "salary" features to numeric types because some functions won't be able to work with string types

df['department'].replace(['sales', 'accounting', 'hr', 'technical', 'support', 'management',

        'IT', 'product_mng', 'marketing', 'RandD'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], inplace = True)

df['salary'].replace(['low', 'medium', 'high'], [0, 1, 2], inplace = True)



df.head()
# Move the reponse variable "turnover" to the front of the table

front = df['turnover']

df.drop(labels=['turnover'], axis=1,inplace = True)

df.insert(0, 'turnover', front)

df.head()
# The dataset contains 10 columns and 14999 observations

df.shape
# Check the type of our features. 

df.dtypes
# Looks like about 76% of employees stayed and 24% of employees left. 

# NOTE: When performing cross validation, its important to maintain this turnover ratio

turnover_rate = df.turnover.value_counts() / 14999

turnover_rate
# Overview of summary

# On average, employees who left had a lower satisfaction level of -20%**, worked 8hours more per month, 

# had lower salary, and had a lower promotion rate

turnover_Summary = df.groupby('turnover')

turnover_Summary.mean()
# Display the statistical overview of the employees

df.describe()
#Correlation Matrix

corr = df.corr()

corr = (corr)

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)

sns.plt.title('Heatmap of Correlation Matrix')

corr
#Department   V.S.   Turnover

clarity_color_table = pd.crosstab(index=df["department"], 

                          columns=df["turnover"])



clarity_color_table.plot(kind="bar", 

                 figsize=(5,5),

                 stacked=True)
#Department   V.S.   Salary

clarity_color_table = pd.crosstab(index=df["department"], 

                          columns=df["salary"])



clarity_color_table.plot(kind="bar", 

                 figsize=(5,5),

                 stacked=True)
#Salary   V.S.   Turnover

clarity_color_table = pd.crosstab(index=df["salary"], 

                          columns=df["turnover"])



clarity_color_table.plot(kind="bar", 

                 figsize=(5,5),

                 stacked=True)
#Promotion   V.S.   Turnover

clarity_color_table = pd.crosstab(index=df["promotion"], 

                          columns=df["turnover"])



clarity_color_table.plot(kind="bar", 

                 figsize=(5,5),

                 stacked=True)
#YearsAtCompany   V.S.   Turnover

clarity_color_table = pd.crosstab(index=df["yearsAtCompany"], 

                          columns=df["turnover"])



clarity_color_table.plot(kind="bar", 

                 figsize=(5,5),

                 stacked=True)
#Frequency table of Turnover VS YearsAtCompany

df.loc[(df['turnover'] == 1),'yearsAtCompany'].plot(kind='hist', normed=1, bins=15, stacked=False, alpha=1)
#projectCount V.S. turnover

clarity_color_table = pd.crosstab(index=df["projectCount"], 

                          columns=df["turnover"])



clarity_color_table.plot(kind="bar", 

                 figsize=(5,5),

                 stacked=True)
# Here is a histogram that shows the frequency of people who left the company and the amount of projects they had.

# It looks like a majority of the employes who left only had two projects

# Also, you can see an increase in turnover as the number of projects increase. 

# 3 Projects seems to be the sweet spot.

df.loc[(df['turnover'] == 1),'projectCount'].plot(kind='hist', normed=1, bins=15, stacked=False, alpha=1)
fig = plt.figure(figsize=(10,4),)

ax=sns.kdeplot(df.loc[(df['turnover'] == 0),'evaluation'] , color='b',shade=True,label='no turnover')

ax=sns.kdeplot(df.loc[(df['turnover'] == 1),'evaluation'] , color='r',shade=True, label='turnover')

plt.title('Last evaluation')
#KDEPlot: Kernel Density Estimate Plot

fig = plt.figure(figsize=(10,4))

ax=sns.kdeplot(df.loc[(df['turnover'] == 0),'averageMonthlyHours'] , color='b',shade=True, label='no turnover')

ax=sns.kdeplot(df.loc[(df['turnover'] == 1),'averageMonthlyHours'] , color='r',shade=True, label='turnover')

plt.title('Average monthly hours worked')
#ProjectCount VS AverageMonthlyHours [BOXPLOT]

#Looks like the average employees who stayed worked about 200hours/month. Those that had a turnover worked about 250hours/month and 150hours/month



import seaborn as sns

sns.boxplot(x="projectCount", y="averageMonthlyHours", hue="turnover", data=df)
#ProjectCount VS Evaluation

#Looks like employees who did not leave the company had an average evaluation of around 70% even with different projectCounts

#There is a huge skew in employees who had a turnover though. It drastically changes after 3 projectCounts. 

#Employees that had two projects and a horrible evaluation left. Employees with more than 3 projects and super high evaluations left

import seaborn as sns

sns.boxplot(x="projectCount", y="evaluation", hue="turnover", data=df)
from pandas.tools.plotting import scatter_matrix

import matplotlib.pyplot as plt



# SCATTER MATRIX FOR EMPLOYEES WHO TURNOVERED

fig, ax = plt.subplots(figsize=(10,10))

turnoverDF = df[df['turnover']==1]

scatter_matrix(turnoverDF[['satisfaction','projectCount','evaluation']], alpha=0.2, diagonal='hist', ax=ax)
#Train-Test split

from sklearn.model_selection import train_test_split

label = df.pop('turnover')

data_train, data_test, label_train, label_test = train_test_split(df, label, test_size = 0.2, random_state = 15)
#Logistic Regression

from sklearn.linear_model import LogisticRegression

lg = LogisticRegression()

lg.fit(data_train, label_train)

lg_score_train = lg.score(data_train, label_train)

print("Training score: ",lg_score_train)

lg_score_test = lg.score(data_test, label_test)

print("Testing score: ",lg_score_test)
#SVM

from sklearn.svm import SVC

svm = SVC()

svm.fit(data_train, label_train)

svm_score_train = svm.score(data_train, label_train)

print("Training score: ",svm_score_train)

svm_score_test = svm.score(data_test, label_test)

print("Testing score: ",svm_score_test)
#kNN

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

knn.fit(data_train, label_train)

knn_score_train = knn.score(data_train, label_train)

print("Training score: ",knn_score_train)

knn_score_test = knn.score(data_test, label_test)

print("Testing score: ",knn_score_test)
#random forest

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()

rfc.fit(data_train, label_train)

rfc_score_train = rfc.score(data_train, label_train)

print("Training score: ",rfc_score_train)

rfc_score_test = rfc.score(data_test, label_test)

print("Testing score: ",rfc_score_test)
import pandas_profiling