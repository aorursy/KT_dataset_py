import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from collections import Counter



from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
# Import data



df = pd.read_csv('../input/MFG10YearTerminationData.csv')
# look at the data



# Check size of the dataset

df.shape
# Data types

df.dtypes
# A prelimiary look at the individual columns in the training set

df.head()
df.tail()
# Summarise the numerical data

df.describe()
# Summarise the non-numerical data

df.describe(include=['O'])
# Create new categories for job titles



# Look at full list of job titles and frequency

df.job_title.value_counts()
# The 47 jobs can be separated according to corporate hierarchy

# Use employee, manager, and combined executives and board 

employee = ['Meat Cutter', 'Dairy Person', 'Produce Clerk', 'Baker', 'Cashier',

            'Shelf Stocker', 'Recruiter', 'HRIS Analyst', 'Accounting Clerk',

            'Benefits Admin', 'Labor Relations Analyst', 'Accounts Receiveable Clerk',

            'Accounts Payable Clerk', 'Auditor', 'Compensation Analyst',

            'Investment Analyst', 'Systems Analyst', 'Corporate Lawyer', 'Legal Counsel']



manager = ['Customer Service Manager', 'Processed Foods Manager', 'Meats Manager',

           'Bakery Manager', 'Produce Manager', 'Store Manager', 'Trainer', 'Dairy Manager']



executive = ['Exec Assistant, Finance', 'Exec Assistant, Legal Counsel',

             'CHief Information Officer', 'CEO', 'Exec Assistant, Human Resources',

             'Exec Assistant, VP Stores']



board = ['VP Stores', 'Director, Recruitment', 'VP Human Resources', 'VP Finance',

         'Director, Accounts Receivable', 'Director, Accounting',

         'Director, Employee Records', 'Director, Accounts Payable',

         'Director, HR Technology', 'Director, Investments',

         'Director, Labor Relations', 'Director, Audit', 'Director, Training',

         'Director, Compensation']



# Check all jobs were entered into the categories

total = len(employee) + len(manager) + len(executive) + len(board)

print('Total jobs categorised:', total, 'out of 47')
# Make a copy of job titles in a new column

df['Hierarchy'] = df.job_title



# Replace the job titles in Hierarchy

# The corporate hierarchy intrinsically has order from small to large, 

# so ordinal numbers may be used

df.Hierarchy = df.Hierarchy.replace(employee, 0)

df.Hierarchy = df.Hierarchy.replace(manager, 1)

df.Hierarchy = df.Hierarchy.replace(executive, 2)

df.Hierarchy = df.Hierarchy.replace(board, 3)



# Check that the replacement went to plan

df.Hierarchy.value_counts()
# Create new categories for department names



# Look at full list of departments and frequency

df.department_name.value_counts()
# The departments can be separated according to whether they serve the customer

# or the business

serve_cus = ['Meats', 'Dairy', 'Produce', 'Bakery', 'Customer Service', 'Processed Foods']



serve_biz = ['Store Management', 'Executive', 'Recruitment', 'HR Technology',

             'Accounting', 'Employee Records', 'Accounts Receiveable',

             'Accounts Payable', 'Labor Relations', 'Training', 'Compensation',

             'Audit', 'Investment', 'Information Technology', 'Legal']



# Check all departments were entered into the categories

total = len(serve_cus) + len(serve_biz)

print('Total departments categorised:', total, 'out of 21')
# Make a copy of department names in a new column

df['Service_to'] = df.department_name



# Replace the department names in Service_to

df.Service_to = df.Service_to.replace(serve_cus, 'Customer')

df.Service_to = df.Service_to.replace(serve_biz, 'Business')



# Check the replacement went to plan

df.Service_to.value_counts()
# Create new categories for city names



# Look at full list of cities and frequency

df.city_name.value_counts()
# The cities are in Canada.

# The cities can be separated according to population size.



# The population data for 2011 was obtained from Statistics Canada

# http://www12.statcan.gc.ca/census-recensement/2016/dp-pd/prof/index.cfm?Lang=E

# Used 2011 as it is the most recent before the last year of this dataset (2015)

city_pop_2011 = {'Vancouver':2313328,

                 'Victoria':344615,

                 'Nanaimo':146574,

                 'New Westminster':65976,

                 'Kelowna':179839,

                 'Burnaby':223218,

                 'Kamloops':85678,

                 'Prince George':71974,

                 'Cranbrook':19319,

                 'Surrey':468251,

                 'Richmond':190473,

                 'Terrace':11486,

                 'Chilliwack':77936,

                 'Trail':7681,

                 'Langley':25081,

                 'Vernon':38180,

                 'Squamish':17479,

                 'Quesnel':10007,

                 'Abbotsford':133497,

                 'North Vancouver':48196,

                 'Fort St John':18609,

                 'Williams Lake':10832,

                 'West Vancouver':42694,

                 'Port Coquitlam':55985,

                 'Aldergrove':12083,

                 'Fort Nelson':3561,

                 'Nelson':10230,

                 'New Westminister':65976,

                 'Grand Forks':3985,

                 'White Rock':19339,

                 'Haney':76052,

                 'Princeton':2724,

                 'Dawson Creek':11583,

                 'Bella Bella':1095,

                 'Ocean Falls':129,

                 'Pitt Meadows':17736,

                 'Cortes Island':1007,

                 'Valemount':1020,

                 'Dease Lake':58,

                 'Blue River':215}

# Population notes

# New Westminister is treated as a misspelling of New Westminster

# Used Haney in Maple Ridge, British Columbia because most of the other cities are in BC

# Used Bella Bella 1 (Indian reserve) for Bella Bella

# Used Central Coast A for Ocean Falls 

# Used Strathcona B for Cortes Island

# Used Dease Lake 9 (Indian reserve) for Dease Lake



# Check dictionary made correctly

print('Cities in dictionary:', len(city_pop_2011), 'out of 40')
# Make a copy of city names

df['Pop'] = df.city_name



# Map from city name to population

df.Pop = df.Pop.map(city_pop_2011)



# Make a new column for population category

df['Pop_category'] = df.Pop



# Categorise according to population size

# >= 100,000 is City

# 10,000 to 99,999 is Rural

# < 10,000 is Remote

# Guidance from Australian Institute of Health and Welfare

# http://www.aihw.gov.au/rural-health-rrma-classification/

city_ix = (df['Pop'] >= 100000)

rural_ix = ((df['Pop'] < 100000) & (df['Pop'] >= 10000))

remote_ix = (df['Pop'] < 10000)

df.loc[city_ix, 'Pop_category'] = 'City'

df.loc[rural_ix, 'Pop_category'] = 'Rural'

df.loc[remote_ix, 'Pop_category'] = 'Remote'



# Check the replacement went to plan

df.Pop_category.value_counts()
# As the category names are based on population size, the data could be represented

# by an ordinal category instead of a nominal category.

# Convert from nominal to ordinal 

df.Pop_category = df.Pop_category.replace('Remote', 0)

df.Pop_category = df.Pop_category.replace('Rural', 1)

df.Pop_category = df.Pop_category.replace('City', 2)



# Check the replacement went to plan

df.Pop_category.value_counts()
# Convert STATUS from string to numerical

df.STATUS = df.STATUS.map({'ACTIVE':1, 'TERMINATED':0})
# Data visualisation



# Separate data to avoid one excessively overlapping the other in plots.

# One set for the terminated (out of company) and working (in company)

out_of_co = df[df.STATUS == 0]

in_co = df[df.STATUS == 1]
# Start with a broad look at each group in terms of age and length of service

f, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)



ax1.scatter(out_of_co.age, out_of_co.length_of_service, color='r')

ax1.set_xlabel('Age')

ax1.set_ylabel('Length of service')

ax1.set_title('Out of company')



ax2.scatter(in_co.age, in_co.length_of_service, color='b')

ax2.set_xlabel('Age')

ax2.set_title('In company')
# Scatter plot of out of company dataset, with histograms of the axes

g = sns.jointplot(out_of_co.age, out_of_co.length_of_service, color='r')
# When someone leaves the company, look at their age, length of service, city size,

# and position in hierarchy. Separated by gender

g = sns.FacetGrid(out_of_co, col='Pop_category', row='Hierarchy', palette='Set1_r', 

                  hue='gender_short', margin_titles=True)

g = (g.map(plt.scatter, 'age', 'length_of_service').add_legend())
# Do the same for people who are working in the company

g = sns.FacetGrid(in_co, col='Pop_category', row='Hierarchy', palette='Set1_r', 

                  hue='gender_short', margin_titles=True)

g = (g.map(plt.scatter, 'age', 'length_of_service').add_legend())
# Out of interest, look at when termination is voluntary or involuntary

g = sns.FacetGrid(out_of_co, col='Pop_category', row='termreason_desc', palette='Set1_r', 

                  hue='termtype_desc', margin_titles=True)

g = (g.map(plt.scatter, 'age', 'length_of_service').add_legend())
# Out of interest, look at number of terminations per year.

# Count terminations per year

term_per_year = Counter(out_of_co.STATUS_YEAR)

term_per_year_df = pd.DataFrame.from_dict(term_per_year, orient='index')

term_per_year_df = term_per_year_df.sort_index()

term_per_year_df.plot(kind='bar')
# Preprocessing for machine learning models



# Drop the employee ID, record date, birth date, termination date, termination reason, 

# termination type, gender_full, STATUS_YEAR, and store_name features

# Also drop job_title (replaced with Hierarchy), department_name (replaced with Service_to),

# and city_name and Pop (replaced with Pop_category)

drop_cols = ['EmployeeID', 'recorddate_key', 'birthdate_key', 'orighiredate_key',

             'terminationdate_key', 'gender_full', 'termreason_desc',

             'termtype_desc', 'STATUS_YEAR', 'store_name', 'job_title', 'department_name',

             'city_name', 'Pop']



df = df.drop(drop_cols, axis=1)
# The gender, business unit and Service_to categories are nominal, so they will

# be exploded instead of being converted to ordinal values

dummy_cols = ['gender_short', 'BUSINESS_UNIT', 'Service_to']

df = pd.get_dummies(df, columns=dummy_cols)
# Separate the label from the dataset

label = df.STATUS

df = df.drop('STATUS', axis=1)
# Machine learning classification models



# Only age and length of service have double digit values. It should be okay to

# leave the values at their normal scales.



# Split data for training and testing. Specify random state for repeatability.

X_train, X_test, y_train, y_test = train_test_split(df, label, test_size=0.3,

                                                    random_state=10)
# KNN

model = KNeighborsClassifier(n_neighbors=5, weights='uniform')

model.fit(X_train, y_train)

score = model.score(X_test, y_test)

print('KNN model score is %0.4f' %score)
# SVC

model = SVC(C=1, kernel='rbf', random_state=10)

model.fit(X_train, y_train)

score = model.score(X_test, y_test)

print('SVC model score is %0.4f' %score)
# Random Forest

model = RandomForestClassifier(n_estimators=10, criterion='gini', min_samples_split=2,

                               oob_score=False, random_state=10)

model.fit(X_train, y_train)

score = model.score(X_test, y_test)

print('Random Forest model score is %0.4f' %score)