# Prepare SQL query
import os
import sqlite3
import pandas as pd
import numpy as np

path = "../input/"
# file_path = os.path.join(path,file_name)

try:
    # Change the current working Directory    
    os.chdir(path)
    print("Directory changed")
except OSError:
    print("Can't change the Current Working Directory")  
# This part originally prepares database and SQL query on local computers
# but is not executed in this Kaggle Notebook. We will deal with .csv files directly from now on.

# Skip the current cell and continue to the next step.
def skip():
    
    # Clear example.db if it exists

    if os.path.exists('example.db'):
        os.remove('example.db')

    # Create a database
    conn = sqlite3.connect('example.db')

    # Load some csv data
    visits = pd.read_csv('visits.csv')
    fitness_tests = pd.read_csv('fitness_tests.csv')
    applications = pd.read_csv('applications.csv')
    purchases = pd.read_csv('purchases.csv')

    # Add the data to our database
    visits.to_sql('visits', conn, dtype={
        'first_name':'VARCHAR(256)',
        'last_name':'VARCHAR(256)',
        'email':'VARCHAR(256)',
        'gender':'VARCHAR(256)',
        'visit_date': 'DATE'
    })
    fitness_tests.to_sql('fitness_tests', conn, dtype={
        'first_name':'VARCHAR(256)',
        'last_name':'VARCHAR(256)',
        'email':'VARCHAR(256)',
        'gender':'VARCHAR(256)',
        'fitness_test_date': 'DATE'
    })
    applications.to_sql('applications', conn, dtype={
        'first_name':'VARCHAR(256)',
        'last_name':'VARCHAR(256)',
        'email':'VARCHAR(256)',
        'gender':'VARCHAR(256)',
        'application_date': 'DATE'
    })
    purchases.to_sql('purchases', conn, dtype={
        'first_name':'VARCHAR(256)',
        'last_name':'VARCHAR(256)',
        'email':'VARCHAR(256)',
        'gender':'VARCHAR(256)',
        'purchases_date': 'DATE'
    })
    
    # Here's an example of a query that just displays some data
    sql_query('''
    SELECT *
    FROM visits
    LIMIT 5
    ''')

    # Here's an example where we save the data to a DataFrame
    df = sql_query('''
    SELECT *
    FROM applications
    LIMIT 5
    ''')

# Make a convenience function for running SQL queries
def sql_query(query):
    try:
        df = pd.read_sql(query, conn)
    except Exception as e: 
        print(e.message)
    return df


# Load some csv data
visits = pd.read_csv('visits.csv')
fitness_tests = pd.read_csv('fitness_tests.csv')
applications = pd.read_csv('applications.csv')
purchases = pd.read_csv('purchases.csv')
# Examine visits here
visits.head(5)
# Examine fitness_tests here
fitness_tests.head(5)
# Examine applications here
applications.head(5)
# Examine purchases here
purchases.head(5)
print(visits.info()) #see if visit_time column is datetime object
print(type(visits.visit_date[0])) #original is str, need to change to datetime
from datetime import datetime
visits["VisitDate"] =  pd.to_datetime(visits["visit_date"], format="%m-%d-%y")
visits.head(5)
visits1 = visits[visits.VisitDate >= '2017-07-01']
visits1.head(5)
len(fitness_tests)
merge_on = ['first_name', 'last_name', 'gender', 'email']
#if you don't need to join the index, you can use merge
#don't use join here. will have error
merged = visits1.merge(fitness_tests, on=merge_on, how='left').merge(applications, on=merge_on, how='left').merge(purchases, on=merge_on, how='left')
print(len(merged))
merged.head(5)
import pandas as pd
from matplotlib import pyplot as plt
df = merged
df['ab_test_group'] = df.fitness_test_date.apply(lambda x: 'B' if pd.isnull(x) else 'A')
df.head(5)
ab_counts = df.groupby('ab_test_group').count()
ab_counts
plt.figure(figsize=(12,7))
plt.axis('equal')
plt.pie(ab_counts.email, autopct='%1.2f%%')
plt.legend(['A','B'])
#plt.savefig('ab_test_pie_chart') #doesn't work in this notebook
df['is_application'] = df.application_date.apply(lambda x: 'No Application' if pd.isnull(x) else 'Application')
df.head(5)
app_counts = df.groupby(['ab_test_group', 'is_application']).count().reset_index()
app_counts
app_pivot = app_counts.pivot(index='ab_test_group', columns='is_application',values='email').reset_index()
app_pivot
app_pivot['Total'] = app_pivot['Application'] + app_pivot['No Application']
app_pivot
app_pivot['Percent with Application'] = app_pivot['Application'] / app_pivot['Total']
app_pivot
from scipy.stats import chi2_contingency

# Contingency table
#         application      |  no application
# ----+------------------+------------
# A |    ct11    |  ct12
# B |    ct21     |  ct22


X = [[250, 2254], 
    [325, 2175]]
chi2, pval, dof, expected = chi2_contingency(X)

print("chi2 test statistic is {0:.10f}".format(chi2))
print("pval is {0:.10f}".format(pval))
print("dof is {}".format(dof))
print("expected is {}".format(expected))

print("We get pval = {0:.3f} < 0.05. In this case, the null hypothesis is that there’s no significant difference between the data in A and B. \
We reject that hypothesis, which means people that appeared in fit test are more likely to proceed with an application.".format(pval))
df['is_member'] = df.purchase_date.apply(lambda x: 'Not Member' if pd.isnull(x) else 'Member')
df.head(5)
just_apps = df[df.is_application == 'Application']
just_apps.head(5)
member_pivot = just_apps.groupby(['is_member','ab_test_group']).count().reset_index().pivot(index='ab_test_group', columns='is_member',values='email').reset_index()
member_pivot['Total'] = member_pivot['Member'] + member_pivot['Not Member']
member_pivot['Percent Purchase'] = member_pivot['Member'] / member_pivot['Total']

member_pivot
from scipy.stats import chi2_contingency

# Contingency table
#         purchase      |  no purchase
# ----+------------------+------------
# A |    ct11    |  ct12
# B |    ct21     |  ct22


X = [[200, 50], 
     [250, 75]]
chi2, pval, dof, expected = chi2_contingency(X)

print("chi2 test statistic is {0:.10f}".format(chi2))
print("pval is {0:.10f}".format(pval))
print("dof is {}".format(dof))
print("expected is {}".format(expected))

print("We get pval = {0:.3f} > 0.05. In this case, the null hypothesis is that there’s no significant difference between the data in A and B. \
We accept that hypothesis, which means among people who picked up applications, people that previously appeared in fit test are not more likely to proceed with purchasing membership.".format(pval))
final_member_pivot = df.groupby(['is_member','ab_test_group']).count().reset_index().pivot(index='ab_test_group', columns='is_member',values='email').reset_index()

final_member_pivot['Total'] = final_member_pivot['Member'] + final_member_pivot['Not Member']
final_member_pivot['Percent Purchase'] =final_member_pivot['Member'] / final_member_pivot['Total']
final_member_pivot
X = [[200, 2304], 
     [250, 2250]]
chi2, pval, dof, expected = chi2_contingency(X)

print("chi2 test statistic is {0:.10f}".format(chi2))
print("pval is {0:.10f}".format(pval))
print("dof is {}".format(dof))
print("expected is {}".format(expected))

print("We get pval = {0:.3f} < 0.05. In this case, the null hypothesis is that there’s no significant difference between the data in A and B. \
We reject that hypothesis, which means among all people who visit MuscleHub, people that previously appeared in fit test are more likely to proceed with purchasing membership.".format(pval))
fig = plt.figure(figsize=(12,7))
ax = fig.add_subplot(1, 1, 1)
height = np.array(app_pivot['Percent with Application'])
plt.bar(x=np.array(range(len(app_pivot))), height=height)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Fitness Test','No Fitness Test'])
ax.set_yticks([0, 0.05, 0.10, 0.15])
ax.set_yticklabels(['0%', '5%', '10%', '15%'])
plt.title("Percent of visitors who submitted application")
fig = plt.figure(figsize=(12,7))
ax = fig.add_subplot(1, 1, 1)
height = np.array(member_pivot['Percent Purchase'])
plt.bar(x=np.array(range(len(member_pivot))), height=height)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Fitness Test','No Fitness Test'])
ax.set_yticks([0, 0.25, 0.75, 1])
ax.set_yticklabels(['0%', '25%', '75%', '100%'])
plt.title("Percent of applicants who purchase membership ")
fig = plt.figure(figsize=(12,7))
ax = fig.add_subplot(1, 1, 1)
height = np.array(final_member_pivot['Percent Purchase'])
plt.bar(x=np.array(range(len(final_member_pivot))), height=height)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Fitness Test','No Fitness Test'])
ax.set_yticks([0, 0.05, 0.10, 0.15])
ax.set_yticklabels(['0%', '5%', '10%', '15%'])
plt.title("Percent of visitors who purchase membership ")