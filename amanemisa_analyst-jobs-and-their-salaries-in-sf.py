import pandas as pd

import numpy as np

import sqlite3

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
con = sqlite3.connect('../input/database.sqlite')
JobTitle_with_data_summery = pd.read_sql_query('''

        select COUNT(*) AS Nums, AVG(TotalPay) AS TotalPay_Avg, 

        AVG(Benefits) AS Benefits_Avg, AVG(TotalPayBenefits) AS TotalPayBenefits_Avg

        from 

        Salaries

        WHERE JobTitle LIKE '%data%'

        ''',con)

JobTitle_with_data_summery
JobTitle_with_data = pd.read_sql_query('''

        SELECT JobTitle, COUNT(*) AS Nums

        FROM

        (select *

        from 

        Salaries

        WHERE JobTitle LIKE '%data%') AS a

        GROUP BY JobTitle

        ''',con)

JobTitle_with_data
JobTitle_with_analyst_summery = pd.read_sql_query('''

        select COUNT(*) AS Nums, AVG(TotalPay) AS TotalPay_Avg, 

        AVG(Benefits) AS Benefits_Avg, AVG(TotalPayBenefits) AS TotalPayBenefits_Avg,

        MIN(TotalPay) AS TotalPay_Min, 

        MIN(Benefits) AS Benefits_Min, MIN(TotalPayBenefits) AS TotalPayBenefits_Min

        from 

        Salaries

        WHERE JobTitle LIKE '%analyst%'

        ''',con)

JobTitle_with_analyst_summery
JobTitle_with_analyst = pd.read_sql_query('''

        SELECT upper(JobTitle) AS JobTitle, 

        COUNT(*) AS Nums

        FROM

        (select *

        from 

        Salaries

        WHERE JobTitle LIKE '%analyst%' AND TotalPay>0) AS analyst_table

        GROUP BY upper(JobTitle)

        ''',con)

JobTitle_with_analyst
Job_Salaries = pd.read_sql_query('''

        SELECT CASE

        WHEN upper(JobTitle) LIKE '%ADMIN%' THEN 'ADMINISTRATIVE ANALYST'

        WHEN upper(JobTitle) LIKE '%RETIREMENT%' THEN 'RETIREMENT ANALYST'

        WHEN upper(JobTitle) LIKE '%HUMAN RESOURCE%' THEN 'HUMAN RESOURCES ANALYST'

        WHEN upper(JobTitle) LIKE '%BENEFIT%' THEN 'BENEFITS ANALYST'

        WHEN (upper(JobTitle) LIKE '%COMP APP%') OR (upper(JobTitle) LIKE '%COMPUTER APPLICATION%') THEN 'COMPUTER APPLICATIONS ANALYST'

        WHEN upper(JobTitle) LIKE '%FEASIBILITY%' THEN 'FEASIBILITY ANALYST'

        WHEN upper(JobTitle) LIKE '%HEALTH%' THEN 'HEALTH CARE ANALYST'

        WHEN upper(JobTitle) LIKE '%MEDICAL%' THEN 'MEDICAL STAFF SERVICES DEPARTMENT ANALYST'

        WHEN upper(JobTitle) LIKE '%BUSINESS%' THEN 'BUSINESS ANALYST'

        WHEN upper(JobTitle) LIKE '%OPERATOR%' THEN 'OPERATOR ANALYST'

        WHEN (upper(JobTitle) LIKE '%PROGRAMMER%') OR (upper(JobTitle) LIKE '%PRG ANALYST%') THEN 'PROGRAMMER ANALYST'

        WHEN upper(JobTitle) LIKE '%PERF%' THEN 'PERFORMANCE ANALYST'

        WHEN (upper(JobTitle) LIKE '%MANAGEMENT%') OR (upper(JobTitle) LIKE '%MGMT%') THEN 'MANAGEMENT ANALYST'

        WHEN upper(JobTitle) LIKE '%PERSONNEL%' THEN 'PERSONNEL ANALYST'

        WHEN (upper(JobTitle) LIKE '%PROGRAM ANALYST%') OR (upper(JobTitle) LIKE '%PROGRAM SUPPORT%') THEN 'PROGRAM ANALYST'

        WHEN upper(JobTitle) LIKE '%SAFETY%' THEN 'SAFETY ANALYST'

        WHEN upper(JobTitle) LIKE '%SECURITY%' THEN 'SECURITY ANALYST'

        WHEN upper(JobTitle) LIKE '%UTILITY%' THEN 'UTILITY ANALYST'

        WHEN upper(JobTitle) LIKE '%WATER%' THEN 'WATER OPERATIONS ANALYST'

        END AS Job, COUNT(*) AS Nums, AVG(TotalPay) AS TotalPay_Avg, 

        AVG(Benefits) AS Benefits_Avg, AVG(TotalPayBenefits) AS TotalPayBenefits_Avg

        FROM

        (select *

        from 

        Salaries

        WHERE JobTitle LIKE '%analyst%' AND TotalPay>0) AS analyst_table

        GROUP BY Job

        ORDER BY Nums DESC

        ''',con)

Job_Salaries
Hot_Job = Job_Salaries['Job'][0:10]

Hot_Job.values
Hot_Job_Salaries = pd.read_sql_query('''

        SELECT JobTitle, 

        CASE

        WHEN upper(JobTitle) LIKE '%ADMIN%' THEN 'ADMINISTRATIVE ANALYST'

        WHEN upper(JobTitle) LIKE '%BUSINESS%' THEN 'BUSINESS ANALYST'

        WHEN upper(JobTitle) LIKE '%PERSONNEL%' THEN 'PERSONNEL ANALYST' 

        WHEN (upper(JobTitle) LIKE '%PROGRAMMER%') OR (upper(JobTitle) LIKE '%PRG ANALYST%') THEN 'PROGRAMMER ANALYST'

        WHEN upper(JobTitle) LIKE '%RETIREMENT%' THEN 'RETIREMENT ANALYST'

        WHEN upper(JobTitle) LIKE '%BENEFIT%' THEN 'BENEFITS ANALYST'

        WHEN (upper(JobTitle) LIKE '%PROGRAM ANALYST%') OR (upper(JobTitle) LIKE '%PROGRAM SUPPORT%') THEN 'PROGRAM ANALYST'

        WHEN upper(JobTitle) LIKE '%UTILITY%' THEN 'UTILITY ANALYST'

        WHEN upper(JobTitle) LIKE '%PERF%' THEN 'PERFORMANCE ANALYST'

        WHEN upper(JobTitle) LIKE '%HEALTH%' THEN 'HEALTH CARE ANALYST'

        END AS Job, 

        TotalPay, Benefits, TotalPayBenefits, Year

        FROM

        (select *

        from 

        Salaries

        WHERE JobTitle LIKE '%analyst%') AS analyst_table

        

        ''',con)
Hot_Job_Salaries.describe()
fig = plt.figure(figsize=(8,6))

ax = fig.add_subplot(1,1,1)

# I want to plot a boxplot. On x-axis I put the jobgroups with corresponding employee numbers, 

# and I want to order them by employee numbers (from biggest to smallest).

order = Hot_Job.values

sns.boxplot(x = 'Job', y = 'TotalPay', data = Hot_Job_Salaries,ax=ax,order = order)

Hot_Job_Number = Job_Salaries['Job'][0:10]+' '+Job_Salaries['Nums'][0:10].map(str)

ax.set_xticklabels(Hot_Job_Number.values)

plt.xticks(size = 10, rotation = 80)

ax.set_xlabel('Jobs with corresponding Employee numbers')

fig.suptitle('TotalPays of ten hot analyst jobs in SF')
Hot_Job_Salaries.dtypes
Hot_Job_Salaries["Benefits"]=Hot_Job_Salaries["Benefits"].apply(pd.to_numeric)
fig = plt.figure(figsize=(8,6))

ax = fig.add_subplot(1,1,1)

sns.boxplot(x = 'Job', y = 'Benefits', data = Hot_Job_Salaries,ax=ax,order = order)

ax.set_xticklabels(Hot_Job_Number.values)

plt.xticks(size = 10, rotation = 80)

ax.set_xlabel('Jobs with corresponding Employee numbers')

fig.suptitle('Benefits of ten hot analyst jobs in SF')
fig = plt.figure(figsize=(8,6))

ax = fig.add_subplot(1,1,1)

sns.boxplot(x = 'Job', y = 'TotalPayBenefits', data = Hot_Job_Salaries,ax=ax,order = order)

ax.set_xticklabels(Hot_Job_Number.values)

plt.xticks(size = 10, rotation = 80)

ax.set_xlabel('Jobs with corresponding Employee numbers')

fig.suptitle('TotalPay+Benefits of ten hot analyst jobs in SF')