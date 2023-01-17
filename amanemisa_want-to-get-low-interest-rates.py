import pandas as pd

import numpy as np

import sqlite3

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
con = sqlite3.connect('../input/database.sqlite')

loan = pd.read_csv('../input/loan.csv')
loan.dtypes
loan.head()
plt.rc("figure", figsize=(6, 4))

loan["loan_amnt"].hist()

plt.title("distribution of loan amount")
plt.rc("figure", figsize=(6, 4))

loan["int_rate"].hist()

plt.title("distribution of interest rate")
loan[["loan_amnt","annual_inc"]].dropna().describe()
loan_rate_related = pd.read_sql_query( """

SELECT loan_amnt, term, int_rate, grade, emp_title, emp_length, home_ownership, annual_inc,issue_d,

purpose, title, addr_state,application_type,

CASE WHEN loan_amnt < 8000 THEN 'low' 

     WHEN loan_amnt >= 8000 AND loan_amnt < 13000 THEN 'medium-low'

     WHEN loan_amnt >= 13000 AND loan_amnt < 20000 THEN 'medium-high'

     WHEN loan_amnt >= 20000 THEN 'high' END as loan_amnt_level,

CASE WHEN annual_inc < 45000 THEN 'low'

     WHEN annual_inc >= 45000 AND annual_inc <65000 THEN 'medium-low'

     WHEN annual_inc >= 65000 AND annual_inc < 90000 THEN 'medium-high'

     WHEN annual_inc >= 90000 THEN 'high' END as annual_inc_level

FROM loan

""",con)
loan_rate_related.head()
loan_rate_related.shape
loan_rate_related.isnull().sum()
loan_rate_related = loan_rate_related.dropna(subset=["loan_amnt","term","int_rate","grade","emp_length","home_ownership","annual_inc",

                                              "issue_d","purpose","addr_state","application_type"])
loan_rate_related.shape
loan_rate_related.dtypes
loan_rate_related["int_rate"]=loan_rate_related["int_rate"].apply(lambda x: float(x.rstrip("%")))
order = ["low", "medium-low","medium-high","high"]

sns.boxplot(x='loan_amnt_level',y="int_rate",data = loan_rate_related,order=order)

plt.title("how 'loan amount' affects 'interest rate' ")
plt.rc("figure", figsize=(6, 4))

sns.boxplot(x='term',y="int_rate",data = loan_rate_related)

plt.title("how 'term' affects 'interest rate'")
plt.rc("figure", figsize=(6, 4))

sns.boxplot(x='grade',y="int_rate",data = loan_rate_related,order=["A","B","C","D","E","F","G"])

plt.title("how 'grade' affects 'interest rate'")
loan["emp_title"].unique()
loan["emp_length"].unique()
order = ['1 year', '2 years', '3 years', '4 years',

       '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years', 'n/a']

plt.rc("figure", figsize=(6, 4))

sns.boxplot(x='emp_length',y="int_rate",data = loan_rate_related,order=order)

plt.title("how 'employee length' affects 'interest rate'")

plt.xticks(size = 10,rotation = 80)
sns.boxplot(x='annual_inc_level',y="int_rate",data = loan_rate_related)

plt.title("how 'annual income' affects 'interest rate'")
plt.rc("figure", figsize=(6, 4))

sns.boxplot(x='home_ownership',y="int_rate",data = loan_rate_related)

plt.title("how 'home_ownership' affects 'int_rate'")
loan_rate_related["issue_d"].unique()
loan_rate_related["issue_d"] = loan_rate_related["issue_d"].str.split("-")
loan_rate_related["issue_month"] = loan_rate_related["issue_d"].str[0]
loan_rate_related["issue_year"] = loan_rate_related["issue_d"].str[1]
order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

sns.boxplot(x='issue_month',y="int_rate",data = loan_rate_related,order = order)

plt.title("how 'issu_month' affects 'interest rate'")
order = np.sort(loan_rate_related["issue_year"].unique().tolist())

sns.boxplot(x='issue_year',y="int_rate",data = loan_rate_related, order = order)

plt.title("how 'issue_year' affects 'interest rate'")
rate_by_purpose = pd.read_sql_query( """

SELECT purpose, avg(int_rate) AS avg_rate

FROM loan

GROUP BY purpose

ORDER BY avg_rate desc

""",con)

rate_by_purpose
order = rate_by_purpose["purpose"].tolist()

sns.boxplot(x='purpose',y="int_rate",data = loan_rate_related, order = order)

plt.xticks(size = 10,rotation = 80)

plt.title("how 'purpose' affects 'interest rate'")
rate_by_state = pd.read_sql_query( """

SELECT addr_state, avg(int_rate) AS avg_rate

FROM loan

GROUP BY addr_state

ORDER BY avg_rate desc

""",con)

rate_by_state
plt.rc("figure", figsize=(9, 4))

order = rate_by_state["addr_state"].tolist()

sns.boxplot(x='addr_state',y="int_rate",data = loan_rate_related, order = order)

plt.xticks(size = 10,rotation = 80)

plt.title("how 'state' affects 'interest rate'")
plt.rc("figure", figsize=(6, 4))

sns.boxplot(x='application_type',y="int_rate",data = loan_rate_related)

plt.title("how 'application type' affects 'interest rate'")