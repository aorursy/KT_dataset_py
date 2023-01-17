import pandas as pd
import numpy as np
df = pd.read_csv('../input/charity_navigator_scraped.csv')
df.rename({'compensation_leader_compensation': 'comp_leader_income'}, axis = 'columns', inplace = True)

n = 0
for compensation in df['comp_leader_income']:
    if df.loc[n, 'comp_leader_income'] == 'None reported\\r\\n                 ':
        df.loc[n, 'comp_leader_income'] = None
        n += 1
    elif df.loc[n, 'comp_leader_income'] == '--\\r\\n                 ':
        df.loc[n, 'comp_leader_income'] = None
        n += 1
    elif df.loc[n, 'comp_leader_income'] == 'NaN':
        df.loc[n, 'comp_leader_income'] = None
        n += 1
    else:
        n += 1
df.rename({'compensation_leader_expense_percent': 'comp_leader_expense_pct'}, axis = 'columns', inplace = True)

n = 0
for compensation_percent in df['comp_leader_expense_pct']:
    if df.loc[n, 'comp_leader_expense_pct'] == '--\\r\\n                 ':
        df.loc[n, 'comp_leader_expense_pct'] = None
        n += 1
    else:
        n += 1
df.rename({'compensation_leader_title': 'comp_leader_title'}, axis = 'columns', inplace = True)
df['city'] = df['city'].astype(str)
n = 0
for city in df['city']:
    if '\\r\\n\\t\\t\\t' in df.loc[n, 'city']:
        df.loc[n, 'city'] = str(df.loc[n, 'city']) \
        .split('\\r\\n\\t\\t\\t')[1]
        n += 1
    elif "' " in df.loc[n, 'city']:
        df.loc[n, 'city'] = str(df.loc[n, 'city']) \
        .split("' ")[1]
        n += 1
    else:
        n+= 1
def strip(column):
    column = str(column).replace('$', '')
    column = str(column).replace('%', '')
    column = str(column).replace(',', '')
    return column

df['comp_leader_income'] = pd.to_numeric \
(df['comp_leader_income'].apply(strip), errors='coerce')
df['comp_leader_expense_pct'] = pd.to_numeric \
(df['comp_leader_expense_pct'].apply(strip), errors='coerce')

df['administrative_expenses'] = df['administrative_expenses'].apply(strip).astype(float)
df['excess_or_deficit_for_year'] = df['excess_or_deficit_for_year'].apply(strip).astype(float)
df['fundraising_expenses'] = df['fundraising_expenses'].apply(strip).astype(int)
df['net_assets'] = df['net_assets'].apply(strip).astype(int)
df['other_revenue'] = df['other_revenue'].apply(strip).astype(float)
df['payments_to_affiliates'] = df['payments_to_affiliates'].apply(strip).astype(int)
df['program_expenses'] = df['program_expenses'].apply(strip).astype(int)
df['total_contributions'] = df['total_contributions'].apply(strip).astype(int)

df['charity_name'] = df['charity_name'].astype(str)
df['charity_url'] = df['charity_url'].astype(str)
df['cn_advisory'] = df['cn_advisory'].astype(str)
df['comp_leader_title'] = df['comp_leader_title'].astype(str)
df['organization_type'] = df['organization_type'].astype(str)
df['state'] = df['state'].astype(str)
df['org_type'] = df['organization_type'].str.split(' : ',  expand = True)[0]
df['org_category'] = df['organization_type'].str.split(' : ',  expand = True)[1]
df['org_category'] = df['org_category'].str.split(' ,This rating represents consolidated financial data for these organizations:',  expand = True)[0]
df.drop('organization_type', axis = 1, inplace = True)
n = 0
for org_category in df['org_category']:
    if df.loc[n, 'org_category'][-3:] == ' , ':
        df.loc[n, 'org_category'] = df.loc[n, 'org_category'][:-3]
        n += 1
    elif df.loc[n, 'org_category'][-2:] == ', ':
        df.loc[n, 'org_category'] = df.loc[n, 'org_category'][:-2]
        n += 1
    elif df.loc[n, 'org_category'][-2:] == ' ,':
        df.loc[n, 'org_category'] = df.loc[n, 'org_category'][:-2]
        n += 1
    elif df.loc[n, 'org_category'][-1] == ' ':
        df.loc[n, 'org_category'] = df.loc[n, 'org_category'][:-1]
        n += 1
    else:
        n+= 1
org_type_id = {'Human and Civil Rights': 0, 'Education': 1, 
               'International': 2, 'Religion': 3, 
               'Community Development': 4, 'Environment': 5, 
               'Health': 6, 'Arts, Culture, Humanities': 7, 
               'Human Services': 8, 'Animals': 9,
               'Research and Public Policy': 10}

n = 0
for org_type in df['org_type']:
    df.loc[n, 'org_type_id'] = org_type_id[df.loc[n, 'org_type']]
    n += 1
df.info()
df.head()
df.to_csv('charity_navigator_clean.csv')