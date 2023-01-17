import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



print('Libraries ready.')
mydata = pd.read_csv(r'../input/the-human-freedom-index/hfi_cc_2019.csv')

mydata.head()
rawdata = mydata[['year', 'countries', 'region', 'pf_ss_disappearances', 

                  'pf_religion', 'pf_expression_control', 'pf_movement_women', 

                  'pf_expression', 'ef_government_consumption', 'ef_legal_military', 

                  'ef_legal_integrity', 'ef_legal_enforcement', 'ef_regulation_business_bribes', 'ef_score']].copy()

rawdata.head()
rawdata.info()
rawdata['pf_ss_disappearances'] = pd.to_numeric(rawdata['pf_ss_disappearances'], errors = 'coerce')

rawdata['pf_religion'] = pd.to_numeric(rawdata['pf_religion'], errors = 'coerce')

rawdata['pf_movement_women'] = pd.to_numeric(rawdata['pf_movement_women'], errors = 'coerce')

rawdata['pf_expression_control'] = pd.to_numeric(rawdata['pf_expression_control'], errors = 'coerce')

rawdata['pf_expression'] = pd.to_numeric(rawdata['pf_expression'], errors = 'coerce')

rawdata['ef_government_consumption'] = pd.to_numeric(rawdata['ef_government_consumption'], errors = 'coerce')

rawdata['ef_legal_military'] = pd.to_numeric(rawdata['ef_legal_military'], errors = 'coerce')

rawdata['ef_legal_integrity'] = pd.to_numeric(rawdata['ef_legal_integrity'], errors = 'coerce')

rawdata['ef_legal_enforcement'] = pd.to_numeric(rawdata['ef_legal_enforcement'], errors = 'coerce')

rawdata['ef_regulation_business_bribes'] = pd.to_numeric(rawdata['ef_regulation_business_bribes'], errors = 'coerce')

rawdata['ef_score'] = pd.to_numeric(rawdata['ef_score'], errors = 'coerce')



print('Conversion done.')
rawdata = rawdata.dropna()

rawdata.info()
chartdata = rawdata[['pf_ss_disappearances', 

                  'pf_religion', 'pf_expression_control', 'pf_movement_women', 

                  'pf_expression', 'ef_government_consumption', 'ef_legal_military', 

                  'ef_legal_integrity', 'ef_legal_enforcement', 'ef_regulation_business_bribes', 'ef_score']].copy()

corr = chartdata.corr()

fig, ax = plt.subplots(figsize=(5,5))

plt.title('General Matrix')

sns.heatmap(

    corr, 

    square=True,

    cmap= "YlGnBu",

    ax = ax,

    vmin = -1,

    vmax = 1

)
fig, ax = plt.subplots(figsize=(10,5))

plt.title('Integrity of the Legal System by Year')

plt.xlim(0, 10)

ax = sns.distplot(rawdata[rawdata['year'] == 2017]['ef_legal_integrity'], label = '2017', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2016]['ef_legal_integrity'], label = '2016', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2015]['ef_legal_integrity'], label = '2015', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2014]['ef_legal_integrity'], label = '2014', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2013]['ef_legal_integrity'], label = '2013', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2012]['ef_legal_integrity'], label = '2012', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2011]['ef_legal_integrity'], label = '2011', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2010]['ef_legal_integrity'], label = '2010', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2009]['ef_legal_integrity'], label = '2008', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2008]['ef_legal_integrity'], label = '2009', hist = False)

plt.xlabel('Score')

plt.ylabel('Integrity of the Legal System')



fig, ax = plt.subplots(figsize=(10,5))

plt.title('Government Consumption by Year')

plt.xlim(0, 10)

ax = sns.distplot(rawdata[rawdata['year'] == 2017]['ef_government_consumption'], label = '2017', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2016]['ef_government_consumption'], label = '2016', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2015]['ef_government_consumption'], label = '2015', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2014]['ef_government_consumption'], label = '2014', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2013]['ef_government_consumption'], label = '2013', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2012]['ef_government_consumption'], label = '2012', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2011]['ef_government_consumption'], label = '2011', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2010]['ef_government_consumption'], label = '2010', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2009]['ef_government_consumption'], label = '2009', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2008]['ef_government_consumption'], label = '2008', hist = False)

plt.xlabel('Score')

plt.ylabel('Government Consumption')



chartdata = rawdata[['ef_legal_integrity', 'ef_government_consumption']].copy()

corr = chartdata.corr()

fig, ax = plt.subplots(figsize=(10,10))

plt.title('Q1 General Matrix')

pal = sns.light_palette("navy", reverse = True, as_cmap = True)

sns.heatmap(

    corr, 

    square=True,

    linewidths=.5,

    ax = ax,

    annot = True,

    vmin = -1,

    vmax = 0,

    cmap = pal

)
fig, ax = plt.subplots(figsize=(10,5))

plt.title('Legal Enforcement of Contracts')

plt.xlim(0, 10)

ax = sns.distplot(rawdata[rawdata['year'] == 2017]['ef_legal_enforcement'], label = '2017', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2016]['ef_legal_enforcement'], label = '2016', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2015]['ef_legal_enforcement'], label = '2015', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2014]['ef_legal_enforcement'], label = '2014', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2013]['ef_legal_enforcement'], label = '2013', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2012]['ef_legal_enforcement'], label = '2012', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2011]['ef_legal_enforcement'], label = '2011', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2010]['ef_legal_enforcement'], label = '2010', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2009]['ef_legal_enforcement'], label = '2009', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2008]['ef_legal_enforcement'], label = '2008', hist = False)

plt.xlabel('Score')

plt.ylabel('Legal Enforcement of Contracts')



fig, ax = plt.subplots(figsize=(10,5))

plt.title('Extra Payments, Bribes, Favoritism')

plt.xlim(0, 10)

ax = sns.distplot(rawdata[rawdata['year'] == 2017]['ef_regulation_business_bribes'], label = '2017', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2016]['ef_regulation_business_bribes'], label = '2016', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2015]['ef_regulation_business_bribes'], label = '2015', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2014]['ef_regulation_business_bribes'], label = '2014', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2013]['ef_regulation_business_bribes'], label = '2013', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2012]['ef_regulation_business_bribes'], label = '2012', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2011]['ef_regulation_business_bribes'], label = '2011', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2010]['ef_regulation_business_bribes'], label = '2010', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2009]['ef_regulation_business_bribes'], label = '2008', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2008]['ef_regulation_business_bribes'], label = '2009', hist = False)

plt.xlabel('Score')

plt.ylabel('Extra Payments, Bribes, Favoritism')



chartdata = rawdata[['ef_legal_enforcement', 'ef_regulation_business_bribes']].copy()

corr = chartdata.corr()

fig, ax = plt.subplots(figsize=(10,10))

plt.title('Q2 General Matrix')

pal = sns.light_palette("navy", reverse = True, as_cmap = True)

sns.heatmap(

    corr, 

    square=True,

    linewidths=.5,

    ax = ax,

    annot = True,

    vmin = 0,

    vmax = 1,

    cmap = pal

)
fig, ax = plt.subplots(figsize=(10,5))

plt.title('Freedom of Expression and Information')

plt.xlim(0, 10)

ax = sns.distplot(rawdata[rawdata['year'] == 2017]['pf_expression'], label = '2017', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2016]['pf_expression'], label = '2016', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2015]['pf_expression'], label = '2015', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2014]['pf_expression'], label = '2014', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2013]['pf_expression'], label = '2013', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2012]['pf_expression'], label = '2012', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2011]['pf_expression'], label = '2011', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2010]['pf_expression'], label = '2010', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2009]['pf_expression'], label = '2009', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2008]['pf_expression'], label = '2008', hist = False)

plt.xlabel('Score')

plt.ylabel('Freedom of Expression and Information')



fig, ax = plt.subplots(figsize=(10,5))

plt.title('Economic Freedom')

plt.xlim(0, 10)

ax = sns.distplot(rawdata[rawdata['year'] == 2017]['ef_score'], label = '2017', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2016]['ef_score'], label = '2016', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2015]['ef_score'], label = '2015', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2014]['ef_score'], label = '2014', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2013]['ef_score'], label = '2013', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2012]['ef_score'], label = '2012', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2011]['ef_score'], label = '2011', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2010]['ef_score'], label = '2010', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2009]['ef_score'], label = '2008', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2008]['ef_score'], label = '2009', hist = False)

plt.xlabel('Score')

plt.ylabel('Economic Freedom')



chartdata = rawdata[['pf_expression', 'ef_score']].copy()

corr = chartdata.corr()

fig, ax = plt.subplots(figsize=(10,10))

plt.title('Q3 General Matrix')

pal = sns.light_palette("navy", reverse = True, as_cmap = True)

sns.heatmap(

    corr, 

    square=True,

    linewidths=.5,

    ax = ax,

    annot = True,

    vmin = 0,

    vmax = 1,

    cmap = pal

)
fig, ax = plt.subplots(figsize=(10,5))

plt.title('Political Pressures and Controls on Media Content')

plt.xlim(0, 10)

ax = sns.distplot(rawdata[rawdata['year'] == 2017]['pf_expression_control'], label = '2017', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2016]['pf_expression_control'], label = '2016', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2015]['pf_expression_control'], label = '2015', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2014]['pf_expression_control'], label = '2014', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2013]['pf_expression_control'], label = '2013', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2012]['pf_expression_control'], label = '2012', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2011]['pf_expression_control'], label = '2011', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2010]['pf_expression_control'], label = '2010', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2009]['pf_expression_control'], label = '2009', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2008]['pf_expression_control'], label = '2008', hist = False)

plt.xlabel('Score')

plt.ylabel('Political Pressures and Controls on Media Content')



fig, ax = plt.subplots(figsize=(10,5))

plt.title('Freedom of Religion')

plt.xlim(0, 10)

ax = sns.distplot(rawdata[rawdata['year'] == 2017]['pf_religion'], label = '2017', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2016]['pf_religion'], label = '2016', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2015]['pf_religion'], label = '2015', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2014]['pf_religion'], label = '2014', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2013]['pf_religion'], label = '2013', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2012]['pf_religion'], label = '2012', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2011]['pf_religion'], label = '2011', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2010]['pf_religion'], label = '2010', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2009]['pf_religion'], label = '2008', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2008]['pf_religion'], label = '2009', hist = False)

plt.xlabel('Score')

plt.ylabel('Freedom of Religion')



chartdata = rawdata[['pf_expression_control', 'pf_religion']].copy()

corr = chartdata.corr()

fig, ax = plt.subplots(figsize=(10,10))

plt.title('Q4 General Matrix')

pal = sns.light_palette("navy", reverse = True, as_cmap = True)

sns.heatmap(

    corr, 

    square=True,

    linewidths=.5,

    ax = ax,

    annot = True,

    vmin = 0,

    vmax = 1,

    cmap = pal

)
fig, ax = plt.subplots(figsize=(10,5))

plt.title('Military Interference in Rule of Law and Politics')

plt.xlim(0, 10)

ax = sns.distplot(rawdata[rawdata['year'] == 2017]['ef_legal_military'], label = '2017', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2016]['ef_legal_military'], label = '2016', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2015]['ef_legal_military'], label = '2015', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2014]['ef_legal_military'], label = '2014', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2013]['ef_legal_military'], label = '2013', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2012]['ef_legal_military'], label = '2012', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2011]['ef_legal_military'], label = '2011', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2010]['ef_legal_military'], label = '2010', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2009]['ef_legal_military'], label = '2009', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2008]['ef_legal_military'], label = '2008', hist = False)

plt.xlabel('Score')

plt.ylabel('Military Interference in Rule of Law and Politics')



fig, ax = plt.subplots(figsize=(10,5))

plt.title('Disappearances, Conflicts, and Terrorism')

plt.xlim(0, 10)

ax = sns.distplot(rawdata[rawdata['year'] == 2017]['pf_ss_disappearances'], label = '2017', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2016]['pf_ss_disappearances'], label = '2016', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2015]['pf_ss_disappearances'], label = '2015', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2014]['pf_ss_disappearances'], label = '2014', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2013]['pf_ss_disappearances'], label = '2013', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2012]['pf_ss_disappearances'], label = '2012', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2011]['pf_ss_disappearances'], label = '2011', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2010]['pf_ss_disappearances'], label = '2010', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2009]['pf_ss_disappearances'], label = '2008', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2008]['pf_ss_disappearances'], label = '2009', hist = False)

plt.xlabel('Score')

plt.ylabel('Disappearances, Conflicts, and Terrorism')



chartdata = rawdata[['ef_legal_military', 'pf_ss_disappearances']].copy()

corr = chartdata.corr()

fig, ax = plt.subplots(figsize=(10,10))

plt.title('Q5 General Matrix')

pal = sns.light_palette("navy", reverse = True, as_cmap = True)

sns.heatmap(

    corr, 

    square=True,

    linewidths=.5,

    ax = ax,

    annot = True,

    vmin = 0,

    vmax = 1,

    cmap = pal

)
fig, ax = plt.subplots(figsize=(10,5))

plt.title('Freedom of Women’s Movement')

plt.xlim(0, 10)

ax = sns.distplot(rawdata[rawdata['year'] == 2017]['pf_movement_women'], label = '2017', hist = False, kde_kws = {'bw': 1})

ax = sns.distplot(rawdata[rawdata['year'] == 2016]['pf_movement_women'], label = '2016', hist = False, kde_kws = {'bw': 1})

ax = sns.distplot(rawdata[rawdata['year'] == 2015]['pf_movement_women'], label = '2015', hist = False, kde_kws = {'bw': 1})

ax = sns.distplot(rawdata[rawdata['year'] == 2014]['pf_movement_women'], label = '2014', hist = False, kde_kws = {'bw': 1})

ax = sns.distplot(rawdata[rawdata['year'] == 2013]['pf_movement_women'], label = '2013', hist = False, kde_kws = {'bw': 1})

ax = sns.distplot(rawdata[rawdata['year'] == 2012]['pf_movement_women'], label = '2012', hist = False, kde_kws = {'bw': 1})

ax = sns.distplot(rawdata[rawdata['year'] == 2011]['pf_movement_women'], label = '2011', hist = False, kde_kws = {'bw': 1})

ax = sns.distplot(rawdata[rawdata['year'] == 2010]['pf_movement_women'], label = '2010', hist = False, kde_kws = {'bw': 1})

ax = sns.distplot(rawdata[rawdata['year'] == 2009]['pf_movement_women'], label = '2009', hist = False, kde_kws = {'bw': 1})

ax = sns.distplot(rawdata[rawdata['year'] == 2008]['pf_movement_women'], label = '2008', hist = False, kde_kws = {'bw': 1})

plt.xlabel('Score')

plt.ylabel('Freedom of Women’s Movement')



fig, ax = plt.subplots(figsize=(10,5))

plt.title('Economic Freedom')

plt.xlim(0, 10)

ax = sns.distplot(rawdata[rawdata['year'] == 2017]['ef_score'], label = '2017', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2016]['ef_score'], label = '2016', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2015]['ef_score'], label = '2015', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2014]['ef_score'], label = '2014', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2013]['ef_score'], label = '2013', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2012]['ef_score'], label = '2012', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2011]['ef_score'], label = '2011', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2010]['ef_score'], label = '2010', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2009]['ef_score'], label = '2008', hist = False)

ax = sns.distplot(rawdata[rawdata['year'] == 2008]['ef_score'], label = '2009', hist = False)

plt.xlabel('Score')

plt.ylabel('Economic Freedom')



chartdata = rawdata[['pf_movement_women', 'ef_score']].copy()

corr = chartdata.corr()

fig, ax = plt.subplots(figsize=(10,10))

plt.title('Q6 General Matrix')

pal = sns.light_palette("navy", reverse = True, as_cmap = True)

sns.heatmap(

    corr, 

    square=True,

    linewidths=.5,

    ax = ax,

    annot = True,

    vmin = 0,

    vmax = 1,

    cmap = pal

)
#q1

a17 = rawdata[rawdata['year'] == 2017]['ef_legal_integrity'].corr(rawdata['ef_government_consumption'])

a16 = rawdata[rawdata['year'] == 2016]['ef_legal_integrity'].corr(rawdata['ef_government_consumption'])

a15 = rawdata[rawdata['year'] == 2015]['ef_legal_integrity'].corr(rawdata['ef_government_consumption'])

a14 = rawdata[rawdata['year'] == 2014]['ef_legal_integrity'].corr(rawdata['ef_government_consumption'])

a13 = rawdata[rawdata['year'] == 2013]['ef_legal_integrity'].corr(rawdata['ef_government_consumption'])

a12 = rawdata[rawdata['year'] == 2012]['ef_legal_integrity'].corr(rawdata['ef_government_consumption'])

a11 = rawdata[rawdata['year'] == 2011]['ef_legal_integrity'].corr(rawdata['ef_government_consumption'])

a10 = rawdata[rawdata['year'] == 2010]['ef_legal_integrity'].corr(rawdata['ef_government_consumption'])

a09 = rawdata[rawdata['year'] == 2009]['ef_legal_integrity'].corr(rawdata['ef_government_consumption'])

a08 = rawdata[rawdata['year'] == 2008]['ef_legal_integrity'].corr(rawdata['ef_government_consumption'])





#q2

b17 = str(rawdata[rawdata['year'] == 2017]['ef_legal_enforcement'].corr(rawdata['ef_regulation_business_bribes']))

b16 = str(rawdata[rawdata['year'] == 2016]['ef_legal_enforcement'].corr(rawdata['ef_regulation_business_bribes']))

b15 = str(rawdata[rawdata['year'] == 2015]['ef_legal_enforcement'].corr(rawdata['ef_regulation_business_bribes']))

b14 = str(rawdata[rawdata['year'] == 2014]['ef_legal_enforcement'].corr(rawdata['ef_regulation_business_bribes']))

b13 = str(rawdata[rawdata['year'] == 2013]['ef_legal_enforcement'].corr(rawdata['ef_regulation_business_bribes']))

b12 = str(rawdata[rawdata['year'] == 2012]['ef_legal_enforcement'].corr(rawdata['ef_regulation_business_bribes']))

b11 = str(rawdata[rawdata['year'] == 2011]['ef_legal_enforcement'].corr(rawdata['ef_regulation_business_bribes']))

b10 = str(rawdata[rawdata['year'] == 2010]['ef_legal_enforcement'].corr(rawdata['ef_regulation_business_bribes']))

b09 = str(rawdata[rawdata['year'] == 2009]['ef_legal_enforcement'].corr(rawdata['ef_regulation_business_bribes']))

b08 = str(rawdata[rawdata['year'] == 2008]['ef_legal_enforcement'].corr(rawdata['ef_regulation_business_bribes']))





#q3

c17 = str(rawdata[rawdata['year'] == 2017]['pf_expression'].corr(rawdata['ef_score']))

c16 = str(rawdata[rawdata['year'] == 2016]['pf_expression'].corr(rawdata['ef_score']))

c15 = str(rawdata[rawdata['year'] == 2015]['pf_expression'].corr(rawdata['ef_score']))

c14 = str(rawdata[rawdata['year'] == 2014]['pf_expression'].corr(rawdata['ef_score']))

c13 = str(rawdata[rawdata['year'] == 2013]['pf_expression'].corr(rawdata['ef_score']))

c12 = str(rawdata[rawdata['year'] == 2012]['pf_expression'].corr(rawdata['ef_score']))

c11 = str(rawdata[rawdata['year'] == 2011]['pf_expression'].corr(rawdata['ef_score']))

c10 = str(rawdata[rawdata['year'] == 2010]['pf_expression'].corr(rawdata['ef_score']))

c09 = str(rawdata[rawdata['year'] == 2009]['pf_expression'].corr(rawdata['ef_score']))

c08 = str(rawdata[rawdata['year'] == 2008]['pf_expression'].corr(rawdata['ef_score']))





#q4

d17 = str(rawdata[rawdata['year'] == 2017]['pf_expression_control'].corr(rawdata['pf_religion']))

d16 = str(rawdata[rawdata['year'] == 2016]['pf_expression_control'].corr(rawdata['pf_religion']))

d15 = str(rawdata[rawdata['year'] == 2015]['pf_expression_control'].corr(rawdata['pf_religion']))

d14 = str(rawdata[rawdata['year'] == 2014]['pf_expression_control'].corr(rawdata['pf_religion']))

d13 = str(rawdata[rawdata['year'] == 2013]['pf_expression_control'].corr(rawdata['pf_religion']))

d12 = str(rawdata[rawdata['year'] == 2012]['pf_expression_control'].corr(rawdata['pf_religion']))

d11 = str(rawdata[rawdata['year'] == 2011]['pf_expression_control'].corr(rawdata['pf_religion']))

d10 = str(rawdata[rawdata['year'] == 2010]['pf_expression_control'].corr(rawdata['pf_religion']))

d09 = str(rawdata[rawdata['year'] == 2009]['pf_expression_control'].corr(rawdata['pf_religion']))

d08 = str(rawdata[rawdata['year'] == 2008]['pf_expression_control'].corr(rawdata['pf_religion']))





#q5

e17 = str(rawdata[rawdata['year'] == 2017]['ef_legal_military'].corr(rawdata['pf_ss_disappearances']))

e16 = str(rawdata[rawdata['year'] == 2016]['ef_legal_military'].corr(rawdata['pf_ss_disappearances']))

e15 = str(rawdata[rawdata['year'] == 2015]['ef_legal_military'].corr(rawdata['pf_ss_disappearances']))

e14 = str(rawdata[rawdata['year'] == 2014]['ef_legal_military'].corr(rawdata['pf_ss_disappearances']))

e13 = str(rawdata[rawdata['year'] == 2013]['ef_legal_military'].corr(rawdata['pf_ss_disappearances']))

e12 = str(rawdata[rawdata['year'] == 2012]['ef_legal_military'].corr(rawdata['pf_ss_disappearances']))

e11 = str(rawdata[rawdata['year'] == 2011]['ef_legal_military'].corr(rawdata['pf_ss_disappearances']))

e10 = str(rawdata[rawdata['year'] == 2010]['ef_legal_military'].corr(rawdata['pf_ss_disappearances']))

e09 = str(rawdata[rawdata['year'] == 2009]['ef_legal_military'].corr(rawdata['pf_ss_disappearances']))

e08 = str(rawdata[rawdata['year'] == 2008]['ef_legal_military'].corr(rawdata['pf_ss_disappearances']))



#q6

f17 = str(rawdata[rawdata['year'] == 2017]['pf_movement_women'].corr(rawdata['ef_score']))

f16 = str(rawdata[rawdata['year'] == 2016]['pf_movement_women'].corr(rawdata['ef_score']))

f15 = str(rawdata[rawdata['year'] == 2015]['pf_movement_women'].corr(rawdata['ef_score']))

f14 = str(rawdata[rawdata['year'] == 2014]['pf_movement_women'].corr(rawdata['ef_score']))

f13 = str(rawdata[rawdata['year'] == 2013]['pf_movement_women'].corr(rawdata['ef_score']))

f12 = str(rawdata[rawdata['year'] == 2012]['pf_movement_women'].corr(rawdata['ef_score']))

f11 = str(rawdata[rawdata['year'] == 2011]['pf_movement_women'].corr(rawdata['ef_score']))

f10 = str(rawdata[rawdata['year'] == 2010]['pf_movement_women'].corr(rawdata['ef_score']))

f09 = str(rawdata[rawdata['year'] == 2009]['pf_movement_women'].corr(rawdata['ef_score']))

f08 = str(rawdata[rawdata['year'] == 2008]['pf_movement_women'].corr(rawdata['ef_score']))
datatrend = {'Year': [2017, 2016, 2015, 2014, 2013, 2012, 2011, 2010, 2009, 2008],

            'Military Interference in Rule of Law and Politics | Disappearances, Conflicts, and Terrorism': [0.50, 0.47, 0.44, 0.45, 0.55, 0.55, 0.55, 0.57, 0.57, 0.53],

            'Integrity of the Legal System | Government Consumption': [-0.60, -0.60, -0.55, -0.50, -0.54, -0.58, -0.60, -0.62, -0.62, -0.64],

            'Political Pressures and Controls On Media Content | Freedom of Religion': [0.58, 0.59, 0.59, 0.42, 0.48, 0.49, 0.50, 0.50, 0.35, 0.50],

            'Legal Enforcement of Contracts | Extra Payments, Bribes, Favoritism': [0.55, 0.56, 0.56, 0.54, 0.54, 0.53, 0.46, 0.44, 0.44, 0.46],

            'Freedom of Expression and Information | Economic Freedom': [0.56, 0.55, 0.56, 0.56, 0.50, 0.50, 0.47, 0.42, 0.45, 0.45],

            'Freedom of Women’s Movement | Economic Freedom': [0.24, 0.38, 0.38, 0.33, 0.32, 0.45, 0.44, 0.23, 0.30, 0.29]

            }

datatrend = pd.DataFrame(datatrend, 

                         columns = ['Year', 

                                    'Integrity of the Legal System | Government Consumption',

                                    'Legal Enforcement of Contracts | Extra Payments, Bribes, Favoritism',

                                    'Freedom of Expression and Information | Economic Freedom',

                                    'Political Pressures and Controls On Media Content | Freedom of Religion',

                                    'Military Interference in Rule of Law and Politics | Disappearances, Conflicts, and Terrorism',

                                    'Freedom of Women’s Movement | Economic Freedom'

                                   ]

                        )

datatrend.head(10)
#Q1

plt.title('Integrity of the Legal System | Government Consumption')

sns.regplot(x = 'Year', y = 'Integrity of the Legal System | Government Consumption', data = datatrend)

plt.ylabel('')

plt.xlabel('')
#Q2

plt.title('Legal Enforcement of Contracts | Extra Payments/Bribes/Favoritism')

sns.regplot(x = 'Year', y = 'Legal Enforcement of Contracts | Extra Payments, Bribes, Favoritism', data = datatrend)

plt.ylabel('')

plt.xlabel('')
#Q3

plt.title('Freedom of Expression and Information | Economic Freedom')

sns.regplot(x = 'Year', y = 'Freedom of Expression and Information | Economic Freedom', data = datatrend)

plt.ylabel('')

plt.xlabel('')
#Q4

plt.title('Political Pressures and Controls On Media Content | Freedom of Religion')

sns.regplot(x = 'Year', y = 'Political Pressures and Controls On Media Content | Freedom of Religion', data = datatrend)

plt.ylabel('')

plt.xlabel('')
#Q5

plt.title('Military Interference in Rule of Law and Politics | Disappearances, Conflicts, and Terrorism')

sns.regplot(x = 'Year', y = 'Military Interference in Rule of Law and Politics | Disappearances, Conflicts, and Terrorism', data = datatrend)

plt.ylabel('')

plt.xlabel('')
#Q6

plt.title('Freedom of Women’s Movement | Economic Freedom')

sns.regplot(x = 'Year', y = 'Freedom of Women’s Movement | Economic Freedom', data = datatrend)

plt.ylabel('')

plt.xlabel('')