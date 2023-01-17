import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

%matplotlib inline

warnings.filterwarnings('ignore')

pd.pandas.set_option('display.max_columns', None)
data = pd.read_csv('../input/states_all_extended.csv')
data.head()
data.shape
data.describe()
vars_with_na = [var for var in data.columns if data[var].isnull().sum()>1]

print(len(vars_with_na))
dict_missing = { var: np.round(data[var].isnull().mean()*100, 3) for var in vars_with_na}
import collections

sorted_dict = sorted(dict_missing.items(), key=lambda kv: kv[1], reverse=True)
sorted_dict
# create a dataframe of missing values

missings_df = pd.DataFrame.from_dict(sorted_dict)

missings_df.columns = ['columns', 'Percent missing']

missings_df.head()
missings_df.shape
d = missings_df.iloc[:50]

plt.figure(figsize = [20, 10]);

g = sns.barplot(x="columns", y="Percent missing", data=d)

g.set_xticklabels(g.get_xticklabels(), rotation=90);
revenues = data[['YEAR','STATE','TOTAL_REVENUE', 'FEDERAL_REVENUE', 'STATE_REVENUE', 'LOCAL_REVENUE']]
revenues_millions = revenues[['TOTAL_REVENUE', 'FEDERAL_REVENUE', 'STATE_REVENUE', 'LOCAL_REVENUE']]/1000000
revenues_millions.head()
# Create a figure and axes

fig, ax = plt.subplots(2, 2, figsize=(20, 10))



# plot the total revenue

ax[0, 0].hist(revenues_millions.TOTAL_REVENUE.dropna(), bins=50)

ax[0, 0].set_title('Total revenue in Millions')

ax[0, 0].set_xlabel('Revenue in Millions')

ax[0, 0].set_ylabel('Count')



# plot the federal revenue

ax[0, 1].hist(revenues_millions.FEDERAL_REVENUE.dropna(), bins=50)

ax[0, 1].set_title('Federal revenue in Millions')

ax[0, 1].set_xlabel('Revenue in Millions')

ax[0, 1].set_ylabel('Count')



# plot the state revenue

ax[1, 0].hist(revenues_millions.STATE_REVENUE.dropna(), bins=50)

ax[1, 0].set_title('State revenue in Millions')

ax[1, 0].set_xlabel('Revenue in Millions')

ax[1, 0].set_ylabel('Count')



# plot the local revenue

ax[1, 1].hist(revenues_millions.LOCAL_REVENUE.dropna(), bins=50)

ax[1, 1].set_title('Local revenue in Millions')

ax[1, 1].set_xlabel('Revenue in Millions')

ax[1, 1].set_ylabel('Count')

base_color = sns.color_palette()[2]

plt.figure(figsize = [10, 10])

plt.title('Revenue in Millions')

dfm = revenues_millions.melt(var_name='columns')

sns.violinplot(data = dfm, y='columns', x='value', color=base_color, inner = 'quartile')
base_color = sns.color_palette()[3]

plt.figure(figsize = [10, 10])

plt.title('Revenue in Millions')

dfm = revenues_millions.melt(var_name='columns')

sns.boxplot(data = dfm, y='columns', x='value', color=base_color)
dfm = pd.melt(revenues, id_vars =['YEAR'], value_vars =['TOTAL_REVENUE', 'FEDERAL_REVENUE', 'STATE_REVENUE', 'LOCAL_REVENUE'])

dfm.columns = ['Year', 'Revenue_type', 'Dollar_Amount']

dfm.head()

dfm.Dollar_Amount = dfm.Dollar_Amount/1000000

dfm.head()

plt.figure(figsize = [20, 10])

plt.title('Revenue in millions')

sns.lineplot(x='Year', y='Dollar_Amount', hue='Revenue_type' , data=dfm, ci=None)
plt.figure(figsize = [20, 10])

plt.title('Average total revenue in Millions')

(revenues.groupby('YEAR')['TOTAL_REVENUE'].mean()/1000000).plot.bar()
total_rev = pd.concat([revenues['TOTAL_REVENUE']/1000000, revenues['YEAR']], axis=1)

f, ax = plt.subplots(figsize=(20, 10))

fig = sns.boxplot(x='YEAR', y="TOTAL_REVENUE", data=total_rev)

plt.ylabel('Total revenue in millions')

plt.title('Annual total revenue')
plt.figure(figsize = [20, 10])

plt.title('Average federal revenue in Millions')

(revenues.groupby('YEAR')['FEDERAL_REVENUE'].mean()/1000000).plot.bar()
total_rev = pd.concat([revenues['FEDERAL_REVENUE']/1000000, revenues['YEAR']], axis=1)

f, ax = plt.subplots(figsize=(20, 10))

fig = sns.boxplot(x='YEAR', y="FEDERAL_REVENUE", data=total_rev)

plt.xlabel('Federal revenue in millions')

plt.title('Annual Federal revenue')
plt.figure(figsize = [20, 10])

plt.title('Average state revenue in Millions')

(revenues.groupby('YEAR')['STATE_REVENUE'].mean()/1000000).plot.bar()
total_rev = pd.concat([revenues['STATE_REVENUE']/1000000, revenues['YEAR']], axis=1)

f, ax = plt.subplots(figsize=(20, 10))

fig = sns.boxplot(x='YEAR', y="STATE_REVENUE", data=total_rev)

plt.xlabel('State revenue in millions')

plt.title('Annual State revenue')
plt.figure(figsize = [20, 10])

plt.title('Average local revenue in Millions')

(revenues.groupby('YEAR')['LOCAL_REVENUE'].mean()/1000000).plot.bar()

total_rev = pd.concat([revenues['LOCAL_REVENUE']/1000000, revenues['YEAR']], axis=1)

f, ax = plt.subplots(figsize=(20, 10))

fig = sns.boxplot(x='YEAR', y="LOCAL_REVENUE", data=total_rev)

plt.xlabel('Local revenue in millions')

plt.title('Annual Local revenue')
rev_data = revenues.groupby('STATE')['TOTAL_REVENUE'].mean()/1000

rev_data = rev_data.reset_index()

rev_data = rev_data.sort_values('TOTAL_REVENUE', ascending=False)

rev_data.plot.barh(x='STATE', y='TOTAL_REVENUE', figsize=(10, 25))

plt.xlabel('Average Total revenue in thousands')
total_rev = pd.concat([revenues['TOTAL_REVENUE']/1000000, revenues['STATE']], axis=1)

f, ax = plt.subplots(figsize=(10, 25))

fig = sns.boxplot(y='STATE', x="TOTAL_REVENUE", data=total_rev)

plt.xlabel('Total revenue in millions')

plt.title('Annual total revenue')
rev_data = revenues.groupby('STATE')['FEDERAL_REVENUE'].mean()/1000

rev_data = rev_data.reset_index()

rev_data = rev_data.sort_values('FEDERAL_REVENUE', ascending=False)

rev_data.plot.barh(x='STATE', y='FEDERAL_REVENUE', figsize=(10, 25))

plt.xlabel('Average federal revenue in thousands')
total_rev = pd.concat([revenues['FEDERAL_REVENUE']/1000000, revenues['STATE']], axis=1)

f, ax = plt.subplots(figsize=(10, 25))

fig = sns.boxplot(y='STATE', x="FEDERAL_REVENUE", data=total_rev)

plt.xlabel('Federal revenue in millions')

plt.title('Annual Federal revenue')
rev_data = revenues.groupby('STATE')['STATE_REVENUE'].mean()/1000

rev_data = rev_data.reset_index()

rev_data = rev_data.sort_values('STATE_REVENUE', ascending=False)

rev_data.plot.barh(x='STATE', y='STATE_REVENUE', figsize=(10, 25))

plt.xlabel('Average state revenue in thousands')
total_rev = pd.concat([revenues['STATE_REVENUE']/1000000, revenues['STATE']], axis=1)

f, ax = plt.subplots(figsize=(10, 25))

fig = sns.boxplot(y='STATE', x="STATE_REVENUE", data=total_rev)

plt.xlabel('State revenue in millions')

plt.title('Annual State revenue')
rev_data = revenues.groupby('STATE')['LOCAL_REVENUE'].mean()/1000

rev_data = rev_data.reset_index()

rev_data = rev_data.sort_values('LOCAL_REVENUE', ascending=False)

rev_data.plot.barh(x='STATE', y='LOCAL_REVENUE', figsize=(10, 25))

plt.xlabel('Average local revenue in thousands')
total_rev = pd.concat([revenues['LOCAL_REVENUE']/1000000, revenues['STATE']], axis=1)

f, ax = plt.subplots(figsize=(10, 25))

fig = sns.boxplot(y='STATE', x="LOCAL_REVENUE", data=total_rev)

plt.xlabel('Local revenue in millions')

plt.title('Annual State revenue')
expenditures = data[['YEAR','STATE','TOTAL_EXPENDITURE','INSTRUCTION_EXPENDITURE','SUPPORT_SERVICES_EXPENDITURE','OTHER_EXPENDITURE', 'CAPITAL_OUTLAY_EXPENDITURE']]
expenditures.head()
melt_expenditures = pd.melt(expenditures, id_vars =['YEAR'], value_vars =['TOTAL_EXPENDITURE','INSTRUCTION_EXPENDITURE','SUPPORT_SERVICES_EXPENDITURE','OTHER_EXPENDITURE', 'CAPITAL_OUTLAY_EXPENDITURE']) 
melt_expenditures.columns = ['Year', 'Expenditure', 'Dollar_Amount']
melt_expenditures.Dollar_Amount = melt_expenditures.Dollar_Amount/1000000

melt_expenditures.head()
plt.figure(figsize = [20, 10])

plt.title('Expenditures in millions')

sns.lineplot(x='Year', y='Dollar_Amount', hue='Expenditure' , data=melt_expenditures, ci=None)

plt.show()
plt.figure(figsize = [20, 10])

plt.title('Average total expenditure in Millions')

(expenditures.groupby('YEAR')['TOTAL_EXPENDITURE'].mean()/1000000).plot.bar()

plt.show()
total_exp = pd.concat([expenditures['TOTAL_EXPENDITURE']/1000000, expenditures['YEAR']], axis=1)

f, ax = plt.subplots(figsize=(20, 10))

fig = sns.boxplot(x='YEAR', y="TOTAL_EXPENDITURE", data=total_exp)

plt.ylabel('Total expenditures in millions')

plt.title('Average total expenditures')

plt.show()
plt.figure(figsize = [20, 10])

plt.title('Average instruction expenditure in Millions')

(expenditures.groupby('YEAR')['INSTRUCTION_EXPENDITURE'].mean()/1000000).plot.bar()

plt.show()
inst_exp = pd.concat([expenditures['INSTRUCTION_EXPENDITURE']/1000000, expenditures['YEAR']], axis=1)

f, ax = plt.subplots(figsize=(20, 10))

fig = sns.boxplot(x='YEAR', y="INSTRUCTION_EXPENDITURE", data=inst_exp)

plt.ylabel('Instruction expenditures in millions')

plt.title('Average instruction expenditures')

plt.show()
plt.figure(figsize = [20, 10])

plt.title('Average support services expenditure in Millions')

(expenditures.groupby('YEAR')['SUPPORT_SERVICES_EXPENDITURE'].mean()/1000000).plot.bar()

plt.show()
ss_exp = pd.concat([expenditures['SUPPORT_SERVICES_EXPENDITURE']/1000000, expenditures['YEAR']], axis=1)

f, ax = plt.subplots(figsize=(20, 10))

fig = sns.boxplot(x='YEAR', y="SUPPORT_SERVICES_EXPENDITURE", data=ss_exp)

plt.ylabel('Support services expenditures in millions')

plt.title('Average support services expenditures')

plt.show()
plt.figure(figsize = [20, 10])

plt.title('Average capital outlet expenditures in Millions')

(expenditures.groupby('YEAR')['CAPITAL_OUTLAY_EXPENDITURE'].mean()/1000000).plot.bar()

plt.show()
co_exp = pd.concat([expenditures['CAPITAL_OUTLAY_EXPENDITURE']/1000000, expenditures['YEAR']], axis=1)

f, ax = plt.subplots(figsize=(20, 10))

fig = sns.boxplot(x='YEAR', y="CAPITAL_OUTLAY_EXPENDITURE", data=co_exp)

plt.ylabel('Capital outlay expenditures in millions')

plt.title('Capital Outlay Expenditures')

plt.show()
plt.figure(figsize = [20, 10])

plt.title('Average other expenditures in Millions')

(expenditures.groupby('YEAR')['OTHER_EXPENDITURE'].mean()/1000000).plot.bar()

plt.show()
o_exp = pd.concat([expenditures['OTHER_EXPENDITURE']/1000000, expenditures['YEAR']], axis=1)

f, ax = plt.subplots(figsize=(20, 10))

fig = sns.boxplot(x='YEAR', y="OTHER_EXPENDITURE", data=o_exp)

plt.ylabel('Other expenditures in millions')

plt.title('Other Expenditures')

plt.show()
exp_data = expenditures.groupby('STATE')['TOTAL_EXPENDITURE'].mean()/1000

exp_data = exp_data.reset_index()

exp_data = exp_data.sort_values('TOTAL_EXPENDITURE', ascending=False)

exp_data.plot.barh(x='STATE', y='TOTAL_EXPENDITURE', figsize=(10, 25))

plt.xlabel('Average Total expenditure in thousands')

plt.show()
total_exp = pd.concat([expenditures['TOTAL_EXPENDITURE']/1000000, expenditures['STATE']], axis=1)

f, ax = plt.subplots(figsize=(10, 25))

fig = sns.boxplot(y='STATE', x="TOTAL_EXPENDITURE", data=total_exp)

plt.xlabel('Total expenditure in millions')

plt.title('Annual total expenditure')

plt.show()
exp_data = expenditures.groupby('STATE')['INSTRUCTION_EXPENDITURE'].mean()/1000

exp_data = exp_data.reset_index()

exp_data = exp_data.sort_values('INSTRUCTION_EXPENDITURE', ascending=False)

exp_data.plot.barh(x='STATE', y='INSTRUCTION_EXPENDITURE', figsize=(10, 25))

plt.xlabel('Average instruction expenditure in thousands')

plt.show()
total_exp = pd.concat([expenditures['INSTRUCTION_EXPENDITURE']/1000000, expenditures['STATE']], axis=1)

f, ax = plt.subplots(figsize=(10, 25))

fig = sns.boxplot(y='STATE', x='INSTRUCTION_EXPENDITURE', data=total_exp)

plt.xlabel('Instruction expenditure in millions')

plt.title('Annual instruction expenditure')

plt.show()
exp_data = expenditures.groupby('STATE')['SUPPORT_SERVICES_EXPENDITURE'].mean()/1000

exp_data = exp_data.reset_index()

exp_data = exp_data.sort_values('SUPPORT_SERVICES_EXPENDITURE', ascending=False)

exp_data.plot.barh(x='STATE', y='SUPPORT_SERVICES_EXPENDITURE', figsize=(10, 25))

plt.xlabel('Average support services expenditure in thousands')

plt.show()
total_exp = pd.concat([expenditures['SUPPORT_SERVICES_EXPENDITURE']/1000000, expenditures['STATE']], axis=1)

f, ax = plt.subplots(figsize=(10, 25))

fig = sns.boxplot(y='STATE', x='SUPPORT_SERVICES_EXPENDITURE', data=total_exp)

plt.xlabel('Support services expenditure in millions')

plt.title('Annual support services expenditure')

plt.show()
exp_data = expenditures.groupby('STATE')['CAPITAL_OUTLAY_EXPENDITURE'].mean()/1000

exp_data = exp_data.reset_index()

exp_data = exp_data.sort_values('CAPITAL_OUTLAY_EXPENDITURE', ascending=False)

exp_data.plot.barh(x='STATE', y='CAPITAL_OUTLAY_EXPENDITURE', figsize=(10, 25))

plt.xlabel('Average capital outlay expenditure in thousands')

plt.show()
total_exp = pd.concat([expenditures['CAPITAL_OUTLAY_EXPENDITURE']/1000000, expenditures['STATE']], axis=1)

f, ax = plt.subplots(figsize=(10, 25))

fig = sns.boxplot(y='STATE', x='CAPITAL_OUTLAY_EXPENDITURE', data=total_exp)

plt.xlabel('Capital outlay expenditure in millions')

plt.title('Annual capital outlay expenditure')

plt.show()
exp_data = expenditures.groupby('STATE')['OTHER_EXPENDITURE'].mean()/1000

exp_data = exp_data.reset_index()

exp_data = exp_data.sort_values('OTHER_EXPENDITURE', ascending=False)

exp_data.plot.barh(x='STATE', y='OTHER_EXPENDITURE', figsize=(10, 25))

plt.xlabel('Average other expenditure in thousands')

plt.show()
total_exp = pd.concat([expenditures['OTHER_EXPENDITURE']/1000000, expenditures['STATE']], axis=1)

f, ax = plt.subplots(figsize=(10, 25))

fig = sns.boxplot(y='STATE', x='OTHER_EXPENDITURE', data=total_exp)

plt.xlabel('Other expenditure in millions')

plt.title('Annual other expenditure')

plt.show()
rev_exp = data[['YEAR', 'STATE', 'TOTAL_EXPENDITURE', 'TOTAL_REVENUE']]

rev_exp['not_spent'] = rev_exp.TOTAL_REVENUE - rev_exp.TOTAL_EXPENDITURE

rev_exp.head()
def plot_line(df, title,x, y, h=None, figsize=[20, 10]):

    plt.figure(figsize=figsize)

    plt.title(title)

    sns.lineplot(x=x, y=y, hue=h , data=df, ci=None)

    

plot_line(rev_exp, 'Average dollar amount not spent', x='YEAR', y='not_spent')
plt.figure(figsize = [20, 10])

plt.title('Average not spent dollar amount')

(rev_exp.groupby('YEAR')['not_spent'].mean()).plot.bar()
plt.figure(figsize = [20, 10])

plt.title('Median not spent dollar amount')

(rev_exp.groupby('YEAR')['not_spent'].median()).plot.bar()
f, ax = plt.subplots(figsize=(25, 15))

fig = sns.boxplot(x='YEAR', y="not_spent", data=rev_exp)

plt.ylabel('Not spent dollar amount')
exp_data = rev_exp.groupby('STATE')['not_spent'].mean()

exp_data = exp_data.reset_index()

exp_data = exp_data.sort_values('not_spent', ascending=False)

exp_data.plot.barh(x='STATE', y='not_spent', figsize=(10, 25))

plt.xlabel('Average dollar amount not spent')

plt.show()
exp_data = rev_exp.groupby('STATE')['not_spent'].median()

exp_data = exp_data.reset_index()

exp_data = exp_data.sort_values('not_spent', ascending=False)

exp_data.plot.barh(x='STATE', y='not_spent', figsize=(10, 25))

plt.xlabel('Median dollar amount not spent')

plt.show()
f, ax = plt.subplots(figsize=(10, 25))

fig = sns.boxplot(y='STATE', x='not_spent', data=rev_exp)

plt.xlabel('Dollar amount not spent')

plt.title('Annual dollar amount not spent')

plt.show()
melt_demographic = pd.melt(data, id_vars =['YEAR', 'STATE'], value_vars =['GRADES_ALL_AM','GRADES_ALL_AS','GRADES_ALL_HI', 'GRADES_ALL_BL', 'GRADES_ALL_WH','GRADES_ALL_HP', 'GRADES_ALL_TR' ])

melt_demographic.columns = ['Year','STATE', 'Demographic', 'Enrollments']

plot_line(melt_demographic, 'Enrollment demographic', x='Year', y='Enrollments', h='Demographic')
f, ax = plt.subplots(figsize=(40, 20))

sns.barplot(x="Year", y="Enrollments", hue="Demographic", data=melt_demographic, ci=None)
main_land_states = data.STATE.unique()[:51]
for state in main_land_states:

    d = melt_demographic[melt_demographic.STATE == state]

    f, ax = plt.subplots(figsize=(40, 20))

    sns.barplot(x="Year", y="Enrollments", hue="Demographic", data=d, ci=None)

    plt.title('Enrollments demographic in ' + state)