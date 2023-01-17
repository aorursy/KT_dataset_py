#!pip install pywaffle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from pywaffle import Waffle 
import re
plt.style.use('fivethirtyeight')
whs = pd.read_csv('../input/white-house-staff-salaries-20172020/wh_staff_dataset.csv')
cpi = pd.read_csv('../input/inflation-data/cpi_dataset.csv', index_col='year', usecols=['year','average'])
whs.loc[whs['year'].between(1997,2000),'president'] = 'Clinton'
whs.loc[whs['year'].between(2001,2008),'president'] = 'Bush'
whs.loc[whs['year'].between(2009,2016),'president'] = 'Obama'
whs.loc[whs['year'].between(2017,2020),'president'] = 'Trump'
whs['inflation_adjusted_salary'] = ((whs.salary * cpi.loc[2020,'average']) / cpi.loc[whs.year,'average'].reset_index(drop=True)).round(2)
whs.sample(5)
whs.groupby('president')[['gender','status','pay_basis']].agg(['describe'])
print(whs.status.unique())
print(whs.pay_basis.unique())
pd.concat([whs[whs.status=='Employee (part-time)'], whs[whs.pay_basis=='Per Diem']])
# Make a part-timer an employee
whs.loc[whs['status']=='Employee (part-time)','status'] = 'Employee'

# Per_Diem_value * 52 weeks * 5 days
whs.loc[whs['pay_basis']=='Per Diem','salary'] = whs[whs['pay_basis']=='Per Diem'].salary*52*5
whs.loc[whs['pay_basis']=='Per Diem','pay_basis'] = 'Per Annum'
fig,(ax1,ax2,ax3,ax4) = plt.subplots(4,1, figsize=(12,10))
whs.groupby('year').salary.mean().plot.line(color='green', xticks=range(1996,2021), ax=ax1)
whs.groupby('year').inflation_adjusted_salary.mean().plot.line(rot=45, color='lightgreen', ax=ax1)
ax1.set(ylabel='mean', yticks=range(50000,150000,20000))

whs.groupby('year').salary.median().plot.line(color='green', xticks=range(1996,2021), ax=ax2)
whs.groupby('year').inflation_adjusted_salary.median().plot.line(rot=45, color='lightgreen', ax=ax2)
ax2.set(ylabel='median', yticks=range(50000,150000,20000))
ax2.legend(['salary','inflation_adjusted_salary'])

whs.groupby('year').name.count().plot.line(xticks=range(1996,2021), ax=ax3)
ax3.set(ylabel='count')

whs.groupby('year').salary.sum().plot.line(rot=45, sharex=True, color='red',  xticks=range(1996,2021), ax=ax4)
whs.groupby('year').inflation_adjusted_salary.sum().plot.line(rot=45, sharex=True, color='darksalmon', ax=ax4)
ax4.set(ylabel='budget')
ax4.legend(['sum of salaries', 'sum of inflated salaries'])
plt.show()
whs.name.value_counts().value_counts().head(12)
veterans = whs[whs.name.isin(whs.name.value_counts().head(6).index)].reset_index(drop=True)
veterans.name.unique()
df = veterans.pivot(index='year', columns='name', values='salary')
df.plot.line(figsize=(12,8), xticks=range(1997,2021), rot=45, style='.-', linewidth=2)
plt.show()
fig, (ax1,ax2) = plt.subplots(2,1, figsize=(10,12))
df = whs.groupby(['year','gender']).salary.count().unstack()
df.plot.bar(rot=45, ax=ax1, legend=False)
ax1.set(ylabel='count')

df = whs.groupby(['year','gender']).salary.median().unstack().reset_index()
df.plot.scatter(x='year',y='Female',rot=45, ax=ax2, color='blue', xticks=range(1997,2021), s=30, marker='^')
df.plot.scatter(x='year',y='Male',rot=45, ax=ax2, color='red', s=20, marker='s')
ax2.set(ylabel='median salary')
ax2.legend(['Female', 'Male'])

# Basic regression line
m, b = np.polyfit(df.year,df.Male, 1)
ax2.plot(df.year, m*df.year + b, linewidth=1, color='salmon')
m, b = np.polyfit(df.year,df.Female, 1)
ax2.plot(df.year, m*df.year + b, linewidth=1, color='lightblue')

plt.show()
# Who are the highest paid staffers in each administration?
whs.iloc[whs.groupby('president').salary.idxmax()]
# Top 10 highest payed staffers
whs.sort_values('salary', ascending=False).head(10)
# Top 10 highest paid staffers, adjusted for inflation
whs.sort_values('inflation_adjusted_salary', ascending=False).head(10)
fig, (ax1,ax2) = plt.subplots(2, 1, figsize=(12, 6))
df = whs[whs.inflation_adjusted_salary>150000].groupby(['year','status']).salary.count().unstack()
df.plot.bar(rot=45, figsize=(12,8),stacked=True, ax=ax1)
ax1.set(ylabel='count')
ax1.get_legend().remove()
whs.groupby(['year','status']).salary.median().unstack().plot.bar(rot=45, figsize=(12,8),stacked=True, ax=ax2)
ax2.set(ylabel='median salary')
plt.show()
whsx = whs.drop_duplicates(subset=['name','status'], ignore_index=True)
whsy = whsx.groupby('name').status.count()
both = whsx.set_index('name').loc[whsy[whsy > 1].index,:].reset_index().drop_duplicates(subset=['name'], ignore_index=True)
df = both.president.value_counts().reset_index()

plt.figure(
    FigureClass=Waffle,
    rows=5,
    values=df.president,
    labels=list(df['index']),
    colors=['royalblue','seagreen','orangered','darkmagenta'],
    legend={'loc': 'lower left', 'bbox_to_anchor': (0, -0.2), 'framealpha': 0, 'ncol': len(df.president)},
    figsize=(12, 6)
)
plt.show()
fbush = whs[whs['year'].between(2001,2004)]
fobama = whs[whs['year'].between(2009,2012)]
ftrump = whs[whs['year'].between(2017,2020)] 

sclinton = whs[whs['year'].between(1997,2000)]
sbush = whs[whs['year'].between(2005,2008)]
sobama = whs[whs['year'].between(2013,2016)]

fdf = pd.concat([fbush,fobama,ftrump], ignore_index=True)
sdf = pd.concat([sclinton,sbush,sobama], ignore_index=True)
# Basic stats for first terms 
fdf.groupby('president').salary.agg(['mean','median','sum','max','count'])
# Basic stats for second terms
sdf.groupby('president').salary.agg(['mean','median','sum','max','count'])
fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(12, 6))
my_pal = {'Clinton':'darkmagenta','Bush':'seagreen','Obama':'royalblue','Trump':'orangered'}
sns.boxplot(x='president', y='inflation_adjusted_salary', data=fdf, 
            ax=ax1, palette=my_pal, linewidth=2, width=0.4)
ax1.set_title('First Term')
ax1.set_yticks(range(0,330000,30000))
sns.boxplot(x='president', y='inflation_adjusted_salary', data=sdf, 
            ax=ax2, palette=my_pal, linewidth=2, width=0.4)
ax2.set_title('Second Term')
ax2.set_yticks(range(0,330000,30000))
plt.show()

#Full term employees
fte1 = []
for pres in [fbush,fobama,ftrump]:
    fte1.append(pres.name.value_counts().value_counts().sum() - pres.name.value_counts().value_counts().loc[4])
fte2 = []
for pres in [sclinton,sbush,sobama]:
    fte2.append(pres.name.value_counts().value_counts().sum() - pres.name.value_counts().value_counts().loc[4])

fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
sns.pointplot(x=['Bush','Obama','Trump'], y=fte1, ax=ax1)
sns.pointplot(x=['Clinton','Bush','Obama'], y=fte2, ax=ax2)
ax1.set(yticks=range(700,900,40))
plt.show()
whs.groupby('year')[['salary','inflation_adjusted_salary']].agg(['sum','count'])
fig, (ax1,ax2,ax3) = plt.subplots(3,1, figsize=(12, 16), sharex=True)

ax1.set_title('salary distribution')
whs.loc[whs['year']==2000,'salary'].plot.hist(bins=25, color='darkmagenta', alpha=0.5, ax=ax1)
whs.loc[whs['year']==2001,'salary'].plot.hist(bins=25, color='seagreen', alpha=0.5, ax=ax1)
ax1.legend(['Clinton','Bush'])

whs.loc[whs['year']==2008,'salary'].plot.hist(bins=25, color='seagreen', alpha=0.5, ax=ax2)
whs.loc[whs['year']==2009,'salary'].plot.hist(bins=25, color='royalblue', alpha=0.5, ax=ax2)
ax2.legend(['Bush','Obama'])

whs.loc[whs['year']==2016,'salary'].plot.hist(bins=25, color='royalblue', alpha=0.5, ax=ax3)
whs.loc[whs['year']==2017,'salary'].plot.hist(bins=25, color='orangered', alpha=0.5, ax=ax3)
plt.xlabel('salary')
ax3.legend(['Obama','Trump'])
plt.show()