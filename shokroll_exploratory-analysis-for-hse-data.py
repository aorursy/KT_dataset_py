import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame, Series
%matplotlib inline
import calendar

acc=pd.read_csv('../input/IHMStefanini_industrial_safety_and_health_database_with_accidents_description.csv')
acc.head(3)
acc.size
acc.isnull().sum()
del acc['Unnamed: 0']
acc.head(3)
acc.rename(columns={'Data': 'Date', 'Genre': 'Gender', 'Employee or Third Party':'Employee type'}, inplace=True)
acc.Date.max()
acc.Date.min()
acc['Date'] = pd.to_datetime(acc['Date'])
month_order={
    'January':1,
    'February':2,
    'March':3,
    'April':4,
    'May':5,
    'June':6,
    'July':7,
    'August':8,
    'September':9,
    'October':10,
    'November':11,
    'December':12
}
acc.groupby('Date').count()['Local'].plot(figsize=(15,4))
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
def month_n(a):
    b=calendar.month_name[a]
    return b
x=[]
for i in range (0, len(acc.Date)):
                x.append(month_n(acc.Date.loc[i].month))
y=[]
for i in range (0, len(acc.Date)):
                y.append(acc.Date.loc[i].year)
acc['month']=x
acc['year']=y
acc.head(3)
acc_trend=acc.pivot_table(index='month', columns=[ 'year','Accident Level'], aggfunc='count')['Countries']

n=np.nan
acc_trend.replace(n,0,inplace=True)
acc_trend.head(2)
acc_trend[2016]
acc_trend[2016].loc[month_order].plot(kind='bar', figsize=(15,4), width=0.9, cmap='cool', title='2016')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
acc_trend[2017].loc[month_order].plot(kind='bar', figsize=(15,4), width=0.9, cmap='hot', title='2017')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
order={'I':1, 'II':2, 'III':3, 'IV':4, 'V':5}
fig=sns.FacetGrid(acc,aspect=1.2,palette="winter", hue='Gender',col='Industry Sector', legend_out=True)
fig.map(sns.countplot, 'Accident Level', order=order)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

sns.countplot('Employee type',data=acc,palette='cool' )
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
acc['Potential Accident Level'].unique()
order2={'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'VI': 6}
q=Series()
for i in range(0, len(acc)):
    q=q.append(Series(order2[acc.loc[i]['Accident Level']]*order2[acc.loc[i]['Potential Accident Level']]), ignore_index=True)
acc['Accident Impact']=q
acc.head(5)
acc.plot(x='Date', y='Accident Impact', figsize=(15,4), kind='line')
acc.plot(x='Date', y='Accident Impact', figsize=(15,4), kind='line')
plt.text(x='2016-6-15', y=28.5, s='July 2016', color='red', fontsize=12)
plt.vlines(x='2016-7-1', ymin=25, ymax=28, color='red', linestyles=':', linewidth=3)
plt.text(x='2017-3-5', y=28.5, s='March 2017', color='red', fontsize=12)
plt.vlines(x='2017-3-15', ymin=25, ymax=28, color='red', linestyles=':', linewidth=3)

plt.figure(figsize=(15,4))
sns.boxplot(x='month', y='Accident Impact', data=acc, palette='Set3', saturation=1)

wd=[]
for i in range(len(acc)):
    wd.append(acc['Date'].loc[i].weekday())
weekday={'0':'Monday',
        '1':'Tuesday', 
        '2':'Wednesday',
        '3':'Thursday',
        '4':'Friday',
        '5':'Saturday',
        '6':'Sunday'}
wd_order={'Monday':1,
        'Tuesday':2, 
        'Wednesday':3,
        'Thursday':4,
        'Friday':5,
        'Saturday':6,
        'Sunday':7}
wwd=[]
for i in wd:
    wwd.append(weekday[str(i)])
acc['weekday']=wwd
week_d=acc.pivot_table(index='weekday', columns='Industry Sector', aggfunc='count')['Accident Level']
week_d.loc[wd_order].plot(figsize=(10,4), xticks=range(7), cmap='Dark2', kind='line')
plt.ylabel('number of accidents')
acc.groupby('weekday').count()['month'].loc[wd_order].plot(kind='line', figsize=(9,4), xticks=range(0,8), color='#FF6A6A')
sns.factorplot(x='year', y='Accident Impact', data=acc, hue='Industry Sector', aspect=2, size=4)
acc_ind=acc.groupby('Industry Sector').count()['Date']
acc_ind_imp=acc.groupby('Industry Sector')['Accident Impact'].mean()
acc_ind_imp.plot(kind='pie', figsize=(5,5), cmap='Set1', autopct='%.2f', title='Mean Accident Impact')

acc_ind.plot(kind='pie', figsize=(5,5), cmap='Set2', autopct='%.2f', title='Number of Incidents')
acc_cr=acc.pivot_table(index='Critical Risk', columns='Accident Level', aggfunc='count')['month']
acc_cr.replace(n, 0, inplace=True)
acc_cr['total']=acc_cr.sum(axis=1)
acc_cr.style.background_gradient(cmap='Blues')
acc_cr.drop('Others', axis=0, inplace=True)
acc_cr.total.sort_values().plot(kind='barh', figsize=(8,20), xticks=range(0,25), grid=False, width=0.65)
plt.xlabel('total number of accidents')

acc_cr.nlargest(6, 'total').style.background_gradient(cmap='winter')
acc_ind_risk=acc.pivot_table(index='Critical Risk', columns='Industry Sector', aggfunc='count')['Accident Level']
acc_ind_risk.drop('Others', axis=0, inplace=True)
acc_ind_risk.replace(n, 0, inplace=True)
acc_ind_risk['total']=acc_ind_risk.sum(axis=1)
acc_ind_risk
acc_ind_risk.nlargest(6,'total').plot(kind='bar', xticks=range(30), figsize=(15,5), cmap='summer')
plt.xticks(rotation=90)
acc_ind_risk_nt=acc_ind_risk.drop('total', axis=1)
fig=acc_ind_risk_nt.plot(kind='bar', xticks=range(30),yticks=range(0,21), figsize=(15,6), cmap='Paired', width=0.9)
fig.set_facecolor('#2B2B2B')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.text(x=0.9, y=11, s='V', color='#CD661D', fontsize=25)
plt.text(x=3.3, y=16, s='V', color='#ADD8E6', fontsize=25)
plt.text(x=5.4, y=11, s='V', color='#ADD8E6', fontsize=25)
plt.text(x=14.4, y=15, s='V', color='#ADD8E6', fontsize=25)
plt.text(x=18.4, y=17.5, s='V', color='#ADD8E6', fontsize=25)
plt.text(x=21.6, y=10, s='V', color='#FFD39B', fontsize=25)
plt.text(x=28.6, y=9, s='V', color='#FFD39B', fontsize=25)
plt.text(x=30, y=14, s='V', color='#CD661D', fontsize=25)
acc_risk_other=acc.loc[acc['Critical Risk']=='Others']
sns.countplot('Employee type', data=acc_risk_other)
sns.distplot(acc_risk_other['Accident Impact'])
