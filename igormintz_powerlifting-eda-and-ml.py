# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt
import seaborn as sns
# Any results you write to the current directory are saved as output.
power=pd.read_csv('../input/openpowerlifting.csv')
power.set_index('MeetID', inplace=True)
meets=pd.read_csv('../input/meets.csv')
meets.set_index('MeetID', inplace=True)
#adding dates of meets to powerlifting
alldata=power.join(meets)
#removing unwanted columns
alldata.drop(columns=['Squat4Kg','Bench4Kg','Deadlift4Kg'], inplace=True)
#setting dates
alldata['Date']=pd.to_datetime(alldata['Date'])
#removig rows containing 'DQ' (negative score)
alldata=alldata[alldata['Place']!='DQ']
#removing nan
alldata.dropna(subset=['BestSquatKg','BestBenchKg','BestDeadliftKg','BodyweightKg'], inplace=True)

#devision to men and women by date
Women_by_date=alldata[alldata['Sex']=='F']
Men_by_date=alldata[alldata['Sex']=='M']

#womens' age and weight distribution
ax=sns.jointplot(Women_by_date['Age'],Women_by_date['BodyweightKg'],kind='hex')
ax.fig.suptitle('Women')
plt.savefig('women dist',bbox_inches='tight')
#mens' age and weight distribution
ax=sns.jointplot(Men_by_date['Age'],Men_by_date['BodyweightKg'],kind='hex')
ax.fig.suptitle('Men')
plt.savefig('men dist',bbox_inches='tight')
#weight lifted by age, lift type and sex
import matplotlib.patches as mpatches
fig, ax=plt.subplots(1,2, figsize=(18,11))
ax[0].scatter(Women_by_date['Age'],Women_by_date['TotalKg'], label='total',alpha=0.1, color='grey')
ax[0].scatter(Women_by_date['Age'],Women_by_date['BestBenchKg'], label='bench',alpha=0.1, color='green')
ax[0].scatter(Women_by_date['Age'],Women_by_date['BestDeadliftKg'], label='deadlift',alpha=0.1, color='red')
ax[0].scatter(Women_by_date['Age'],Women_by_date['BestSquatKg'], label='squat',alpha=0.1, color='orange')
ax[0].set_title('Women, n=69439')
ax[0].set_xlabel("Age")
ax[0].set_ylabel("kilograms")
grey_patch= mpatches.Patch(color='grey', label='total')
red_patch = mpatches.Patch(color='red', label='deadlift')
orange_patch = mpatches.Patch(color='orange', label='squat')
green_patch = mpatches.Patch(color='green', label='bench')
ax[0].legend(handles=[grey_patch,red_patch,orange_patch,green_patch],fontsize=20)
ax[1].scatter(Men_by_date['Age'],Men_by_date['TotalKg'],alpha=0.1, color='grey')
ax[1].scatter(Men_by_date['Age'],Men_by_date['BestBenchKg'], label='bench',alpha=0.1, color='green')
ax[1].scatter(Men_by_date['Age'],Men_by_date['BestDeadliftKg'], label='deadlift',alpha=0.1, color='red')
ax[1].scatter(Men_by_date['Age'],Men_by_date['BestSquatKg'], label='squat',alpha=0.1, color='orange')
ax[1].set_title('Men n=216256')
ax[1].set_xlabel("Age")
ax[1].set_ylabel("kilograms")
ax[1].legend(handles=[grey_patch,red_patch,orange_patch,green_patch],fontsize=20)
plt.style.use('fivethirtyeight')
plt.suptitle('weight lifted by age, lift type and sex', fontsize=30, y=1.05)
plt.tight_layout()
plt.savefig('weights_by_year', bbox_inches = 'tight')
#womens' equipement (raw,wraps,single ply, multi ply)
fig, ax=plt.subplots(3,4,figsize=(20,13))

#bench women
ax[0,0].set_title('raw, n=45109')
ax[0,0].scatter(Women_by_date[Women_by_date['Equipment']=='Raw']['BodyweightKg'],Women_by_date[Women_by_date['Equipment']=='Raw']['BestBenchKg'] ,label='raw',alpha=0.3)
ax[0,0].set_ylabel('bench (kg)')
ax[0,1].set_title('wraps, n=10857')
ax[0,1].scatter(Women_by_date[Women_by_date['Equipment']=='Wraps']['BodyweightKg'],Women_by_date[Women_by_date['Equipment']=='Wraps']['BestBenchKg'],label='wraps',alpha=0.3)
ax[0,2].set_title('single-ply, n=12743')
ax[0,2].scatter(Women_by_date[Women_by_date['Equipment']=='Single-ply']['BodyweightKg'],Women_by_date[Women_by_date['Equipment']=='Single-ply']['BestBenchKg'],label='single-ply',alpha=0.3)
ax[0,3].set_title('multi-ply, n=730')
ax[0,3].scatter(Women_by_date[Women_by_date['Equipment']=='Multi-ply']['BodyweightKg'],Women_by_date[Women_by_date['Equipment']=='Multi-ply']['BestBenchKg'],label='multi-ply',alpha=0.3)

#squat women
ax[1,0].scatter(Women_by_date[Women_by_date['Equipment']=='Raw']['BodyweightKg'],Women_by_date[Women_by_date['Equipment']=='Raw']['BestSquatKg'] ,label='raw',alpha=0.3)
ax[1,0].set_ylabel('squat (kg)')
ax[1,1].scatter(Women_by_date[Women_by_date['Equipment']=='Wraps']['BodyweightKg'],Women_by_date[Women_by_date['Equipment']=='Wraps']['BestSquatKg'],label='wraps',alpha=0.3)
ax[1,2].scatter(Women_by_date[Women_by_date['Equipment']=='Single-ply']['BodyweightKg'],Women_by_date[Women_by_date['Equipment']=='Single-ply']['BestSquatKg'],label='single-ply',alpha=0.3)
ax[1,3].scatter(Women_by_date[Women_by_date['Equipment']=='Multi-ply']['BodyweightKg'],Women_by_date[Women_by_date['Equipment']=='Multi-ply']['BestSquatKg'],label='multi-ply',alpha=0.3)

#deadlift women
ax[2,0].scatter(Women_by_date[Women_by_date['Equipment']=='Raw']['BodyweightKg'],Women_by_date[Women_by_date['Equipment']=='Raw']['BestDeadliftKg'] ,label='raw',alpha=0.3)
ax[2,0].set_xlabel('body weight')
ax[2,0].set_ylabel('deadlift (kg)')
ax[2,1].scatter(Women_by_date[Women_by_date['Equipment']=='Wraps']['BodyweightKg'],Women_by_date[Women_by_date['Equipment']=='Wraps']['BestDeadliftKg'],label='wraps',alpha=0.3)
ax[2,1].set_xlabel('body weight')
ax[2,2].scatter(Women_by_date[Women_by_date['Equipment']=='Single-ply']['BodyweightKg'],Women_by_date[Women_by_date['Equipment']=='Single-ply']['BestDeadliftKg'],label='single-ply',alpha=0.3)
ax[2,2].set_xlabel('body weight')
ax[2,3].scatter(Women_by_date[Women_by_date['Equipment']=='Multi-ply']['BodyweightKg'],Women_by_date[Women_by_date['Equipment']=='Multi-ply']['BestDeadliftKg'],label='multi-ply',alpha=0.3)
ax[2,3].set_xlabel('body weight')

plt.style.use('fivethirtyeight')
plt.suptitle("women's lifts by body weight and equipment",fontsize=30, y=1.05)
plt.tight_layout()
plt.savefig('women_equipement', bbox_inches = 'tight')


#mens'equipement (raw,wraps,single ply, multi ply)
fig, ax=plt.subplots(3,4,figsize=(20,13))
#bench men
ax[0,0].scatter(Men_by_date[Men_by_date['Equipment']=='Raw']['BodyweightKg'],Men_by_date[Men_by_date['Equipment']=='Raw']['BestBenchKg'] ,label='raw',alpha=0.3)
ax[0,0].set_ylabel('bench (kg)')
ax[0,0].set_title('raw, n=87375')
ax[0,1].scatter(Men_by_date[Men_by_date['Equipment']=='Wraps']['BodyweightKg'],Men_by_date[Men_by_date['Equipment']=='Wraps']['BestBenchKg'],label='wraps',alpha=0.3)
ax[0,1].set_title('wraps, n=32983')
ax[0,2].scatter(Men_by_date[Men_by_date['Equipment']=='Single-ply']['BodyweightKg'],Men_by_date[Men_by_date['Equipment']=='Single-ply']['BestBenchKg'],label='single-ply',alpha=0.3)
ax[0,2].set_title('single-ply, n=91528')
ax[0,3].scatter(Men_by_date[Men_by_date['Equipment']=='Multi-ply']['BodyweightKg'],Men_by_date[Men_by_date['Equipment']=='Multi-ply']['BestBenchKg'],label='multi-ply',alpha=0.3)
ax[0,3].set_title('multi-ply, n=4370')

#squat men
ax[1,0].scatter(Men_by_date[Men_by_date['Equipment']=='Raw']['BodyweightKg'],Men_by_date[Men_by_date['Equipment']=='Raw']['BestSquatKg'] ,label='raw',alpha=0.3)
ax[1,0].set_ylim([0,600]) #there are 2 negative values
ax[1,0].set_ylabel('squat (kg)')
ax[1,1].scatter(Men_by_date[Men_by_date['Equipment']=='Wraps']['BodyweightKg'],Men_by_date[Men_by_date['Equipment']=='Wraps']['BestSquatKg'],label='wraps',alpha=0.3)
ax[1,2].scatter(Men_by_date[Men_by_date['Equipment']=='Single-ply']['BodyweightKg'],Men_by_date[Men_by_date['Equipment']=='Single-ply']['BestSquatKg'],label='single-ply',alpha=0.3)
ax[1,2].set_ylim([0,600])
ax[1,3].scatter(Men_by_date[Men_by_date['Equipment']=='Multi-ply']['BodyweightKg'],Men_by_date[Men_by_date['Equipment']=='Multi-ply']['BestSquatKg'],label='multi-ply',alpha=0.3)

#deadlift men
ax[2,0].scatter(Men_by_date[Men_by_date['Equipment']=='Raw']['BodyweightKg'],Men_by_date[Men_by_date['Equipment']=='Raw']['BestDeadliftKg'] ,label='raw',alpha=0.3)
ax[2,0].set_ylabel('deadlift (kg)')
ax[2,0].set_xlabel('body weight')
ax[2,1].scatter(Men_by_date[Men_by_date['Equipment']=='Wraps']['BodyweightKg'],Men_by_date[Men_by_date['Equipment']=='Wraps']['BestDeadliftKg'],label='wraps',alpha=0.3)
ax[2,1].set_xlabel('body weight')
ax[2,2].scatter(Men_by_date[Men_by_date['Equipment']=='Single-ply']['BodyweightKg'],Men_by_date[Men_by_date['Equipment']=='Single-ply']['BestDeadliftKg'],label='single-ply',alpha=0.3)
ax[2,2].set_xlabel('body weight')
ax[2,3].scatter(Men_by_date[Men_by_date['Equipment']=='Multi-ply']['BodyweightKg'],Men_by_date[Men_by_date['Equipment']=='Multi-ply']['BestDeadliftKg'],label='multi-ply',alpha=0.3)
ax[2,3].set_xlabel('body weight')

plt.suptitle("men's lifts by body weight and equipment",fontsize=30, y=1.05)
plt.style.use('fivethirtyeight')
plt.tight_layout()
plt.savefig('men_equipement', bbox_inches = 'tight')
#adding a 'year' column
Men_by_date['year'] = pd.DatetimeIndex(Men_by_date['Date']).year
Women_by_date['year'] = pd.DatetimeIndex(Women_by_date['Date']).year
#removing 2018 beacus only has two months
Men_by_date=Men_by_date[Men_by_date['year']!=2018]
Women_by_date=Women_by_date[Women_by_date['year']!=2018]

#grouuping by date and taking the best lifts
menssq=Men_by_date[['BestSquatKg', 'year']].groupby('year').max()
mensbe=Men_by_date[['BestBenchKg', 'year']].groupby('year').max()
mensde=Men_by_date[['BestDeadliftKg', 'year']].groupby('year').max()
mensto=Men_by_date[['TotalKg', 'year']].groupby('year').max()


womenssq=Women_by_date[['BestSquatKg', 'year']].groupby('year').max()
womensbe=Women_by_date[['BestBenchKg', 'year']].groupby('year').max()
womensde=Women_by_date[['BestDeadliftKg', 'year']].groupby('year').max()
womensto=Women_by_date[['TotalKg', 'year']].groupby('year').max()
#mens max plots by year
fig=plt.figure()
plt.figure(figsize=(20,5))
plt.plot(mensbe.index,mensbe['BestBenchKg'], label='Mens bench')
plt.plot(menssq.index,menssq['BestSquatKg'], label='Mens squat')
plt.plot(mensde.index,mensde['BestDeadliftKg'], label='Mens deadlift')
#womens max plots by year
plt.plot(womensbe.index,womensbe['BestBenchKg'], label='Womens bench')
plt.plot(womenssq.index,womenssq['BestSquatKg'], label='Womens squat')
plt.plot(womensde.index,womensde['BestDeadliftKg'], label='Womens deadlift')
plt.ylabel('weight in kg')
plt.legend(fontsize=15)
plt.suptitle("best lifts by year and gender",fontsize=30, y=1.05)
plt.style.use('fivethirtyeight')
plt.tight_layout()
plt.savefig('max by year', bbox_inches = 'tight')
#mens best lifts by equipment and year
raw_men_be=Men_by_date[Men_by_date['Equipment']=='Raw'][['BestBenchKg','year']].groupby('year').max()
raw_men_sq=Men_by_date[Men_by_date['Equipment']=='Raw'][['BestSquatKg','year']].groupby('year').max()
raw_men_de=Men_by_date[Men_by_date['Equipment']=='Raw'][['BestDeadliftKg','year']].groupby('year').max()
wraps_men_be=Men_by_date[Men_by_date['Equipment']=='Wraps'][['BestBenchKg','year']].groupby('year').max()
wraps_men_sq=Men_by_date[Men_by_date['Equipment']=='Wraps'][['BestSquatKg','year']].groupby('year').max()
wraps_men_de=Men_by_date[Men_by_date['Equipment']=='Wraps'][['BestDeadliftKg','year']].groupby('year').max()
single_ply_men_be=Men_by_date[Men_by_date['Equipment']=='Single-ply'][['BestBenchKg','year']].groupby('year').max()
single_ply_men_sq=Men_by_date[Men_by_date['Equipment']=='Single-ply'][['BestSquatKg','year']].groupby('year').max()
single_ply_men_de=Men_by_date[Men_by_date['Equipment']=='Single-ply'][['BestDeadliftKg','year']].groupby('year').max()
multi_ply_men_be=Men_by_date[Men_by_date['Equipment']=='Multi-ply'][['BestBenchKg','year']].groupby('year').max()
multi_ply_men_sq=Men_by_date[Men_by_date['Equipment']=='Multi-ply'][['BestSquatKg','year']].groupby('year').max()
multi_ply_men_de=Men_by_date[Men_by_date['Equipment']=='Multi-ply'][['BestDeadliftKg','year']].groupby('year').max()
#womens best lifts by equipment and year
raw_women_be=Women_by_date[Women_by_date['Equipment']=='Raw'][['BestBenchKg','year']].groupby('year').max()
raw_women_sq=Women_by_date[Women_by_date['Equipment']=='Raw'][['BestSquatKg','year']].groupby('year').max()
raw_women_de=Women_by_date[Women_by_date['Equipment']=='Raw'][['BestDeadliftKg','year']].groupby('year').max()
wraps_women_be=Women_by_date[Women_by_date['Equipment']=='Wraps'][['BestBenchKg','year']].groupby('year').max()
wraps_women_sq=Women_by_date[Women_by_date['Equipment']=='Wraps'][['BestSquatKg','year']].groupby('year').max()
wraps_women_de=Women_by_date[Women_by_date['Equipment']=='Wraps'][['BestDeadliftKg','year']].groupby('year').max()
single_ply_women_be=Women_by_date[Women_by_date['Equipment']=='Single-ply'][['BestBenchKg','year']].groupby('year').max()
single_ply_women_sq=Women_by_date[Women_by_date['Equipment']=='Single-ply'][['BestSquatKg','year']].groupby('year').max()
single_ply_women_de=Women_by_date[Women_by_date['Equipment']=='Single-ply'][['BestDeadliftKg','year']].groupby('year').max()
multi_ply_women_be=Women_by_date[Women_by_date['Equipment']=='Multi-ply'][['BestBenchKg','year']].groupby('year').max()
multi_ply_women_sq=Women_by_date[Women_by_date['Equipment']=='Multi-ply'][['BestSquatKg','year']].groupby('year').max()
multi_ply_women_de=Women_by_date[Women_by_date['Equipment']=='Multi-ply'][['BestDeadliftKg','year']].groupby('year').max()
#mens' equipment trends (top lifts per year)
fig,ax=plt.subplots(3,1, figsize=(18,6),sharex=True)
ax[0].hold(True)

ax[0].plot(raw_men_be.index,raw_men_be['BestBenchKg'],label='raw')
ax[0].plot(wraps_men_be.index,wraps_men_be['BestBenchKg'], label='wraps')
ax[0].plot(single_ply_men_be.index,single_ply_men_be['BestBenchKg'],label='single-ply')
ax[0].plot(multi_ply_men_be.index,multi_ply_men_be['BestBenchKg'], label='multi_ply')
ax[0].set_title('benchpress')
ax[0].set_ylabel('kg')

ax[1].hold(True)
ax[1].plot(raw_men_sq.index,raw_men_sq['BestSquatKg'],label='raw')
ax[1].plot(wraps_men_sq.index,wraps_men_sq['BestSquatKg'], label='wraps')
ax[1].plot(single_ply_men_sq.index,single_ply_men_sq['BestSquatKg'],label='single-ply')
ax[1].plot(multi_ply_men_sq.index,multi_ply_men_sq['BestSquatKg'], label='multi_ply')
ax[1].set_title('squat')
ax[1].set_ylabel('kg')

ax[2].hold(True)
ax[2].plot(raw_men_de.index,raw_men_de['BestDeadliftKg'],label='raw')
ax[2].plot(wraps_men_de.index,wraps_men_de['BestDeadliftKg'], label='wraps')
ax[2].plot(single_ply_men_de.index,single_ply_men_de['BestDeadliftKg'],label='single-ply')
ax[2].plot(multi_ply_men_de.index,multi_ply_men_de['BestDeadliftKg'], label='multi_ply')
ax[2].set_title('deadlift')
ax[2].set_ylabel('kg')

ax[2].legend()
plt.suptitle("mens' equipment trends (top lifts per year)", y=1.05, size=25)
plt.style.use('fivethirtyeight')
plt.tight_layout()
plt.savefig('men equip and year', bbox_inches = 'tight')
#womens' equipment trends (top lifts per year)
fig,ax=plt.subplots(3,1, figsize=(18,6),sharex=True)
ax[0].hold(True)

ax[0].plot(raw_women_be.index,raw_women_be['BestBenchKg'],label='raw')
ax[0].plot(wraps_women_be.index,wraps_women_be['BestBenchKg'], label='wraps')
ax[0].plot(single_ply_women_be.index,single_ply_women_be['BestBenchKg'],label='single-ply')
ax[0].plot(multi_ply_women_be.index,multi_ply_women_be['BestBenchKg'], label='multi_ply')
ax[0].set_title('benchpress')
ax[0].set_ylabel('kg')

ax[1].hold(True)
ax[1].plot(raw_women_sq.index,raw_women_sq['BestSquatKg'],label='raw')
ax[1].plot(wraps_women_sq.index,wraps_women_sq['BestSquatKg'], label='wraps')
ax[1].plot(single_ply_women_sq.index,single_ply_women_sq['BestSquatKg'],label='single-ply')
ax[1].plot(multi_ply_women_sq.index,multi_ply_women_sq['BestSquatKg'], label='multi_ply')
ax[1].set_title('squat')
ax[1].set_ylabel('kg')

ax[2].hold(True)
ax[2].plot(raw_women_de.index,raw_women_de['BestDeadliftKg'],label='raw')
ax[2].plot(wraps_women_de.index,wraps_women_de['BestDeadliftKg'], label='wraps')
ax[2].plot(single_ply_women_de.index,single_ply_women_de['BestDeadliftKg'],label='single-ply')
ax[2].plot(multi_ply_women_de.index,multi_ply_women_de['BestDeadliftKg'], label='multi_ply')
ax[2].set_title('deadlift')
ax[2].set_ylabel('kg')

ax[2].legend()
plt.suptitle("womens' equipment trends (top lifts per year)", y=1.05, size=25)
plt.style.use('fivethirtyeight')
plt.tight_layout()
plt.savefig('women equip and year', bbox_inches = 'tight')
#popularity of equipment over the years
men_raw_by_year=Men_by_date[Men_by_date['Equipment']=='Raw'][['Name','year']].groupby('year').count()
men_wraps_by_year=Men_by_date[Men_by_date['Equipment']=='Wraps'][['Name','year']].groupby('year').count()
men_single_by_year=Men_by_date[Men_by_date['Equipment']=='Single-ply'][['Name','year']].groupby('year').count()
men_multi_by_year=Men_by_date[Men_by_date['Equipment']=='Multi-ply'][['Name','year']].groupby('year').count()
women_raw_by_year=Women_by_date[Women_by_date['Equipment']=='Raw'][['Name','year']].groupby('year').count()
women_wraps_by_year=Women_by_date[Women_by_date['Equipment']=='Wraps'][['Name','year']].groupby('year').count()
women_single_by_year=Women_by_date[Women_by_date['Equipment']=='Single-ply'][['Name','year']].groupby('year').count()
women_multi_by_year=Women_by_date[Women_by_date['Equipment']=='Multi-ply'][['Name','year']].groupby('year').count()
#equipment popularity by users
fig,ax=plt.subplots(2,1, figsize=(18,6),sharex=True)
ax[0].hold(True)

ax[0].plot(men_raw_by_year.index,men_raw_by_year['Name'],label='raw')
ax[0].plot(men_wraps_by_year.index,men_wraps_by_year['Name'], label='wraps')
ax[0].plot(men_single_by_year.index,men_single_by_year['Name'],label='single-ply')
ax[0].plot(men_multi_by_year.index,men_multi_by_year['Name'], label='multi_ply')
ax[0].set_title('men')
ax[1].hold(True)
ax[1].plot(women_raw_by_year.index,women_raw_by_year['Name'],label='raw')
ax[1].plot(women_wraps_by_year.index,women_wraps_by_year['Name'], label='wraps')
ax[1].plot(women_single_by_year.index,women_single_by_year['Name'],label='single-ply')
ax[1].plot(women_multi_by_year.index,women_multi_by_year['Name'], label='multi_ply')
ax[1].set_title('women')


ax[1].legend()
plt.suptitle("equipment popularity by users", y=1.05, size=25)
plt.style.use('fivethirtyeight')
plt.tight_layout()
plt.savefig('equipment popularity by users', bbox_inches = 'tight')
#check fo NaN
sns.heatmap(Men_by_date.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#prediction for raw lifts
data_for_ML_raw=pd.concat([Men_by_date,Women_by_date])
data_for_ML_raw.drop(['Name','Age','WeightClassKg','Division','TotalKg','Place','Wilks', 'MeetPath','Federation','Date','MeetCountry','MeetState','MeetTown','MeetName','year'], axis=1, inplace=True)
data_for_ML_raw.dropna(inplace=True)
data_for_ML_raw.head()
#data_for_ML_raw.drop(data_for_ML_raw[data_for_ML_raw['Equipment']!='Raw'], inplace=True)
#creat dummies for 'Sex', 'Equipment' and 'Division'
data_for_ML_raw = pd.concat([data_for_ML_raw, pd.get_dummies(data_for_ML_raw['Sex'], drop_first=True)], axis=1)
data_for_ML_raw.drop(['Sex'],axis=1, inplace=True)
#training and fitting
X=data_for_ML_raw[['BodyweightKg','BestSquatKg','BestBenchKg','M']]
y=data_for_ML_raw['BestDeadliftKg']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df
#getting predictions and plotting predictions vs reality
predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)
#root mean square error
from sklearn import metrics
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
#prediction formula
print("Function for the first Graph") 
m1 = lm.coef_[0]
m2=lm.coef_[1]
m3=lm.coef_[2]
m4=lm.coef_[3]
b = lm.intercept_
print(' y = {0}* body weight+{1}*squat+{2}*bench + {3}*gender(1 if man, 0 if woman)+{4}'.format(m1,m2,m3,m4,b))
