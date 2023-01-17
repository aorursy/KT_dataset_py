# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



import matplotlib.pyplot as plt

import seaborn as sns
os.chdir('../input/')

os.listdir()
df = pd.read_csv("hotel-booking-demand/hotel_bookings.csv")
df.shape
sum(df.duplicated())
df.drop_duplicates(inplace=True)
pd.DataFrame({'feature':list(df.columns),'Datatype':[df[x].dtype for x in df.columns]})
df.head(10)
import missingno as msno

msno.matrix(df)
df.isna().sum(axis=0)
(df.company.isna().sum())/len(df.company)
df.market_segment.value_counts()
df.distribution_channel.value_counts()
len(df.company)-df.company.isna().sum()
len(df[(df.market_segment=='Corporate') & (df.distribution_channel=='Corporate') & (df.company != 'NaN')])
cleaned_df =df.drop('company',axis=1)
temp = df[df.agent.isna()].loc[:,['market_segment','distribution_channel']]
temp.market_segment.value_counts()
temp.distribution_channel.value_counts()
df.agent.value_counts()
cleaned_df.drop('agent',axis=1,inplace=True)
cleaned_df.dropna(axis=0,how='any',inplace=True)
cleaned_df.shape
cleaned_df.arrival_date_year.value_counts()
city = cleaned_df[cleaned_df.hotel=='City Hotel']

resort = cleaned_df[cleaned_df.hotel== 'Resort Hotel']

f, axes = plt.subplots(1, 3, figsize=(30, 7))

axes[0].hist([city['arrival_date_month'], resort['arrival_date_month']], color=['r','b'], alpha=0.5,bins=12)

axes[1].hist([city['arrival_date_week_number'], resort['arrival_date_week_number']], color=['r','b'], alpha=0.5)

axes[2].hist([city['arrival_date_day_of_month'], resort['arrival_date_day_of_month']], color=['r','b'], alpha=0.5)
sns.countplot(x="arrival_date_day_of_month", hue="is_canceled", data=cleaned_df)
cleaned_df[(cleaned_df.stays_in_week_nights ==0)&(cleaned_df.stays_in_weekend_nights ==0)].is_canceled.hist(bins=[0,0.5,1])
len(cleaned_df[(cleaned_df.stays_in_week_nights ==0)&(cleaned_df.stays_in_weekend_nights ==0)])/len(cleaned_df)
((cleaned_df.is_repeated_guest ==1) == (cleaned_df.previous_bookings_not_canceled >=1)).value_counts()
cleaned_df.reservation_status.value_counts()
eda_df = cleaned_df.drop(['arrival_date_year','arrival_date_month','arrival_date_day_of_month','country','is_repeated_guest','reservation_status','reservation_status_date'],axis=1)
eda_df.info()
eda_df.describe()
eda_df[['children','babies',

'previous_cancellations',

'previous_bookings_not_canceled',

'required_car_parking_spaces']].hist()
corr = eda_df.corr()

ax = sns.heatmap(

    corr, 

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(20, 220, n=200),

    square=True

)

ax.set_xticklabels(

    ax.get_xticklabels(),

    rotation=45,

    horizontalalignment='right'

);
corr['is_canceled'].sort_values(ascending=False)
corr1 = eda_df[eda_df.hotel=='City Hotel'].corr()

ax = sns.heatmap(

    corr1, 

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(20, 220, n=200),

    square=True

)

ax.set_xticklabels(

    ax.get_xticklabels(),

    rotation=45,

    horizontalalignment='right'

);
corr1['is_canceled'].sort_values(ascending=False)
corr2 = eda_df[eda_df.hotel=='Resort Hotel'].corr()

ax = sns.heatmap(

    corr2, 

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(20, 220, n=200),

    square=True

)

ax.set_xticklabels(

    ax.get_xticklabels(),

    rotation=45,

    horizontalalignment='right'

);
corr2['is_canceled'].sort_values(ascending=False)
corr3 = pd.DataFrame({'Whole data':corr['is_canceled'],'CityHotel':corr1['is_canceled'],'ResortHotel':corr2['is_canceled']})
ax = sns.heatmap(

    corr3, 

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(20, 220, n=200),

    square=True

)

ax.set_xticklabels(

    ax.get_xticklabels(),

    rotation=45,

    horizontalalignment='right'

);
cate = []

for i in eda_df.columns:

    cate.append((eda_df[i].dtype == 'object'))

cat_features = eda_df[eda_df.columns[cate]]
cat_features
cat_label = eda_df.is_canceled.map({1:'canceled',0:'not canceled'})

cat_df = pd.concat([cat_label,cat_features],axis=1)
cat_df
import scipy

pvalue={}

#pvalues = pd.DataFrame(data = np.zeros((9,9)),index=list(cat_df.columns),columns=list(cat_df.columns))

#for i in cat_df.columns:

for j in cat_features.columns:

    tab = pd.crosstab(cat_label,cat_features[j], margins = False)

    chi2, p, dof, ex = scipy.stats.chi2_contingency(tab)

    pvalue[j] = p
pvalue