import sys

import warnings



if not sys.warnoptions:

    warnings.simplefilter("ignore")



import fbprophet

import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split, KFold

import seaborn as sns
train_df = pd.read_csv('/kaggle/input/hse-pml-2/train_resort.csv')

test_df = pd.read_csv('/kaggle/input/hse-pml-2/test_resort.csv')
train_df.rename(columns={'amount_spent_per_room_night_scaled' : 'target'}, inplace=True)
train_df.head()
desc_df = pd.read_excel('/kaggle/input/hse-pml-2/column_names.xlsx')
desc_df
unique_ids = ['reservation_id']

dates = ['booking_date', 'checkin_date', 'checkout_date']

num_fts = ['numberofadults', 'numberofchildren', 'total_pax', 'roomnights']

cat_fts = list(set(train_df.columns) - set(['target']) - set(unique_ids) - set(dates) - set(num_fts))

# len(unique_ids) + len(dates) + len(num_fts) + len(cat_fts) + 1 == len(train_df.columns)
train_df.isnull().sum()[train_df.isnull().sum()>0]
print('В season_holidayed_code пропущено {0:.4f} % значений'.format(

    train_df.isnull().sum()['season_holidayed_code'] / train_df.shape[0]*100))

print('В state_code_residence пропущено {0:.4f} % значений'.format(

    train_df.isnull().sum()['state_code_residence'] / train_df.shape[0]*100))
np.sort(pd.unique(train_df.state_code_residence)) #нет номера 17
test_df.isnull().sum()[test_df.isnull().sum()>0]
np.sort(pd.unique(test_df.state_code_residence)) #нет номера 17
train_df.state_code_residence.fillna(17, inplace=True)

test_df.state_code_residence.fillna(17, inplace=True)
# переведем дату в другой формат



def prepare_date(df):

    df['booking_date'] = pd.to_datetime(df['booking_date'])

    df['checkin_date'] = pd.to_datetime(df['checkin_date'])

    df['checkout_date'] = pd.to_datetime(df['checkout_date'])

    

prepare_date(train_df)

prepare_date(test_df)
plt.figure(figsize=(8,6))

plt.style.use('bmh')

_ = sns.distplot(train_df.target, kde=False)
pd.unique(train_df.season_holidayed_code), pd.unique(test_df.season_holidayed_code) 
train_df.season_holidayed_code.fillna(5, inplace=True)
def sh_code_analysis(date_col, df, codes, is_train=False):

    for code in codes:

        plt.figure(figsize = (16, 8))

        plot = sns.countplot(x= date_col , data = df[df.season_holidayed_code == code])

        unique_date_count = df[df.season_holidayed_code == code][date_col].nunique()

        freq = unique_date_count//80

        if freq ==0:

            freq = 1

        plot.set_xticklabels(labels= [x.get_text()[:10] for x in plot.get_xticklabels()][::freq], 

                         rotation = 90, fontsize=10)

        _ = plot.set(xticks = range(0,len(plot.get_xticklabels()),freq))

        plt.title('Code = {}'.format(code), fontsize = 14)

        plt.xlabel(date_col, fontsize = 12)

        plt.ylabel('count', fontsize = 12)

    

    plt.figure(figsize = (8,6))

    sns.countplot(x = 'season_holidayed_code', data = df)

    

    

    

    if is_train == True:

        

        for c in codes:

            plt.figure(figsize = (8,6))

            sns.distplot(df[df.season_holidayed_code == c]['target'], kde=False, label=str(c))

            plt.legend()

    

        plt.figure(figsize = (8,6))

        sns.boxplot(x = df.season_holidayed_code, y = df.target)

    

    
sh_code_analysis('checkin_date', train_df, [1,2,3,4,5], is_train=True)
test_df.season_holidayed_code.fillna(5, inplace=True)
sh_code_analysis('checkin_date', test_df, [1,2,3,4,5])
train_df.checkin_date.min(), train_df.checkin_date.max()
test_df.checkin_date.min(), test_df.checkin_date.max()
_ = sns.countplot(x = train_df.booking_date.apply(lambda x : x.year))
_ = sns.countplot(x = test_df.booking_date.apply(lambda x : x.year))
# сколько поездок забронированы на даты из прошлого

(test_df.checkin_date < test_df.booking_date).sum(), (test_df.checkout_date < test_df.booking_date).sum()
np.all(test_df[test_df.checkin_date < test_df.booking_date].index == test_df[

    test_df.checkout_date < test_df.booking_date].index)

print('Наблюдения с бронированием в прошлое {}'.format(test_df[test_df.checkin_date 

                                                              < test_df.booking_date].index.values))
(train_df.checkin_date < train_df.booking_date).sum(), (train_df.checkout_date < train_df.booking_date).sum()
((train_df.numberofadults + train_df.numberofchildren == train_df.total_pax).sum()/train_df.shape[0],

(test_df.numberofadults + test_df.numberofchildren == test_df.total_pax).sum()/test_df.shape[0])
fig, axs = plt.subplots(len(num_fts), 2, constrained_layout=True, figsize = (15, 8))

for i, ft in enumerate(num_fts):

    sns.distplot(train_df[ft], kde=False, label = 'train', bins = 10, ax=axs[i][0])

    axs[i][0].legend()

    sns.distplot(test_df[ft], kde=False, label = 'train', bins = 10, ax=axs[i][1], color = 'r')

    axs[i][1].legend()

    
fig, axs = plt.subplots(1, 2, constrained_layout=True, figsize = (10, 8))

sns.distplot(train_df.roomnights, kde=True, label = 'train', ax=axs[0], bins = train_df.roomnights.nunique())

axs[0].set_title('Train', fontsize = 14)

sns.distplot(test_df.roomnights, kde=True, label='test', ax=axs[1], bins=test_df.roomnights.nunique(), color='red')

_ = axs[1].set_title('Test', fontsize = 14)
train_df.roomnights.min(), train_df.roomnights.max()
train_df.roomnights.value_counts().sort_index()
test_df.roomnights.min(), test_df.roomnights.max()
train_df[cat_fts].dtypes
# на тесте больше значений memberid, чем на трейне

train_df.memberid.nunique(), test_df.memberid.nunique()
def te_isin_tr(col):

    return np.all(np.isin(pd.unique(test_df[col]), pd.unique(train_df[col])))
eda_cat_fts = list(set(cat_fts) - set(['memberid']))
#какие вообще значения принимают категориальные признаки

for i in eda_cat_fts:

    print(i, pd.unique(train_df[i]))
for ft in eda_cat_fts:

    print(ft, 'ok?   ', te_isin_tr(ft), ', type:', train_df.dtypes[ft])
train_df.state_code_residence = train_df.state_code_residence.astype(int)

test_df.state_code_residence = test_df.state_code_residence.astype(int)



train_df.season_holidayed_code = train_df.season_holidayed_code.astype(int)

test_df.season_holidayed_code = test_df.season_holidayed_code.astype(int)
from sklearn.preprocessing import LabelEncoder



def eda_le(le_fts, cat_fts):

    eda_cat_fts = cat_fts.copy()

    for ft in le_fts:

        le = LabelEncoder()

        train_df['edaLE_' + ft] = le.fit_transform(train_df[ft])

        test_df['edaLE_' + ft] = le.transform(test_df[ft])

        eda_cat_fts.remove(ft)

        eda_cat_fts.append('edaLE_' + ft)

    

    return eda_cat_fts



eda_cat_fts = eda_le(['member_age_buckets', 'cluster_code'], cat_fts=eda_cat_fts)

# eda_cat_fts.remove('reservationstatusid_code')

eda_cat_fts.remove('resort_id')
def EDAcat(fts):

    fig, axs = plt.subplots(len(fts), 3, constrained_layout=True, figsize = (18, 5*len(fts)))

    for i, ft in enumerate(fts):

        n_unique = train_df[ft].nunique()

        sns.countplot(train_df[ft], ax=axs[i][0])

        axs[i][0].title.set_text('Train')

        axs[i][0].set_xticklabels(labels= [l for l in axs[i][0].get_xticklabels()], 

                                  fontsize=[10 if n_unique<=10 else 8][0])

        if n_unique <=10:

            sns.boxplot(x=train_df[ft], y=train_df.target, ax = axs[i][1])

            axs[i][1].title.set_text('Train')

        else:

            sns.scatterplot(x=train_df[ft], y=train_df.target, ax=axs[i][1])

            axs[i][1].title.set_text('Train')

        sns.countplot(test_df[ft], ax=axs[i][2])

        axs[i][2].set_xticklabels(labels= [l for l in axs[i][2].get_xticklabels()], 

                                  fontsize=[10 if n_unique<=10 else 8][0])

        axs[i][2].title.set_text('Test')    
EDAcat(eda_cat_fts)
_ = sns.scatterplot(y = train_df.state_code_residence, x = train_df.state_code_resort, alpha=0.01)
train_df[['memberid', 'reservation_id', 'resort_id']].head()
x = set(train_df.reservation_id).intersection(train_df.memberid)

print(x)

x = set(train_df.reservation_id).intersection(train_df.resort_id)

print(x)

x = set(train_df.memberid).intersection(train_df.resort_id)

print(x)
def tr8te (fts):

    num_of_intersec = []

    train_share = []

    test_share = []

    index = []

    

    for ft in fts:

        train_set = set(train_df[ft])

        test_set = set(test_df[ft])

        intersec_set = train_set.intersection(test_set)

        

        num_of_intersec.append(len(intersec_set))

        train_share.append(len(intersec_set)/len(train_set))

        test_share.append(len(intersec_set)/len(test_set))

        index.append(ft)

        

    df = pd.DataFrame({'IntersectionNumber' : num_of_intersec, 'TrainShare': train_share, 

                       'TestShare': test_share}, index=index)

    return df

        
tr8te(['memberid', 'reservation_id', 'resort_id'])
def make_date_fts(dates, df):

    for date in dates:

        df['weekday_' + date] = df[date].apply(lambda x: x.day_name())

        df['month_' + date] = df[date].apply(lambda x: x.month_name())

        df['quarter_' + date] = df[date].apply(lambda x: x.quarter)

        df['year_' + date] = df[date].apply(lambda x: x.year)





make_date_fts(dates, train_df)

make_date_fts(dates, test_df)
def plot_dates():

    weekday_fts=[]

    month_fts = []

    quarter_fts = []

    year_fts = []

    

    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July',

              'August', 'September', 'October', 'November', 'December']

   

    

    for col in train_df.columns:

        if 'weekday_' in col:

            weekday_fts.append(col)

        elif 'month_' in col:

            month_fts.append(col)

        elif 'quarter_' in col:

            quarter_fts.append(col)

        elif 'year_' in col:

            year_fts.append(col)



    fig, axs = plt.subplots(2, 1, constrained_layout=True, figsize = (12, 10))

    for ft in weekday_fts:

        mean_df = train_df.groupby([ft])['target'].mean()

        mean_df = mean_df.reindex(weekdays)

        mean_df = mean_df.reset_index()

        sns.lineplot(x=ft, y='target', data=mean_df, sort=False, label=ft, ax=axs[0])

    axs[0].set_xticklabels(weekdays, fontsize=12, rotation=45)

    axs[0].set_title('Weekday', fontsize=14)

    axs[0].set_xlabel('')

    axs[0].set_ylabel('target',fontsize=12)

    

    for ft in month_fts:

        mean_df = train_df.groupby([ft])['target'].mean()

        mean_df = mean_df.reindex(months)

        mean_df = mean_df.reset_index()

        sns.lineplot(x=ft, y='target', data=mean_df, sort=False, label=ft, ax=axs[1])

    axs[1].set_xticklabels(months, fontsize=12, rotation=45)

    axs[1].set_title('Month', fontsize=14)

    axs[1].set_xlabel('')

    axs[1].set_ylabel('target',fontsize=12)

    

    

    

plot_dates()