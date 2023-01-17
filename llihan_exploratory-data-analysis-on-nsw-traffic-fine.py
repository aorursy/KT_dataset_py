# invite people for the science party

import pandas as pd

import numpy as np

import seaborn as sns

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from matplotlib.dates import MonthLocator, DateFormatter

from scipy import stats

import matplotlib.pyplot as plt

from wordcloud import WordCloud
pd.set_option('max_column', 100)

pd.set_option('max_row', 250)

pd.set_option('display.float_format', lambda x: '%.3f' % x)
df = pd.read_csv(

    '../input/penalty_data_set_2.csv', 

    dtype={

        'CAMERA_IND': 'category',

        'SCHOOL_ZONE_IND': 'category',

        'SPEED_IND': 'category',

        'POINT_TO_POINT_IND': 'category',

        'RED_LIGHT_CAMERA_IND': 'category',

        'SPEED_CAMERA_IND': 'category',

        'SEATBELT_IND': 'category',

        'MOBILE_PHONE_IND': 'category',

        'PARKING_IND': 'category',

        'CINS_IND': 'category',

        'FOOD_IND': 'category',

        'BICYCLE_TOY_ETC_IND': 'category',

    }, parse_dates=['OFFENCE_MONTH'], date_parser=lambda x: pd.datetime.strptime(x, '%d/%m/%Y'))
df.columns
df.head()
print('The dataset covers the records from {0} to {1}'.format(

    df['OFFENCE_MONTH'].min(), df['OFFENCE_MONTH'].max()

))
print('There are total {0} unique offence codes in the dataset'.format(len(df['OFFENCE_CODE'].unique())))
fig, ax = plt.subplots(1, figsize=(22, 6), )

monthly_fine_dist = df.groupby('OFFENCE_MONTH').agg({'TOTAL_VALUE': 'count', }).reset_index()

plt.plot('OFFENCE_MONTH', 'TOTAL_VALUE', data=monthly_fine_dist, linewidth=3)

months = MonthLocator(range(1, 13), bymonthday=1, interval=3)

monthsFmt = DateFormatter("%b '%y")

ax.xaxis.set_major_locator(months)

ax.xaxis.set_major_formatter(monthsFmt)

_ = plt.xticks(rotation=45)

_ = plt.tick_params(axis='both', which='major', labelsize=16)

_ = plt.title('Monthly number of fines issued', fontweight='bold', size=21)
offence_code_lookup = df[['OFFENCE_CODE', 'OFFENCE_DESC']].drop_duplicates()

most_common_fines = df.groupby('OFFENCE_CODE').agg({'OFFENCE_MONTH': 'count'}).sort_values(by='OFFENCE_MONTH', ascending=False)[:10].reset_index()

most_common_fines = most_common_fines.merge(offence_code_lookup, how='left' )

most_common_fines
wordcloud = WordCloud(max_font_size=60, width=800, height=400).generate(' '.join(df['OFFENCE_DESC'].tolist()))

plt.figure(figsize=(20,10))

plt.imshow(wordcloud, interpolation='bilinear')

_ = plt.axis("off")
fine_category_fields = [

    'SCHOOL_ZONE_IND',

    'SPEED_IND',

    'POINT_TO_POINT_IND',

    'RED_LIGHT_CAMERA_IND',

    'SPEED_CAMERA_IND',

    'SEATBELT_IND',

    'MOBILE_PHONE_IND',

    'PARKING_IND',

    'CINS_IND',

    'FOOD_IND',

    'BICYCLE_TOY_ETC_IND'

]

_offence_trend = {}

for i, _field in enumerate(fine_category_fields):

    _offence_trend[_field] = df[df[_field] == 'Y'].groupby('OFFENCE_MONTH').agg({'FACE_VALUE': 'count'}).reset_index().sort_values('OFFENCE_MONTH').iloc[:,-1].tolist()



df_trend = pd.DataFrame(_offence_trend)

df_trend['months'] = df['OFFENCE_MONTH'].map(lambda x: x.strftime("%b '%y")).unique()





fig, ax = plt.subplots(1, figsize=(20, 12))

for column in df_trend.drop('months', axis=1):

    plt.plot(df_trend[column], marker='',  linewidth=2, label=column)





plt.legend(loc=1, ncol=1)

plt.title("Number of offences for each category", fontsize=18, fontweight='bold')

plt.xlabel("Months", size=14)

plt.ylabel("Number of fines", size=14)

_ = ax.set_xticklabels(df_trend['months'])
fig, ax = plt.subplots(1, figsize=(22, 6), )

_speed_bands = df[df['SPEED_BAND'].notnull()]['SPEED_BAND']

sns.countplot(_speed_bands)

ax.set_title('Different speeding categories', size=18, fontweight='bold')

plt.xlabel("Speeding subcategories", size=14)

plt.ylabel("Number of fines", size=14)
fine_cat_to_id = {}

for i, _field in enumerate(fine_category_fields):

    fine_cat_to_id[i] = _field

    df[_field + '_CODES'] = df[_field].cat.codes



fine_category_code_fields = list(map(lambda x: x+ '_CODES', fine_category_fields))

df['FINE_CATEGORY_ID'] = df[fine_category_code_fields].apply(lambda x: x.argmax(), raw=True, axis=1)

df['FINE_CATEGORY_TEXT'] = df['FINE_CATEGORY_ID'].apply(lambda x: fine_cat_to_id[x])
fig, ax = plt.subplots(1, figsize=(12, 5))

sns.boxplot(x="FINE_CATEGORY_ID", y="FACE_VALUE", data=df, ax=ax, showfliers=False)

_ = ax.set_xticklabels(fine_category_fields, rotation=30)