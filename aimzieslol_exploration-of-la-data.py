import pandas as p

import numpy as np

import re



import seaborn as sns

import matplotlib.pyplot as plt



import datashader

import datashader.transfer_functions as dstf



%matplotlib inline
def read_only_col(*colname):

    cols = []

    

    for s in colname:

        cols.append(s)

    

    return p.read_csv('Crime_Data_from_2010_to_Present.csv', usecols=cols, nrows=6000)



def print_basic_stuff(df):

    print(df.describe())

    print('\n# of nulls:\n%s' % str(df.isna().sum()))
df = p.read_csv('Crime_Data_from_2010_to_Present.csv', nrows=1)

df.columns
dr_df = read_only_col('Date Reported')

print_basic_stuff(dr_df)
dr_df['Date Reported'] = p.to_datetime(dr_df['Date Reported'], infer_datetime_format=True)

dr_counts = dr_df.groupby('Date Reported')['Date Reported'].count()

dr_counts = dr_counts.sort_index()
fig = plt.figure(figsize=(20, 5))

plt.subplot(1, 2, 1)

ax = sns.scatterplot(x=dr_counts.index.values, y=dr_counts.values, size=dr_counts.values, hue=dr_counts.values, legend=False)

ax.xaxis.set_major_locator(ticker.MaxNLocator(10))

fig.autofmt_xdate()

plt.subplot(1, 2, 2)

sns.distplot(dr_counts)

plt.show()
do_df = read_only_col('Date Occurred')

print_basic_stuff(do_df)
do_df['Date Occurred'] = p.to_datetime(do_df['Date Occurred'], infer_datetime_format=True)

do_count = do_df.groupby('Date Occurred')['Date Occurred'].count()

do_count = do_count.sort_index()
fig = plt.figure(figsize=(20, 5))

plt.subplot(1, 2, 1)

ax = sns.scatterplot(x=do_count.index.values, y=do_count.values, size=do_count.values, hue=do_count.values, legend=False)

ax.xaxis.set_major_locator(ticker.MaxNLocator(10))

fig.autofmt_xdate()

plt.subplot(1, 2, 2)

sns.distplot(do_count)

plt.show()
time_diff = dr_df['Date Reported'] - do_df['Date Occurred']

time_diff_days = time_diff.astype('<m8[D]')
fig = plt.figure(figsize=(20, 5))

plt.subplot(1, 2, 1)

ax = sns.scatterplot(x=time_diff_days.index.values, y=time_diff_days.values, size=time_diff_days.values, hue=time_diff_days.values)

ax.xaxis.set_major_locator(ticker.MaxNLocator(5))

plt.subplot(1, 2, 2)

sns.distplot(time_diff_days)

plt.show()
to_df = read_only_col('Time Occurred')

print_basic_stuff(to_df)



print('%% of bad Time Occurred entries: %.02f' % \

      ((to_df[to_df['Time Occurred'] < 100].count() / to_df.shape[0]) * 100))
to_df[to_df['Time Occurred'] < 100] = 100

as_times = p.to_datetime(to_df['Time Occurred'], format='%H%M').dt.time
fig = plt.figure(figsize=(10, 5))

sns.scatterplot(y=as_times.index.values, x=as_times.values)

ax.xaxis.set_major_locator(ticker.MaxNLocator(10))

fig.autofmt_xdate()

plt.show()
area_df = read_only_col('Area ID')

print_basic_stuff(area_df)



print('%% of Area ID that have actual numbers: %f' % ((area_df[area_df['Area ID'] > 0].count() / area_df.shape[0]) * 100))
area_name_df = read_only_col('Area Name')

print_basic_stuff(area_name_df)
an_count = area_name_df.groupby('Area Name')['Area Name'].count()



fig = plt.figure(figsize=(20, 5))

plt.subplot(1, 2, 1)

sns.scatterplot(y=an_count.values, x=an_count.index.values, size=an_count.values)

fig.autofmt_xdate()

plt.subplot(1, 2, 2)

sns.distplot(an_count)

plt.show()
rep_dist_df = read_only_col('Reporting District')

print_basic_stuff(rep_dist_df)
fig = plt.figure(figsize=(10, 5))

sns.distplot(rep_dist_df)

# sns.countplot(data=rep_dist_df)

fig.autofmt_xdate()

plt.show()
cc_df = read_only_col('Crime Code')

print_basic_stuff(cc_df)



fig = plt.figure(figsize=(20, 5))

sns.distplot(cc_df)

# sns.scatterplot(data=cc_df)

fig.autofmt_xdate()

plt.show()
ccd_df = read_only_col('Crime Code Description')

print_basic_stuff(ccd_df)



ccd_counts = ccd_df.groupby('Crime Code Description')['Crime Code Description'].count()



fig = plt.figure(figsize=(20, 5))

plt.subplot(1, 2, 1)

sns.distplot(ccd_counts)

plt.subplot(1, 2, 2)

sns.scatterplot(x=ccd_counts.index.values, y=ccd_counts.values, size=ccd_counts.values)

fig.autofmt_xdate()

plt.show()
mmo_df = read_only_col('MO Codes')

print_basic_stuff(mmo_df)



# mmo_df = p.read_csv('Crime_Data_from_2010_to_Present.csv', usecols=['MO Codes'])

mmo_df['MO Codes'].fillna(0, inplace=True)



mmo_df['MO Codes'] = mmo_df['MO Codes'].apply(lambda x: re.sub(r'[^0-9\s]', '0', str(x)))

mmo_df['MO Codes'] = mmo_df['MO Codes'].apply(lambda x: np.array(x.split(' ' )).astype('int').sum())



fig = plt.figure(figsize=(10, 5))

sns.scatterplot(x=mmo_df['MO Codes'].index.values, y=mmo_df['MO Codes'].values)

fig.autofmt_xdate()

plt.show()
mmo_df = mmo_df.drop(mmo_df[mmo_df['MO Codes'] > 15000].index.values)



fig = plt.figure(figsize=(10, 5))

sns.scatterplot(x=mmo_df['MO Codes'].index.values, y=mmo_df['MO Codes'].values)

fig.autofmt_xdate()

plt.show()
va_df = read_only_col('Victim Age')

print_basic_stuff(va_df)
median_age = va_df['Victim Age'].quantile(.50)



zero_va_df = va_df[va_df['Victim Age'] == 0].copy()



zero_va_df['Victim Age'] = median_age



va_df.update(zero_va_df)



fig = plt.figure(figsize=(10, 5))

sns.scatterplot(data=va_df)

fig.autofmt_xdate()

plt.show()
vd_df = read_only_col('Victim Descent')

print_basic_stuff(vd_df)
vd_df.fillna('X', inplace=True)

vd_series = vd_df['Victim Descent']



vd_count = vd_df.groupby('Victim Descent')['Victim Descent'].count()



fig = plt.figure(figsize=(10, 5))

sns.distplot(vd_count)

fig.autofmt_xdate()

plt.show()
prem_df = read_only_col('Premise Code')

print_basic_stuff(prem_df)



prem_count = prem_df.groupby('Premise Code')['Premise Code'].count()



fig = plt.figure(figsize=(10, 5))

sns.distplot(prem_df)

plt.show()
pd_df = read_only_col('Premise Description')

print_basic_stuff(pd_df)



pd_count = pd_df.groupby('Premise Description')['Premise Description'].count()



fig = plt.figure(figsize=(20, 5))

plt.subplot(1, 2, 1)

sns.distplot(pd_count)

plt.subplot(1, 2, 2)

sns.scatterplot(x=pd_count.index.values, y=pd_count.values, size=pd_count.values)

fig.autofmt_xdate()

plt.show()
wuc_df = read_only_col('Weapon Used Code')

print_basic_stuff(wuc_df)
wud_df = read_only_col('Weapon Description')

print_basic_stuff(wud_df)
wuc_df['WeaponUsed'] = [1 if x > 0 else 0 for x in wuc_df['Weapon Used Code'].values]



print('%% of crimes in which weapons were used: %.02f' % (wuc_df['WeaponUsed'].sum() / len(wuc_df['WeaponUsed']) * 100))
sc_df = read_only_col('Status Code')

print_basic_stuff(sc_df)
# sc_count = sc_df.groupby('Status Code')['Status Code'].count()



# fig = plt.figure(figsize=(20, 5))

# plt.subplot(1, 2, 1)

# sns.distplot(sc_count)

# plt.subplot(1, 2, 2)

# sns.scatterplot(x=sc_count.index.values, y=sc_count.values, size=sc_count.values)

# fig.autofmt_xdate()

# plt.show()
sd_df = read_only_col('Status Description')

print_basic_stuff(sd_df)



sd_count = sd_df.groupby('Status Description')['Status Description'].count()



fig = plt.figure(figsize=(20, 5))

plt.subplot(1, 2, 1)

sns.distplot(sd_count)

plt.subplot(1, 2, 2)

sns.scatterplot(x=sd_count.index.values, y=sd_count.values, size=sd_count.values)

fig.autofmt_xdate()

plt.show()
crime_codes_arr = ['Crime Code 1', 'Crime Code 2', 'Crime Code 3', 'Crime Code 4']



cc_df = read_only_col(*crime_codes_arr)

print_basic_stuff(cc_df)



fig = plt.figure(figsize=(20, 5))

sns.scatterplot(data=cc_df)

fig.autofmt_xdate()

plt.show()
a_df = read_only_col('Address')

print_basic_stuff(a_df)
cs_df = read_only_col('Cross Street')

print_basic_stuff(cs_df)
l_df = read_only_col('Location ')

print_basic_stuff(l_df)



l_df['Location '] = l_df['Location '].transform(lambda x: re.sub(r'\((-?\d+\.?\d+),\s+?(-?\d+\.?\d+)\)', r'\1,\2', x))

split_loc = l_df['Location '].str.split(',', expand=True)



l_df['lon'] = split_loc[0]

l_df['lat'] = split_loc[1]



l_df.drop(['Location '], axis=1, inplace=True)



l_df = l_df.astype('float32')



dstf.shade(datashader.Canvas(plot_width=640, plot_height=480).points(l_df, x='lat', y='lon'))