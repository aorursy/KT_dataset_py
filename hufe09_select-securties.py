# !pip install tushare
import tushare as ts

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.font_manager import FontProperties

import seaborn as sns
def get_font(url, file_path):

    import requests

    r = requests.get(url)

    print("Download: " + file_path)

    with open(file_path, "wb") as f:

        f.write(r.content)

        f.close()

    return file_path





url = "https://github.com/hufe09/pydata_practice/raw/master/fonts/msyh.ttf"

file_path = "msyh.ttf"
pd.set_option('display.max_columns', 100)  # 设置显示数据的最大列数，防止出现省略号…，导致数据显示不全

pd.set_option('expand_frame_repr', False)  # 当列太多时不自动换行

%matplotlib inline

font = FontProperties(fname=get_font(url, file_path),

                      size=12)  # 指定文字,用来正常显示中文标签



sns.set_style('darkgrid')
def get_all_securties():

    df_all = ts.get_industry_classified()

    df_finance = df_all[df_all['c_name'] == '金融行业']

    return df_finance[df_finance['name'].str.contains('证')]





all_securties = get_all_securties()

all_securties
all_securties.shape
all_securties = pd.read_excel('https://github.com/hufe09/pydata_practice/raw/master/select_securties/securties.xlsx', converters={'code': str})

all_securties
def set_start_and_end_dates():

    import datetime

    today = datetime.date.today()

    this_month_start = datetime.datetime(today.year, today.month, 1)

    this_month_of_last_year = (

        this_month_start - datetime.timedelta(days=365)).strftime('%Y-%m-%d')

    last_month_end = (this_month_start -

                      datetime.timedelta(days=1)).strftime('%Y-%m-%d')

    return this_month_of_last_year, last_month_end





start_date = set_start_and_end_dates()[0]

end_date = set_start_and_end_dates()[1]

start_date, end_date
# 选择000776(广发证券)一年的数据

df_776 = ts.get_hist_data("000776", start=start_date, end=end_date)

#　随机打印１０条

df_776.sample(10)
df_776.index
df_776.index = pd.to_datetime(df_776.index)

df_776.index
df_776.info()
df_776.describe()
df_776['p_change'].sum()
df_776['volume'].sum() / 1000000
df_776.plot(figsize=[16, 5])

plt.show()
# !pip install plotly # Plotly是cufflinks依赖包

# !pip install cufflinks
import plotly.offline

import cufflinks as cf

cf.go_offline()

cf.set_config_file(offline=False, world_readable=True, theme='ggplot')
df_776.iplot(title='广发证券')
df_776.iplot(kind='candle', title='广发证券')
# help(df_776.iplot)
df_776_month = df_776.resample('M').mean()

df_776_Q = df_776.resample('Q').mean()

df_776_Y = df_776.resample('A-JUL').mean()
# days_xticks = pd.period_range(this_month_of_last_year, last_month_end, periods=30,freq='30D')

days_xticks = pd.period_range(start_date, end_date, freq='60D')

days_xticks = [str(i) for i in days_xticks]

days_xticks
quarter_xticks = pd.period_range(start_date, end_date, freq="Q")

quarter_xticks = [str(i) for i in quarter_xticks]

quarter_xticks
year_xticks = pd.period_range(start_date, end_date, freq="A-JUL")

year_xticks = [str(i) for i in year_xticks]

year_xticks
fig = plt.figure(figsize=[20, 10])

plt.suptitle('广发证券成交量', FontProperties=font, fontsize=16)

plt.subplot(221)

plt.plot(df_776.volume, '-', label='Days/天')

plt.xticks(days_xticks, rotation=-45)

plt.ylabel('成交量', FontProperties=font)

plt.legend(prop=font, loc="best")

plt.subplot(222)

plt.plot(df_776_month.volume, '-', label='Months/月')

plt.xticks(rotation=-45)

plt.ylabel('成交量', FontProperties=font)

plt.legend(prop=font, loc="best")

plt.subplot(223)

plt.plot(df_776_Q.volume, '-', label='Quarters/季度')

plt.xticks(rotation=-45)

plt.ylabel('成交量', FontProperties=font)

plt.legend(prop=font, loc="best")

plt.subplot(224)

plt.plot(df_776_Y.volume, '-', label='Years/年')

plt.xticks(year_xticks, rotation=-45)

plt.ylabel('成交量', FontProperties=font)

plt.legend(prop=font, loc="best")

plt.show()
df_776.boxplot(figsize=[16, 5])

plt.ylabel('成交量', FontProperties=font)

plt.show()
date_Ym = sorted(

    set(

        df_776.index.map(lambda x: str(x.year) + "-" + str(x.month).rjust(

            2, '0'))))

print(date_Ym)
datas_py = []

for t in date_Ym:

    month = []

    for i in df_776['volume'].index:

        if i.strftime('%Y-%m') == t:

            # 将单位转换为百万

            month.append(df_776['volume'][i] / 1000000)

    datas_py.append(month)
sum([len(i) for i in datas_py]), df_776.shape
months_of_past_year = pd.period_range(start_date, end_date, freq="M")

months_of_past_year = list(months_of_past_year.strftime('%Y-%m'))

months_of_past_year
datas_pd = []

for month in months_of_past_year:

    # 将单位转换为百万

    datas_pd.append(df_776[month]['volume'].values / 1000000)

datas_pd[:2]
df_776.shape, sum([len(i) for i in datas_py]), sum([len(i) for i in datas_pd])
df_datas_pd = pd.DataFrame(datas_pd, index=pd.to_datetime(date_Ym))

df_datas_pd.head()
df_datas_pd.boxplot(figsize=[16, 5])

plt.ylabel('涨幅', FontProperties=font)

plt.show()
days_of_past_year = pd.period_range(start_date, end_date)

days_of_past_year
print(df_776.keys() == df_776.columns)

df_776.keys()
zeros_arr = np.zeros((len(days_of_past_year)))

zeros_arr
df_776_1 = pd.DataFrame(zeros_arr, index=days_of_past_year)

df_776_1.index = df_776_1.index.to_timestamp()

df_776_1.sample(10)
df_776_1.index, df_776.index
df_merge = df_776_1.join(df_776, how='outer')

df_merge.head()
df_merge = pd.concat([df_776_1, df_776], axis=1)

df_merge.head()
df_merge = df_776_1.combine_first(df_776)

df_merge.head()
df_merge = pd.merge(df_776_1,

                    df_776,

                    left_index=True,

                    right_index=True,

                    how='outer')

df_merge.head()
df_merge['month'] = df_merge.index.strftime('%Y-%m')

df_merge['days'] = df_merge.index.strftime('%Y-%m-%d')

df_merge.drop([0], axis='columns', inplace=True)

df_merge.sample(10)
df_merge.isnull().sum()
df_merge.notnull().head()
# 平均值代替NaN

for column in df_776.columns:

    for month in months_of_past_year:

        df_merge[column][month].fillna(df_merge[column][month].mean(),

                                       inplace=True)
def month_column(datas, column, dates):

    column_arr = []

    for date in dates:

        column_arr.append(datas[date][column].values)

    df_column = pd.DataFrame(column_arr, index=dates)

    return df_column
df_volume_776 = month_column(df_merge, 'volume', months_of_past_year)

df_volume_776 = df_volume_776 / 1000000  # 将单位转换为百万

df_volume_776
df_p_change_776 = month_column(df_merge, 'p_change', months_of_past_year)

df_p_change_776
#coding = utf-8

def days_of_month(month, start=1):

    from datetime import date, timedelta

    import pandas as pd

    CURRENTMONTH = month

    year = CURRENTMONTH[0:4]

    month = CURRENTMONTH[5:7]

    traverseDay = date(int(year), int(month), start)

    intMonth = int(CURRENTMONTH[5:7])

    days = []

    while True:

        if intMonth == traverseDay.month:

            days.append(pd.to_datetime(traverseDay.strftime('%Y-%m-%d')))

            traverseDay = traverseDay + timedelta(days=1)

        else:

            break

    return days





days_of_month('2019-02')
def clean_days_volume(days):

    days_values = []

    for i in days:

        t = i.strftime('%Y-%m-%d')

        m = i.strftime('%Y-%m')

        value = (df_776[t]['volume'].values[0] /

                 1000000) if df_776[t]['volume'].values.size > 0 else (

                     df_776[m]['volume'].mean() / 1000000)

        days_values.append(value)

    return days_values
volume_776_1 = []

for t in months_of_past_year:

    days = days_of_month(t)

    volume_776_1.append(clean_days_volume(days))
df_volume_776_1 = pd.DataFrame(volume_776_1, index=months_of_past_year)

# df_volume_776_1.columns = months_of_past_year

df_volume_776_1
df_volume_776 == df_volume_776_1
df_volume_776.loc['2018-07'][0], df_volume_776_1.loc['2018-07'][0]
def deep_clean(data_frame, start_date, end_date):



    days_of_start_to_end = pd.period_range(start_date, end_date)

    months_of_start_to_end = pd.period_range(start_date, end_date, freq="M")

    months_of_start_to_end = list(months_of_start_to_end.strftime('%Y-%m'))



    zeros_arr = np.zeros((len(days_of_start_to_end)))



    data_frame_1 = pd.DataFrame(zeros_arr, index=days_of_start_to_end)

    data_frame_1.index = data_frame_1.index.to_timestamp()



    df_merge = pd.merge(data_frame_1,

                        data_frame,

                        left_index=True,

                        right_index=True,

                        how='outer')



    df_merge['month'] = df_merge.index.strftime('%Y-%m')

    df_merge['days'] = df_merge.index.strftime('%Y-%m-%d')

    df_merge.drop([0], axis='columns', inplace=True)



    # 平均值代替NaN

    for column in data_frame.columns:

        for month in months_of_start_to_end:

            df_merge[column][month].fillna(df_merge[column][month].mean(),

                                           inplace=True)



    return df_merge





def month_column(datas, column, dates):

    column_arr = []

    for date in dates:

        column_arr.append(datas[date][column].values)

    df_column = pd.DataFrame(column_arr, index=dates)

    return df_column
df_volume_776.index, df_volume_776.keys()
df_776_clean = deep_clean(df_776, start_date, end_date)
df_volume_776 = month_column(df_776_clean, 'volume',

                             months_of_past_year) / 1000000

fig, ax = plt.subplots(figsize=(20, 7))

plt.boxplot(df_volume_776)

ax.set_title('广发过去一年每月成交量(Matplotlib 箱线图)', FontProperties=font, fontsize=16)

ax.set_ylabel('成交量(百万)', FontProperties=font)

ax.set_xticklabels(labels=months_of_past_year, rotation=-45)

plt.show()
df_p_change_776 = month_column(df_776_clean, 'p_change', months_of_past_year)

fig, ax = plt.subplots(figsize=(20, 6))

sns.boxplot(data=df_p_change_776.T)

ax.set_xticklabels(months_of_past_year, rotation=-45)

ax.set_title('广发过去一年每月涨幅(Seaborn 箱线图)', FontProperties=font, fontsize=16)

ax.set_ylabel('涨幅', FontProperties=font)

plt.show()
def get_securties_detail(all_securties):

    datas = {}

    for i in all_securties.index:

        ts_data = ts.get_hist_data(all_securties.loc[i]['code'],

                                   start=start_date,

                                   end=end_date)

        if ts_data is not None:

            datas[all_securties.loc[i]['name']] = deep_clean(

                ts_data, start_date, end_date)

    return datas





securties_datas = get_securties_detail(all_securties)
list(securties_datas.keys())[0], securties_datas[list(

    securties_datas.keys())[0]][:5]
len(securties_datas)
def securties_volume(datas):

    securties_volume_dict = {}

    for i in datas:

        if datas[i] is not None:

            securties_volume_dict[i] = datas[i]['volume'] / 1000000

    return securties_volume_dict





securties_volume_dict = securties_volume(securties_datas)



list(securties_volume_dict.keys())[0], securties_volume_dict[list(

    securties_volume_dict.keys())[0]][:5]
df_volume = pd.DataFrame(securties_volume_dict)

df_volume.sample(10)
df_volume.index
df_volume.describe()
df_volume.info()
print(df_volume.isnull().sum().sum())

if df_volume.isnull().sum().sum() > 0:

    df_volume.fillna(0, inplace=True)
df_volume.sample(10)
df_volume.iplot(kind='box', title='各支股票成交量')
fig = plt.figure(figsize=[20, 6])

ax = fig.add_subplot(1, 1, 1)

ax.boxplot(df_volume.T)

ax.set_title('各支股票成交量', FontProperties=font, fontsize=16)

ax.set_xticklabels(df_volume.columns, FontProperties=font, rotation=-45)

ax.set_ylabel('成交量(百万)', FontProperties=font)

plt.show()
total_volume = df_volume.sum()

# 将各支股票累计成交量按数量大小排序

total_volume_sort_desc = total_volume.sort_values(ascending=False)

total_volume_sort_desc
fig = plt.figure(figsize=[20, 6])

ax = fig.add_subplot(1, 1, 1)

ax.plot(total_volume_sort_desc)

plt.plot(total_volume_sort_desc.index,

         total_volume_sort_desc.values,

         linewidth=1,

         color='b',

         marker='o',

         markerfacecolor='blue',

         markersize=5)

ax.set_title('各支股票累计成交量', FontProperties=font, fontsize=16)

ax.set_xticklabels(total_volume_sort_desc.index,

                   FontProperties=font,

                   rotation=-45)

ax.set_ylabel('成交量(百万)', FontProperties=font)

# 设置数字标签

for a, b in zip(total_volume_sort_desc.index, total_volume_sort_desc.values):

    plt.text(a,

             b,

             np.around(b, decimals=2),

             ha='center',

             va='bottom',

             fontsize=13)

plt.show()
def securties_p_change(datas):

    securties_p_change_dict = {}

    for i in datas:

        if datas[i] is not None:

            securties_p_change_dict[i] = datas[i]['p_change']

    return securties_p_change_dict





securties_p_change_dict = securties_p_change(securties_datas)



list(securties_p_change_dict.keys())[0], securties_p_change_dict[list(

    securties_p_change_dict.keys())[0]][:5]
df_p_change = pd.DataFrame(securties_p_change_dict)

df_p_change.sample(10)
df_p_change.describe()
df_p_change.info()
print(df_p_change.isnull().sum().sum())

if df_p_change.isnull().sum().sum() > 0:

    df_p_change.fillna(0, inplace=True)
df_p_change.iplot(kind='box', title='各支股票涨幅箱线图')
fig = plt.figure(figsize=[20, 6])

ax = fig.add_subplot(1, 1, 1)

ax.boxplot(df_p_change.T)

ax.set_title('各支股票涨幅箱线图', FontProperties=font, fontsize=16)

ax.set_xticklabels(total_volume.index, FontProperties=font, rotation=-45)

y_ticks = [i for i in range(-10, 20, 2)]

# ax.set_yticks(y_ticks)

plt.show()
total_p_change = df_p_change.sum()

# 将各支股票累计成交量按数量大小排序

total_p_change_sort_desc = total_p_change.sort_values(ascending=False)

total_p_change_sort_desc
fig = plt.figure(figsize=[20, 6])

ax = fig.add_subplot(1, 1, 1)

ax.plot(total_p_change_sort_desc)

plt.plot(total_p_change_sort_desc,

         linewidth=1,

         color='b',

         marker='o',

         markerfacecolor='blue',

         markersize=5)

ax.set_title('各支股票累计涨幅', FontProperties=font, fontsize=16)

ax.set_xticklabels(total_p_change_sort_desc.index,

                   FontProperties=font,

                   rotation=-45)

ax.set_ylabel('涨幅', FontProperties=font)

# 设置数字标签

for a, b in zip(total_p_change_sort_desc.index,

                total_p_change_sort_desc.values):

    plt.text(a,

             b,

             np.around(b, decimals=2),

             ha='center',

             va='bottom',

             fontsize=13)

plt.show()
width = 0.25  # 条柱的宽度

ind = np.arange(len(total_p_change))  # 组的 x 坐标位置

b = ind + width  # 中间条柱的x坐标

c = b + width  # 第三个条柱的x坐标



plt.figure(figsize=[20, 6])

plt.bar(ind, total_volume, width, color='r', alpha=.5, label='成交量(百万)')

plt.bar(b, total_p_change, width, color='b', alpha=.5, label='涨幅')

plt.bar(c,

        total_p_change / total_volume * 50,

        width,

        color='y',

        alpha=.5,

        label='涨幅与成交额相关性')

labels = list(securties_datas.keys())  # x 坐标刻度标签

locations = ind + width  # x 坐标刻度位置

plt.xticks(locations, labels, FontProperties=font, rotation=-45)

plt.title('各支股票累计成交量和累计涨幅', FontProperties=font, fontsize=16)

plt.ylabel('成交量(百万)', FontProperties=font)

plt.legend(prop=font, loc='best')

plt.show()
df_totals = pd.DataFrame(

    [total_volume, total_p_change, total_p_change / total_volume * 50],

    index=['累计成交量(百万)', '累计涨幅', '涨幅与成交额相关性']).T
plot = df_totals.plot.bar(figsize=[20, 5])

plot.set_title('各支股票累计成交量和累计涨幅', FontProperties=font, fontsize=16)

plot.set_ylabel('成交量(百万)', FontProperties=font)

plot.set_xticklabels(df_totals.index, FontProperties=font, rotation=-45)

plot.legend(df_totals.columns, prop=font)

plt.show()
df_p_change_month = df_volume.resample('M').sum().to_period('M')

df_p_change_month
df_p_change_month.iplot()
df_p_change_month.index = df_p_change_month.index.to_timestamp()

df_p_change_month.index
df_p_change_month.iplot(kind='histogram',

                        histnorm='probability density',

                        title='所有股票涨幅密度分布(月线)',

                        xTitle='波动百分比',

                        yTitle='Probability Density')
df_p_change_week = df_volume.resample('W').sum().to_period('W')

df_p_change_week.index = df_p_change_week.index.to_timestamp()

df_p_change_week.head()
df_p_change_week.iplot()
df_p_change_week.iplot(kind='histogram',

                       histnorm='probability density',

                       title='所有股票涨幅密度分布(周线)',

                       xTitle='波动百分比',

                       yTitle='Probability Density')
total_volume_top20 = list(total_volume_sort_desc.index)[:20]

total_volume_top20
total_p_change_top20 = list(total_p_change_sort_desc.index)[:20]

total_p_change_top20
golden_share = list(

    set(total_volume_top20).intersection(set(total_p_change_top20)))

golden_share
df_p_change_month[golden_share].iplot(kind='histogram',

                                      histnorm='probability density',

                                      title='“黄金股”涨幅密度分布(月线)',

                                      xTitle='波动百分比',

                                      yTitle='Probability Density')
plot = df_p_change_month[golden_share].plot(

    kind='kde', figsize=[15, 4])  # xticks=np.linspace(-100, 100, 10)

plot.set_title('“黄金股”涨幅波动(月线)', FontProperties=font, fontsize=16)

plot.legend(golden_share, prop=font)

plot.set_xlabel('波动百分比', FontProperties=font)

plt.show()
df_p_change_week[golden_share].iplot(kind='histogram',

                                     histnorm='probability density',

                                     title='“黄金股”涨幅密度分布(周线)',

                                     xTitle='波动百分比',

                                     yTitle='Probability Density')
plot = df_p_change_week[golden_share].plot(

    kind='kde',

    figsize=[15, 4],

    #                      xticks=np.linspace(-20, 30, 10)

)

plot.set_title('“黄金股”涨幅波动(周线)', FontProperties=font, fontsize=16)

plot.legend(golden_share, prop=font)

plot.set_xlabel('波动百分比', FontProperties=font)

plt.show()
df_p_change[golden_share].iplot(kind='box')