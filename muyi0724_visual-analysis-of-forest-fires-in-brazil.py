import numpy as np

import pandas as pd

# 采用离线绘图方式

import plotly

import plotly.offline as py

import plotly.graph_objs as go

plotly.offline.init_notebook_mode()
# 读取数据，注意文件编码格式

df = pd.read_csv('../input/forest-fires-in-brazil/amazon.csv', encoding='ISO-8859-1')

df.head(10)
# 检查数据集是否有需要舍弃的数据

print("共有%d条数据" % len(df))

print("空值检查")

print(df.isna().sum())
# 概览所有统计到的州

df.state.unique()
# 概览月份

df.month.unique()
# 把月份转换成英语

month_map = {'Janeiro': 'January', 'Fevereiro': 'February', 'Março': 'March', 'Abril': 'April', 'Maio': 'May',

          'Junho': 'June', 'Julho': 'July', 'Agosto': 'August', 'Setembro': 'September', 'Outubro': 'October',

          'Novembro': 'November', 'Dezembro': 'December'}

df['month'] = df['month'].map(month_map)

df.month.unique()
# 日期列没有意义，丢掉

df.drop(columns = ['date'], axis=1, inplace=True)

df.head(5)
# 统计各年发生的火灾总数

years = list(df.year.unique())

sub_fires_per_year = []

for i in years:

    #数据集中汇报的火灾数有小数，四舍五入取整

    count = df.loc[df['year'] == i].number.sum().round(0)

    sub_fires_per_year.append(count)

fires_per_year_dic = {'year' : years, 'total_fires' : sub_fires_per_year}

print(fires_per_year_dic)
# 绘制散点图

trace = go.Scatter(x = fires_per_year_dic['year'], y = fires_per_year_dic['total_fires'], mode = 'lines+markers')

layout = go.Layout(title='1998-2017年巴西火灾数量统计图', xaxis={'title':'年份'}, yaxis={'title':'数量'})

data = [trace]

fig = go.Figure(data = data,layout = layout)

py.iplot(fig)
# 根据月份添加季节(巴西在南半球和北半球季节相反)

# 9、10、11月为春季，12、1、2月为夏季，3、4、5月为秋季，6、7、8月为冬季

season_map = {'January': 'Summer', 'February': 'Summer', 'March': 'Autumn', 'April': 'Autumn', 'May': 'Autumn',

              'June': 'Winter', 'July': 'Winter', 'August': 'Winter', 'September': 'Spring', 'October': 'Spring',

              'November': 'Spring', 'December': 'Summer'}

df['season'] = df['month'].map(season_map)

df.season.unique()
# 把数据集按照年份和季节进行分割，并求出每年每月的火灾总数

def split_season(df, season):

    sub_fires = []

    for i in years:

        count = df.loc[(df['year'] == i) & (df['season'] == season)].number.sum().round(0)

        sub_fires.append(count)

    return sub_fires

sub_fires_in_spring = split_season(df, 'Spring')

sub_fires_in_summer = split_season(df, 'Summer')

sub_fires_in_autumn = split_season(df, 'Autumn')

sub_fires_in_winter = split_season(df, 'Winter')

print(sub_fires_in_spring)
# 条形图绘制代码

trace_spring = go.Bar(y = years, x = sub_fires_in_spring,

                name = '春季', orientation = 'h', marker = dict(color = '#EF7A82'))

trace_summer = go.Bar(y = years, x = sub_fires_in_summer,

                name = '夏季', orientation = 'h', marker = dict(color = '#AFDD22'))

trace_autumn = go.Bar(y = years, x = sub_fires_in_autumn,

                name = '秋季', orientation = 'h', marker = dict(color = '#FFA631'))

trace_winter = go.Bar(y = years, x = sub_fires_in_winter,

                name = '冬季', orientation = 'h', marker = dict(color = '#44CEF6'))

data = [trace_spring, trace_summer, trace_autumn, trace_winter]

layout = go.Layout(

    title = '1998年-2017年各季节火灾数量及季节分布统计图',

    barmode='stack',

    xaxis={'title':'数量'},yaxis={'title':'年份'}

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig)
# 把数据集按照season分割

fires_per_season = dict(list(df.groupby(df['season'])))

# 单独处理下12月，1月和2月，以便顺序展示月份

fires_per_season_summer1 = fires_per_season['Summer'].loc[fires_per_season['Summer']['month'].isin(['December'])]

fires_per_season_summer2 = fires_per_season['Summer'].loc[fires_per_season['Summer']['month'].isin(['January', 'February'])]

fires_per_season_summer = fires_per_season_summer1.append(fires_per_season_summer2)

#绘制箱图

trace_summer = go.Box(y=fires_per_season_summer.number, x=fires_per_season_summer.month,

                      name='Summer', boxpoints='all', marker = dict(color = '#AFDD22'))

trace_autumn = go.Box(y=fires_per_season['Autumn'].number, x=fires_per_season['Autumn'].month,

                      name='Autumn', boxpoints='all', marker = dict(color = '#FFA631'))

trace_winter = go.Box(y=fires_per_season['Winter'].number, x=fires_per_season['Winter'].month,

                      name='Winter', boxpoints='all', marker = dict(color = '#44CEF6'))

trace_spring = go.Box(y=fires_per_season['Spring'].number, x=fires_per_season['Spring'].month,

                      name='Spring', boxpoints='all', marker = dict(color = '#EF7A82'))

data = [trace_summer, trace_autumn, trace_winter, trace_spring]

layout = go.Layout(

    title = '1998年-2017年各季节火灾数量盒图',

    xaxis={'title':'月份'},yaxis={'title':'数量'}

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig)
# 把数据集按照state进行分割，并求出每个地区的火灾总数

sub_fires_per_state = df['number'].groupby(df['state']).sum().round(0).sort_values(ascending=False)

# 绘制饼图

trace = go.Pie(labels=sub_fires_per_state.index, values=sub_fires_per_state.values, textinfo='label+value')

data = [trace]

layout = go.Layout(title = '1998年-2017年各州火灾数量饼图', height=800)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig)
# 有部分地区火灾数远超过其他地区，经过检查数据集中的地区有重名情况

# 数据集中Mato Grosso, Paraiba, Rio这三个地区有多个相同的名称，手动处理数据集进行标记后再重新进行探索

df2 = pd.read_csv('../input/forest-fires-in-brazil-relabeling/amazon_v2.csv', encoding='ISO-8859-1')

month_map = {'Janeiro': 'January', 'Fevereiro': 'February', 'Março': 'March', 'Abril': 'April', 'Maio': 'May',

          'Junho': 'June', 'Julho': 'July', 'Agosto': 'August', 'Setembro': 'September', 'Outubro': 'October',

          'Novembro': 'November', 'Dezembro': 'December'}

df2['month'] = df2['month'].map(month_map)

df2.drop(columns = ['date'], axis=1, inplace=True)

df2.head(5)
# 把数据集按照新标记的state进行分割，并求出每个地区的火灾总数

sub_fires_per_state = df2['number'].groupby(df2['state']).sum().round(0).sort_values(ascending=False)

# 绘制饼图

trace = go.Pie(labels=sub_fires_per_state.index, values=sub_fires_per_state.values, textinfo='label+value')

data = [trace]

layout = go.Layout(title='1998年-2017年各州火灾数量饼图', height=800)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig)
#根据州拆分数据集

fires_per_state = dict(list(df2.groupby(df2['state'])))

# print(fires_per_state)

states = df2['state'].unique()

years = df2['year'].unique()

fires_per_state_per_year = []

for state in states:

    sub_fires = fires_per_state[state]['number'].groupby(fires_per_state[state]['year']).sum().round(0)

    fires_per_state_per_year.append(list(sub_fires))

# print(fires_per_state_per_year)
# 绘制热图

trace = go.Heatmap(z=fires_per_state_per_year, x=years, y=states, colorscale='reds')

data = [trace]

layout = go.Layout(title = '各州每年火灾数量热图', 

                   xaxis={'title':'年份', 'nticks':20},yaxis={'title':'州', 'nticks':27},

                   margin={'l':100})

fig = go.Figure(data=data, layout=layout)

py.iplot(fig)
months = df2['month'].unique()

# 月份如果为字符串的话将会按照字典顺序进行统计，所以手动转换为数值类型月份

num_month_map =  {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5,'June': 6,

                  'July': 7, 'August': 8, 'September': 9, 'October': 10,'November': 11, 'December': 12}

df2['month_num'] = df2['month'].map(num_month_map)

fires_per_state = dict(list(df2.groupby(df2['state'])))

# 按照月份分割数据集

fires_per_state_per_month = []

for state in states:

    sub_fires = fires_per_state[state]['number'].groupby(fires_per_state[state]['month_num']).sum().round(0)

    fires_per_state_per_month.append(list(sub_fires))

# print(fires_per_state_per_month)
# 绘制热图

trace = go.Heatmap(z=fires_per_state_per_month, x=months, y=states, colorscale='reds')

data = [trace]

layout = go.Layout(title = '各州每月火灾数量热图', 

                   xaxis={'title':'月份', 'nticks':20},yaxis={'title':'州', 'nticks':27},

                   margin={'l':100})

fig = go.Figure(data=data, layout=layout)

py.iplot(fig)