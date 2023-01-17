import numpy as np
import pandas as pd
%matplotlib inline

from bokeh.io import output_notebook, show
from bokeh.models import ColumnDataSource, HoverTool, Range1d, Span
from bokeh.plotting import figure
from bokeh.palettes import Set1_9
from bokeh.transform import factor_cmap

import datetime

output_notebook()
train = pd.read_pickle('../input/gstore-revenue-data-preprocessing/train.pkl')
test = pd.read_pickle('../input/gstore-revenue-data-preprocessing/test.pkl')
train['visitStartTime'].describe()
test['visitStartTime'].describe()
test_start_time = np.min(test['visitStartTime'])
test_start_time
train_ts = train.set_index('visitStartTime')
test_ts = test.set_index('visitStartTime')
train_daily_sess_count = pd.DataFrame({'count': train_ts.groupby(pd.Grouper(freq='D')).size(), 'set': 'train'})
test_daily_sess_count = pd.DataFrame({'count': test_ts.groupby(pd.Grouper(freq='D')).size(), 'set': 'test'})
daily_sess_count = pd.concat([train_daily_sess_count, test_daily_sess_count], axis=0)
p = figure(x_axis_type='datetime', width=700, height=400, title='daily session count over time')
p.line(
    source=daily_sess_count[daily_sess_count['set'] == 'train'],
    x='visitStartTime',
    y='count',
    color=Set1_9[0])
p.line(
    source=daily_sess_count[daily_sess_count['set'] == 'test'],
    x='visitStartTime',
    y='count',
    color=Set1_9[1])
p.legend.location = 'top_left'
test_start = Span(location=test_start_time.timestamp() * 1000, dimension='height', line_dash='dashed')
p.add_layout(test_start)
show(p)
train_daily_pageviews = pd.DataFrame({
    'count':
    train_ts.groupby(pd.Grouper(freq='D'))['totals.pageviews'].sum(),
    'set':
    'train'
})
test_daily_pageviews = pd.DataFrame({
    'count':
    test_ts.groupby(pd.Grouper(freq='D'))['totals.pageviews'].sum(),
    'set':
    'test'
})
daily_pageviews = pd.concat(
    [train_daily_pageviews, test_daily_pageviews], axis=0)
train_daily_hit_count = pd.DataFrame({
    'count':
    train_ts.groupby(pd.Grouper(freq='D'))['totals.hits'].sum(),
    'set':
    'train'
})
test_daily_hit_count = pd.DataFrame({
    'count':
    test_ts.groupby(pd.Grouper(freq='D'))['totals.hits'].sum(),
    'set':
    'test'
})
daily_hit_count = pd.concat(
    [train_daily_hit_count, test_daily_hit_count], axis=0)
p = figure(x_axis_type='datetime', width=700, height=400, title='pageviews over time')
p.line(
    source=daily_pageviews[daily_pageviews['set'] == 'train'],
    x='visitStartTime',
    y='count',
    color=Set1_9[0])
p.line(
    source=daily_pageviews[daily_pageviews['set'] == 'test'],
    x='visitStartTime',
    y='count',
    color=Set1_9[1])
p.legend.location = 'top_left'
test_start = Span(location=test_start_time.timestamp() * 1000, dimension='height', line_dash='dashed')
p.add_layout(test_start)
show(p)

p = figure(x_axis_type='datetime', width=700, height=400, title='hits over time')
p.line(
    source=daily_hit_count[daily_hit_count['set'] == 'train'],
    x='visitStartTime',
    y='count',
    color=Set1_9[0])
p.line(
    source=daily_hit_count[daily_hit_count['set'] == 'test'],
    x='visitStartTime',
    y='count',
    color=Set1_9[1])
p.legend.location = 'top_left'
test_start = Span(
    location=test_start_time.timestamp() * 1000,
    dimension='height',
    line_dash='dashed')
p.add_layout(test_start)
show(p)
train_daily_hitrate = pd.DataFrame({
    'count':
    train_ts.groupby(pd.Grouper(freq='D')).apply(
        lambda x: (x['totals.hits'].sum() + 1) / (x['totals.pageviews'].sum() + 1)
    ),
    'set':
    'train'
})
test_daily_hitrate = pd.DataFrame({
    'count':
    test_ts.groupby(pd.Grouper(freq='D')).apply(
        lambda x: (x['totals.hits'].sum() + 1) / (x['totals.pageviews'].sum() + 1)
    ),
    'set':
    'test'
})
daily_hitrate = pd.concat([train_daily_hitrate, test_daily_hitrate], axis=0)
p = figure(x_axis_type='datetime', width=700, height=400, title='hits/pageviews ratio over time')
p.line(
    source=daily_hitrate[daily_hitrate['set'] == 'train'],
    x='visitStartTime',
    y='count',
    color=Set1_9[0])
p.line(
    source=daily_hitrate[daily_hitrate['set'] == 'test'],
    x='visitStartTime',
    y='count',
    color=Set1_9[1])
p.legend.location = 'top_left'
test_start = Span(location=test_start_time.timestamp() * 1000, dimension='height', line_dash='dashed')
p.add_layout(test_start)
show(p)
train_daily_mobile_rate = pd.DataFrame({
    'count':
    train_ts.groupby(pd.Grouper(freq='D'))['device.isMobile'].mean(),
    'set':
    'train'
})
test_daily_mobile_rate = pd.DataFrame({
    'count':
    test_ts.groupby(pd.Grouper(freq='D'))['device.isMobile'].mean(),
    'set':
    'test'
})
daily_mobile_rate = pd.concat([train_daily_mobile_rate, test_daily_mobile_rate], axis=0)
p = figure(x_axis_type='datetime', width=700, height=400, title='daily mobile percentage over time')
p.line(
    source=daily_mobile_rate[daily_mobile_rate['set'] == 'train'],
    x='visitStartTime',
    y='count',
    color=Set1_9[0])
p.line(
    source=daily_mobile_rate[daily_mobile_rate['set'] == 'test'],
    x='visitStartTime',
    y='count',
    color=Set1_9[1])
p.legend.location = 'top_left'
test_start = Span(location=test_start_time.timestamp() * 1000, dimension='height', line_dash='dashed')
p.add_layout(test_start)
show(p)
train_daily_channel_grouping = train_ts.groupby([pd.Grouper(freq='D'), 'channelGrouping']).size().reset_index()
train_daily_channel_grouping['set'] = 'train'
test_daily_channel_grouping = test_ts.groupby([pd.Grouper(freq='D'), 'channelGrouping']).size().reset_index()
test_daily_channel_grouping['set'] = 'test'
daily_channel_grouping = pd.concat([train_daily_channel_grouping, test_daily_channel_grouping], axis=0)
daily_channel_grouping.rename(columns={0: 'count'}, inplace=True)
daily_channel_grouping['perc'] = daily_channel_grouping['count']
daily_channel_grouping_merged = pd.merge(
    daily_channel_grouping,
    daily_sess_count.reset_index()[['visitStartTime', 'count'
                                    ]].rename(columns={'count': 'sess_count'}),
    on='visitStartTime')
daily_channel_grouping_merged[
    'perc'] = daily_channel_grouping_merged['count'] / daily_channel_grouping_merged['sess_count']
p = figure(
    x_axis_type='datetime',
    width=700,
    height=600,
    title='channelGrouping percentage shares over time')
channels = list(daily_channel_grouping['channelGrouping'].unique())
for idx, channel in enumerate(channels):
    p.line(
        source=daily_channel_grouping_merged[
            (daily_channel_grouping_merged['channelGrouping'] == channel)
            & (daily_channel_grouping_merged['perc'] < 1)],
        x='visitStartTime',
        y='perc',
        color=Set1_9[idx],
        legend=channel)
p.y_range = Range1d(0, 1)
p.legend.click_policy = 'hide'
p.add_tools(
    HoverTool(tooltips=[('category',
                         '@channelGrouping'), ('percentage',
                                               '@perc'), ('count', '@count')]))
test_start = Span(
    location=test_start_time.timestamp() * 1000,
    dimension='height',
    line_dash='dashed')
p.add_layout(test_start)
show(p)
train_daily_browser = train_ts.groupby([pd.Grouper(freq='D'), 'device.browser']).size().reset_index()
train_daily_browser['set'] = 'train'
test_daily_browser = test_ts.groupby([pd.Grouper(freq='D'), 'device.browser']).size().reset_index()
test_daily_browser['set'] = 'test'
daily_browser = pd.concat([train_daily_browser, test_daily_browser], axis=0)
daily_browser.rename(columns={0: 'count'}, inplace=True)
daily_browser['perc'] = daily_browser['count']

daily_browser_merged = pd.merge(
    daily_browser,
    daily_sess_count.reset_index()[['visitStartTime', 'count'
                                    ]].rename(columns={'count': 'sess_count'}),
    on='visitStartTime')
daily_browser_merged[
    'perc'] = daily_browser_merged['count'] / daily_browser_merged['sess_count']
top_browsers = set(train['device.browser'].value_counts().index[:5])
top_browsers
p = figure(
    x_axis_type='datetime',
    width=700,
    height=600,
    title='browser percentage shares over time')
for idx, browser in enumerate(top_browsers):
    p.line(
        source=daily_browser_merged[
            (daily_browser_merged['device.browser'] == browser)
            & (daily_browser_merged['perc'] < 1)],
        x='visitStartTime',
        y='perc',
        color=Set1_9[idx],
        legend=browser)
p.y_range = Range1d(0, 1)
p.legend.click_policy = 'hide'
p.add_tools(
    HoverTool(tooltips=[('category',
                         '@{device.browser}'), ('percentage',
                                                '@perc'), ('count',
                                                           '@count')]))
test_start = Span(
    location=test_start_time.timestamp() * 1000,
    dimension='height',
    line_dash='dashed')
p.add_layout(test_start)
show(p)
train_daily_os = train_ts.groupby([pd.Grouper(freq='D'), 'device.operatingSystem']).size().reset_index()
train_daily_os['set'] = 'train'
test_daily_os = test_ts.groupby([pd.Grouper(freq='D'), 'device.operatingSystem']).size().reset_index()
test_daily_os['set'] = 'test'
daily_os = pd.concat([train_daily_os, test_daily_os], axis=0)
daily_os.rename(columns={0: 'count'}, inplace=True)
daily_os['perc'] = daily_os['count']

daily_os_merged = pd.merge(
    daily_os,
    daily_sess_count.reset_index()[['visitStartTime', 'count'
                                    ]].rename(columns={'count': 'sess_count'}),
    on='visitStartTime')
daily_os_merged[
    'perc'] = daily_os_merged['count'] / daily_os_merged['sess_count']
top_os = set(train['device.operatingSystem'].value_counts().index[:6])
top_os
p = figure(
    x_axis_type='datetime',
    width=700,
    height=600,
    title='OS percentage shares over time')
for idx, os in enumerate(top_os):
    p.line(
        source=daily_os_merged[
            (daily_os_merged['device.operatingSystem'] == os)
            & (daily_os_merged['perc'] < 1)],
        x='visitStartTime',
        y='perc',
        color=Set1_9[idx],
        legend=os)
p.y_range = Range1d(0, 0.7)
p.legend.click_policy = 'hide'
p.add_tools(
    HoverTool(tooltips=[('category',
                         '@{device.operatingSystem}'), ('percentage',
                                                        '@perc'), ('count',
                                                                   '@count')]))
test_start = Span(
    location=test_start_time.timestamp() * 1000,
    dimension='height',
    line_dash='dashed')
p.add_layout(test_start)
show(p)
train_daily_revenue = pd.DataFrame({
    'sum':
    train_ts.groupby(pd.Grouper(freq='D'))['totals.transactionRevenue'].sum(),
    'set':
    'train'
})
test_daily_revenue = pd.DataFrame({'dummy': test_ts.groupby(pd.Grouper(freq='D')).size(), 'set': 'test'})
test_daily_revenue['sum'] = train_daily_revenue['sum'].mean()
test_daily_revenue.drop('dummy', axis=1, inplace=True)
daily_revenue = pd.concat([train_daily_revenue, test_daily_revenue], axis=0, sort=True)

p = figure(
    x_axis_type='datetime',
    width=700,
    height=400,
    title='revenue over time?')
p.line(
    source=daily_revenue[daily_revenue['set'] == 'train'],
    x='visitStartTime',
    y='sum',
    color=Set1_9[0])
p.line(
    source=daily_revenue[daily_revenue['set'] == 'test'],
    x='visitStartTime',
    y='sum',
    color=Set1_9[1])
p.legend.location = 'top_left'
test_start = Span(
    location=test_start_time.timestamp() * 1000,
    dimension='height',
    line_dash='dashed')
p.add_layout(test_start)
show(p)
