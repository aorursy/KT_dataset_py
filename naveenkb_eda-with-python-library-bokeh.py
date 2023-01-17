import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from bokeh.models import LinearAxis, Range1d
from bokeh.transform import dodge
from bokeh.core.properties import value

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.io import output_notebook, show

from wordcloud import WordCloud

import os
# print(os.listdir("../input"))
np.set_printoptions(suppress=True)
output_notebook()
train = pd.read_csv('../input/train.csv', parse_dates = ['activation_date'])
test = pd.read_csv('../input/test.csv', parse_dates = ['activation_date'])
periods_train = pd.read_csv('../input/periods_train.csv', parse_dates = ['activation_date', 'date_from', 'date_to'])
# train_active = pd.read_csv('../input/train_active.csv')
print('Number of Observations in train is {0} and number of columns is {1}'.format(train.shape[0], train.shape[1]))
print('Number of Observations in periods_train is {0} and number of columns is {1}'.format(periods_train.shape[0], periods_train.shape[1]))
print('A sample of train data')
train.head()
print('A sample of period_train data')
periods_train.head()
""" 
Function to highlight rows based on data type
"""
def dtype_highlight(x):
    if x['type'] == 'object':
        color = '#2b83ba'
    elif (x['type'] == 'int64') | (x['type'] == 'int32'):
        color = '#abdda4'
    elif (x['type'] == 'float64') | (x['type'] == 'float32'):
        color = '#ffffbf'
    elif x['type'] == 'datetime64[ns]':
        color = '#fdae61'
    else:
        color = ''
    return ['background-color : {}'.format(color) for val in x]

train_dtypes = pd.DataFrame(train.dtypes.reset_index())
train_dtypes.columns = ['column', 'type']
periods_train_dtypes = pd.DataFrame(periods_train.dtypes.reset_index())
periods_train_dtypes.columns = ['column', 'type']

train_dtypes.style.apply(dtype_highlight, axis = 1)
periods_train_dtypes.style.apply(dtype_highlight, axis = 1)
train.image_top_1 = train.image_top_1.astype(object)
desc = train.describe(include=['O']).sort_values('unique', axis = 1)
def highlight_row(x):
    if x.name == 'unique':
        color = 'lightblue'
    else:
        color = ''
    return ['background-color: {}'.format(color) for val in x]
desc.style.apply(highlight_row, axis =1)
# Since image column is a ID code for image, let's remove it for time being
train.drop('image', axis = 1, inplace = True)
def color_zero_red(val):
    """
    Takes a scalar and returns a string with
    the css property `'background-color: red'` for negative
    strings, black otherwise.
    """
    color = 'red' if val == 0 else ' '
    return 'background-color: %s' % color
# Summary of numeric columns
train.describe().style.applymap(color_zero_red)
# Missing values in each column
train.isnull().sum().sort_values(ascending =False)
train.deal_probability.describe()
(train.deal_probability ==0).sum()/train.shape[0]
non_zero_probability = train.deal_probability[train.deal_probability !=0]

hist, edges = np.histogram(non_zero_probability, 
                               bins = 50, 
                               range = [0, 1])

hist_edges = pd.DataFrame({'#items': hist, 
                       'left': edges[:-1], 
                       'right': edges[1:]})
hist_edges['cumulative_items'] = hist_edges['#items'].cumsum()
hist_edges['p_interval'] = ['%.2f to %.2f' % (left, right) for left, right in zip(hist_edges['left'], hist_edges['right'])]

src = ColumnDataSource(hist_edges)
hover1 = HoverTool(tooltips=[
    ("probability interval", "@p_interval"),
    ("#Items", "$y")
])

p1 = figure(title="deal_probability histogram",  y_axis_label='No.of.Items', x_axis_label='probability', tools = [hover1], background_fill_color="#E8DDCB")
p1.title.align = 'center'
p1.left[0].formatter.use_scientific = False
p1.below[0].formatter.use_scientific = False
p1.quad(top='#items', bottom=0, left='left', right='right',
        fill_color="#036564", line_color = "#2B2626", source =src)
show(p1)
hover2 = HoverTool(tooltips=[
    ("probability <=", "@right"),
    ("#Items", "$y")
])

p2 = figure(title="deal_probability cumulative",  y_axis_label='No.of.Items', x_axis_label='probability', tools = [hover2], background_fill_color="#E8DDCB")
p2.title.align = 'center'
p2.left[0].formatter.use_scientific = False
p2.below[0].formatter.use_scientific = False
p2.quad(top='cumulative_items', bottom=0, left='left', right='right',
        fill_color="#036564", line_color = "#2B2626", source =src)
# p.line('left', 'cumulative_items', line_color="#9E3030", source = src)
show(p2)
user_item = train.groupby('user_id')['item_id'].count().sort_values()
user_item_dist = user_item.value_counts().reset_index()
user_item_dist.columns = ['No.of Ads', 'No.of Users']

user_item_dist['pct'] = user_item_dist['No.of Users']/user_item_dist['No.of Users'].sum()
user_item_dist.head()
user_item_dist.tail()
user_item_dist[user_item_dist['No.of Ads'] == 1].loc[:,'pct']
user_item_dist[user_item_dist['No.of Users'] == 1].loc[:,'No.of Ads'].max()
user_item_dist[user_item_dist['No.of Users'] == 1].loc[:,'No.of Ads'].min()
user_item_dist2 = user_item_dist[(user_item_dist['No.of Ads'] > 1) & (user_item_dist['No.of Users'] > 1)]

hover3 = HoverTool(tooltips=[
    ("No.of Ads ==", "@right"),
    ("Percent of Users", "$y")
])

p3 = figure(title="Distribution of No. of Ads posted by users",  y_axis_label='No.of.Users', x_axis_label='No.of Ads posted', tools = [hover3], background_fill_color="#E8DDCB")
p3.title.align = 'center'
p3.left[0].formatter.use_scientific = False
p3.below[0].formatter.use_scientific = False
p3.quad(top= user_item_dist2['pct'], bottom=0, left= user_item_dist2['No.of Ads'][:-1], right= user_item_dist2['No.of Ads'][1:],
        fill_color="#036564", line_color = "#2B2626")
show(p3)
user_item = user_item.reset_index()
user_item.columns = ['user_id', 'No.of Ads']
train = train.merge(user_item, on = 'user_id', how = 'left')
train['No.of Ads bin'] = pd.cut(train['No.of Ads'], 20, labels = range(20))

Ads_bin_prob = train.groupby(['No.of Ads bin'])['deal_probability'].mean().reset_index()
Ads_bin_prob.columns = ['No.of Ads bin', 'avg_deal_probability']
hover4 = HoverTool(tooltips=[
    ("Ads bin ", "@right"),
    ("avg_deal_probability", "$y")
])

p4 = figure(title="Distribution of No. of Ads posted by users",  y_axis_label='No.of.Users', x_axis_label='No.of Ads posted', tools = [hover4], background_fill_color="#E8DDCB")
p4.title.align = 'center'
p4.left[0].formatter.use_scientific = False
p4.below[0].formatter.use_scientific = False
p4.quad(top= Ads_bin_prob['avg_deal_probability'], bottom=0, left= Ads_bin_prob['No.of Ads bin'][:-1], right= Ads_bin_prob['No.of Ads bin'][1:],
        fill_color="#036564", line_color = "#2B2626")
show(p4)
train.region.nunique()
f = {'deal_probability':['mean'], 'item_id': ['size']}
user_hist_prob = train.groupby('user_type').agg(f).reset_index()
user_hist_prob.columns = ['user_type','avg_deal_probability', 'user_type_count']
user_hist_prob
user_type_source = ColumnDataSource(data=user_hist_prob)
p5 = figure(x_range = list(user_hist_prob.user_type), plot_width=800, plot_height=400, title = 'user_type distribution and deal_probability mean')
p5.vbar(x = dodge('user_type', -0.20, range=p5.x_range), top = 'user_type_count', width=.4, color='#f45666', source = user_type_source, legend = value('user_type_count'))
p5.y_range =  Range1d(0, user_hist_prob.user_type_count.max())
p5.extra_y_ranges = {"avg_deal_probability": Range1d(start=0, end=1)}
p5.xaxis.axis_label = 'user_type'
p5.yaxis.axis_label = 'user_type_count'
p5.add_layout(LinearAxis(y_range_name="avg_deal_probability", axis_label= 'avg deal_probability'), 'right')
p5.vbar(x = dodge('user_type', 0.20, range=p5.x_range), top = 'avg_deal_probability', y_range_name='avg_deal_probability', width = 0.4, color='lightblue', source = user_type_source, legend = value('avg_deal_probability'))
p5.legend.location = "top_left"
p5.legend.orientation = "horizontal"
show(p5)
f = {'deal_probability':['mean'], 'item_id': ['size']}
parent_cat_hist_prob = train.groupby('parent_category_name').agg(f).reset_index()
parent_cat_hist_prob.columns = ['parent_category_name','avg_deal_probability', 'parent_category_count']
parent_cat_hist_prob
parent_cat_source = ColumnDataSource(data=parent_cat_hist_prob)
p6 = figure(x_range = list(parent_cat_hist_prob.parent_category_name), plot_width=800, plot_height=400, title = 'parent_category distribution and deal_probability mean')
p6.vbar(x = dodge('parent_category_name', -0.16, range=p6.x_range), top = 'parent_category_count', width=.3, color='#f45666', source = parent_cat_source, legend = value('parent_category_count'))
p6.y_range =  Range1d(0, parent_cat_hist_prob.parent_category_count.max())
p6.extra_y_ranges = {"avg_deal_probability": Range1d(start=0, end=1)}
p6.xaxis.axis_label = 'parent_category_name'
p6.yaxis.axis_label = 'parent_category_count'
p6.add_layout(LinearAxis(y_range_name="avg_deal_probability", axis_label= 'avg deal_probability'), 'right')
p6.vbar(x = dodge('parent_category_name', 0.16, range=p6.x_range), top = 'avg_deal_probability', y_range_name='avg_deal_probability', width = 0.3, color='lightblue', source = parent_cat_source, legend = value('avg_deal_probability'))
p6.legend.location = "top_left"
p6.legend.orientation = "horizontal"
show(p6)
train.region.value_counts()