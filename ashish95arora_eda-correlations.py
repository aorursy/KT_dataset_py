#imports
import pandas as pd
import numpy as np
import sklearn
import os
from dateutil import parser
from bokeh.plotting import figure, output_notebook, show
from bokeh.resources import INLINE
from bokeh.models import HoverTool,ColumnDataSource
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
meets=pd.read_csv('../input/meets.csv')
openpowerlifting=pd.read_csv('../input/openpowerlifting.csv')
meets.head()
openpowerlifting.head()
print(meets.shape)
print(openpowerlifting.shape)
powerlifting=pd.merge(openpowerlifting,meets)
powerlifting.columns
powerlifting.describe()
powerlifting.Date.dtype
powerlifting.Date=powerlifting.Date.apply(lambda x:parser.parse(x))
powerlifting.Date.dtype
powerlifting.year=powerlifting.Date.apply(lambda x:x.year)
year_group=powerlifting.groupby(by=powerlifting.year)

date=year_group.BestDeadliftKg.max().index
deadlift=list(year_group.BestDeadliftKg.max())

p = figure(plot_width=800, plot_height=250)
p.line(date,deadlift, color='navy', alpha=0.5)
# line is a glyph through which we just pass the parameters date and BestDeadliftKg
p.yaxis.axis_label = "BestDeadliftKg"
p.xaxis.axis_label = "Year"
output_notebook(resources=INLINE)
show(p)
kg_group=powerlifting.groupby(by=powerlifting.WeightClassKg)

group=kg_group.BestDeadliftKg.max().index
deadlift=list(kg_group.BestDeadliftKg.max())

p = figure(plot_width=800, plot_height=250,y_range=(0,600),x_range=(0,170))
p.scatter(group,deadlift, color='navy', alpha=0.5)
p.yaxis.axis_label = "BestDeadliftKg"
p.xaxis.axis_label = "Group"
output_notebook(resources=INLINE)
show(p)
powerlifting.year=powerlifting.Date.apply(lambda x:x.year)
year_group=powerlifting.groupby(by=powerlifting.year)

date=year_group.BestSquatKg.max().index
squats=list(year_group.BestSquatKg.max())

p = figure(plot_width=800, plot_height=250)
p.line(date,squats, color='navy', alpha=0.5)
# line is a glyph through which we just pass the parameters date and BestDeadliftKg
p.yaxis.axis_label = "BestSquatKg"
p.xaxis.axis_label = "Year"
output_notebook(resources=INLINE)
show(p)
kg_group=powerlifting.groupby(by=powerlifting.WeightClassKg)

maxbygroup=kg_group.BestSquatKg.max()
maxbygroup.dropna(inplace=True)
group=maxbygroup.index
squats=list(maxbygroup)

p = figure(plot_width=800, plot_height=250,y_range=(0,600),x_range=(0,170))
p.scatter(group,squats, color='navy', alpha=0.5)
p.yaxis.axis_label = "BestSquatKg"
p.xaxis.axis_label = "Group"
output_notebook(resources=INLINE)
show(p)
powerlifting.year=powerlifting.Date.apply(lambda x:x.year)
year_group=powerlifting.groupby(by=powerlifting.year)

date=year_group.BestBenchKg.max().index
bench=list(year_group.BestBenchKg.max())

p = figure(plot_width=800, plot_height=250)
p.line(date,bench, color='navy', alpha=0.5)
# line is a glyph through which we just pass the parameters date and BestDeadliftKg
p.yaxis.axis_label = "BestBenchKg"
p.xaxis.axis_label = "Year"
output_notebook(resources=INLINE)
show(p)
kg_group=powerlifting.groupby(by=powerlifting.WeightClassKg)

maxbygroup=kg_group.BestBenchKg.max()
maxbygroup.dropna(inplace=True)
group=maxbygroup.index
bench=list(maxbygroup)

p = figure(plot_width=800, plot_height=250,y_range=(0,600),x_range=(0,170))
p.scatter(group,bench, color='navy', alpha=0.5)
p.yaxis.axis_label = "BestBenchKg"
p.xaxis.axis_label = "Group"
output_notebook(resources=INLINE)
show(p)
data=pd.DataFrame.from_dict(dict(Counter(powerlifting.MeetTown).most_common(10)),orient='index')
data['x'] = data.index
data.columns.values[0]='y'
ax=sns.factorplot(x="x", y="y",size=10,aspect=1, data=data, kind="bar")
ax.set(xlabel='Venues', ylabel='Number of meetups')
plt.show()
data=pd.DataFrame.from_dict(dict(Counter(powerlifting.MeetName).most_common(10)),orient='index')
data['x'] = data.index
data.columns.values[0]='y'
ax=sns.factorplot(x="x", y="y",size=10,aspect=1, data=data, kind="bar",palette="Blues_r")
ax.set(xlabel='Championships', ylabel='Number of meetups')
ax.set_xticklabels(rotation=90)
plt.show()
powerlifting_2000s=powerlifting[(powerlifting['Date'] > '2000-01-01')]
powerlifting_2000s.year=powerlifting_2000s.Date.apply(lambda x:x.year)
year_group=powerlifting_2000s.groupby(by=powerlifting_2000s.year)

date=year_group.TotalKg.mean().index
total_weightlifted=list(year_group.TotalKg.mean())

p = figure(plot_width=800, plot_height=250)
p.line(date,total_weightlifted, color='navy', alpha=0.5)
# line is a glyph through which we just pass the parameters date and BestDeadliftKg
p.yaxis.axis_label = "TotalKg"
p.xaxis.axis_label = "Year"
output_notebook(resources=INLINE)
show(p)
def plot_corr(df,threshold,size=10):
    '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.

    Input:
        df: pandas DataFrame
        threshold: Threshold value for correlation
        size: vertical and horizontal size of the plot
        
    return:
        corr: dataframe representing correlation between variables
    '''
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    newdf = df.select_dtypes(include=numerics)
    f, ax = plt.subplots(figsize=(size, size))
    corr = newdf.corr()
#     corr = corr.where(np.triu(np.ones(corr.shape)).astype(np.bool))
    sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)
    return corr.style.applymap(lambda x:'background-color: #E9967A' if np.abs(x)>threshold and np.abs(x)<1  else 'color: black')

plot_corr(powerlifting,0.7,10)
