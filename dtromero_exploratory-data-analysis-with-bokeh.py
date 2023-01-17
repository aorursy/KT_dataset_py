# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from bokeh.charts import Bar, TimeSeries, output_file, show, output_notebook
from bokeh.models import HoverTool, ColumnDataSource
output_notebook()

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
sample = pd.read_csv(u'../input/sample_submission.csv')
test = pd.read_csv(u'../input/test.csv')
train = pd.read_csv(u'../input/train.csv')

test['fileName'] = 'test'
train['fileName'] = 'train'

full = pd.concat((train, test))
outcomes = train.groupby(['AnimalType', 'OutcomeType'])\
            ['AnimalID'].count().reset_index()
outcomes.columns = ['AnimalType', 'OutcomeType', 'num_animals']
    
hover = HoverTool(
        tooltips=[
            ("Animal", "$x"),  
            ("Outcome", "@OutcomeType"),
            ("Num", "@height{int}"),
        ]
    )
p = Bar(outcomes, label='AnimalType', values='num_animals', agg='sum',\
        stack='OutcomeType', legend='top_left', plot_height=600,\
        tools=[hover]) 
#source = ColumnDataSource(p)
show(p)