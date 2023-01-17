# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Any results you write to the current directory are saved as output.
dateparse = lambda x: pd.datetime.strptime(x, "%m/%d/%Y")

bom = pd.read_csv("../input/FX Rates BOM.csv", parse_dates=['Date'], date_parser=dateparse)
bom.head()
from bokeh.io import output_notebook, show
from bokeh.plotting import figure
output_notebook()
p = figure(x_axis_type='datetime')
p.line(bom["Date"],bom["USD"])
show(p)