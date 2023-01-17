# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/Iris.csv")
df.describe()
from bokeh.io import output_notebook, show

from bokeh.charts import BoxPlot

from bokeh.layouts import gridplot

from bokeh.plotting import figure

from bokeh.models import HoverTool

from bokeh.resources import CDN

from bokeh.resources import INLINE

from bokeh.embed import components

from bokeh.util.string import encode_utf8

from bokeh.charts import Histogram

output_notebook()
box_plot = BoxPlot(df, label=['Species'], values='PetalWidthCm',color='Species', title='Species Boxplot',legend='top_right') 

show(gridplot([[box_plot]],plot_width=600, plot_height=400))
df.head()
import seaborn as sbn

sbn.pairplot(df.drop("Id", axis=1), hue="Species", size=3)