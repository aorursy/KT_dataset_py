# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np, pandas as pd
!pip install pyecharts
from pyecharts.charts import Pie

from pyecharts import options as opts


df= pd.read_csv('../input/polarbearrosechart/polarbear-rose.csv')
c = df['District'].values.tolist()

d = df['Case Rate'].values.tolist()

#create the color_series for the rosechart

color_series = ['#802200','#B33000','#FF4500','#FAA327','#9ECB3C',

 '#6DBC49','#37B44E','#14ADCF','#209AC9','#1E91CA',

 '#2C6BA0','#2B55A1','#2D3D8E','#44388E','#6A368B',

 '#D02C2A','#D44C2D','#F57A34','#FA8F2F','#D99D21']

rosechart = Pie(init_opts=opts.InitOpts(width='3000px', height='900px'))

# set the color

rosechart.set_colors(color_series)

# add the data to the rosechart

rosechart.add("", [list(z) for z in zip(c, d)],

        radius=["35%", "140%"],  # 20% inside radius，95% ourside radius

        center=["20%", "80%"],   # center of the chart

        rosetype="area")# set the global options for the chart

rosechart.set_global_opts(legend_opts=opts.LegendOpts(is_show=True),

                     toolbox_opts=opts.ToolboxOpts())# set the series options

rosechart.set_series_opts(label_opts=opts.LabelOpts(is_show=False, font_size=12, formatter="{b}:{c}%"),)
chart=rosechart.render_notebook()

chart