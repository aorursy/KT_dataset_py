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
!pip install --upgrade pip

!pip install bokeh==0.12.16
import pandas as pd

from bokeh.plotting import figure, output_file, show
df = pd.read_csv('../input/advertising-dataset/advertising.csv')
df
df.shape
from bokeh.plotting import figure, output_file, show

from bokeh.io import show



output_file("./p1.html")



p1 = figure(plot_width=400, plot_height=400)



p1.circle(df['TV'], df['Sales'], size=2, color="navy", alpha=0.5)



show(p1)



output_file("./p2.html")



p2 = figure(plot_width=400, plot_height=400)



p2.circle(df['Radio'], df['Sales'], size=2, color="navy", alpha=0.5)



show(p2)
fig1 = figure(plot_width=400, plot_height=400)



fig1.vbar(x=df['Sales'], bottom=0, top=df['TV'],color='blue', width=0.75)



show(fig1)
fig2 = figure(plot_width=400, plot_height=400)



fig2.vbar(x=df['Sales'], bottom=0, top=df['Radio'],color='blue', width=0.75)



show(fig2)
from bokeh.layouts import gridplot



grid = gridplot([[p1,p2], [fig1, fig2]], plot_width=250, plot_height=250)



show(grid)