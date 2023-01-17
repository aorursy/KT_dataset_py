# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from ipywidgets import interact

import numpy as np

import time

from bokeh.io import push_notebook, show, output_notebook

from bokeh.models import HoverTool, CustomJS, Slider

from bokeh.plotting import figure

output_notebook()
N = 1000

x = np.linspace(0, 4*np.pi, 2000)

y = np.sin(x)

TOOLS="crosshair,pan,wheel_zoom,box_zoom,reset,tap,box_select,lasso_select"

hover = HoverTool(tooltips=None, mode="vline")
p = figure(title="simple line example",plot_height=300, plot_width=600, y_range=(-5,5), tools=TOOLS)

p.add_tools(hover)

r = p.line(x, y, color="#2222aa", line_width=3)
def update(f, w=1, A=1, phi=0):

    if f == "sin": func = np.sin

    elif f == "cos": func = np.cos

    elif f == "tan": func = np.tan

    r.data_source.data['y'] = A * func(w * x + phi)

    hover.mode = "vline"

    push_notebook(handle=target)
# get and explicit handle to update the next show cell with

target = show(p, notebook_handle=True)



interact(update, f=["sin", "cos", "tan"], w=(0,50,.1), A=(1,5,.1), phi=(0, 20, 0.1))



callback = CustomJS(code="""

if (IPython.notebook.kernel !== undefined) {

    var kernel = IPython.notebook.kernel;

    cmd = "update(" + cb_obj.value + ")";

    kernel.execute(cmd, {}, {});

}

""")



A_Slider = Slider(start=1, 

                end=5,

                value=1,

                step=.1,

                title="w",

                callback=callback)