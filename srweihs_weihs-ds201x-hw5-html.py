# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import seaborn as sns
import matplotlib.pyplot as plt 
%matplotlib inline
lemon = pd.read_csv("../input/Lemonade2016-2.xls")
lemon
lemon['Sales'] = lemon['Lemon']+lemon['Orange']
lemon['Revenue'] =lemon['Sales']+lemon['Price']
lemon
from bokeh.io import show, output_notebook
from bokeh.plotting import figure
lemon
p = figure(title = "Sales by Temperature. With Revenue as Circle Size",x_axis_label = "Temperature",y_axis_label = "Sales")
p.circle(lemon['Temperature'],lemon['Orange'],size = lemon['Revenue']/2, line_color="navy", fill_color="orange",fill_alpha=0.5)
p.circle(lemon['Temperature'],lemon['Lemon'], size=lemon['Revenue']/2, line_color="navy",fill_color="green", fill_alpha=0.5)
show(p)
output("Weihs-DS201X-HW5-plot1.html")
from bokeh.plotting import *
from bokeh.models import ColumnDataSource
D_source = ColumnDataSource(lemon)
TOOLS = "pan,wheel_zoom,reset,save,box_select,lasso_select"
p1 = figure(tools=TOOLS, title="Revenue by Leaflets")
p1.circle('Leaflets','Revenue',color="blue",source=D_source)
p2 = figure(tools=TOOLS, title="Revenue vs. Sales")
p2.circle('Sales','Revenue',color="green",source=D_source)
p3 = figure(tools=TOOLS,title="Sales by Temperature")
p3.circle('Temperature','Sales',line_color="red",fill_color=None,source=D_source)
p = gridplot([[p1,p2,p3]],toolbar_location="right")
show(p)
output_file("Weihs-DS201X-HW5-plot2.html")
lemon.boxplot(column="Revenue", by="Location",figsize=(15,10),fontsize=15,rot=45)
p = figure(title="Lemon and Orange Sales by Temperature")
p.circle(x=lemon['Temperature'],y=lemon['Lemon'],size=10,alpha=0.5,color="green",legend="Lemon")
p.triangle(x=lemon["Temperature"],y=lemon["Orange"], size=10,alpha=0.3,color="orange",legend="Orange")
p.legend.location = "top_left"
show(p)

