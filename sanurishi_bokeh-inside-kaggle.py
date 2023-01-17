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
from bokeh.plotting import figure

print("imported bokeh figure")

from bokeh.io import output_file, show



x = [1,2,3,4,5,6]

y = [1,2,3,4,5,6]
p = figure(title  ='sample' ,x_axis_label= 'x',y_axis_label= 'y' )
p.line(x ,y, legend = "test", line_width = 2)

output_file("test.html")
show(p)

print("not shown")