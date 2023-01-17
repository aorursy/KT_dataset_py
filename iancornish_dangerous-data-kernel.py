# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

drugs = pd.read_csv('../input/drugsComTest_raw.csv')
drugs.head()
# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt 
plt.scatter('usefulCount','rating', data=drugs)
from bokeh.plotting import figure, output_file, show
output_file("plot.html")

# create a new plot with a title and axis labels
p = figure(title="scatter", x_axis_label='Rating', y_axis_label='User Count')

# add a line renderer with legend and line thickness
p.scatter(drugs['rating'], drugs['userCount'], legend="Key", line_width=2)

# show the results
show(p)