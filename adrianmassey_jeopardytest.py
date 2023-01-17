# Imports from __future__ in case we're running Python 2

from __future__ import division, print_function

from __future__ import absolute_import, unicode_literals



# Our numerical workhorses

import numpy as np

import scipy.integrate



# Import pyplot for plotting

import matplotlib.pyplot as plt



# Seaborn, useful for graphics

import seaborn as sns



# Import Bokeh modules for interactive plotting

import bokeh.io

import bokeh.mpl

import bokeh.plotting



# Magic function to make matplotlib inline; other style specs must come AFTER

%matplotlib inline



# Set up Bokeh for inline viewing

bokeh.io.output_notebook()
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