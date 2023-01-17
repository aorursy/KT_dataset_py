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
# Importing Libraries
import pandas as pd
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.offline as pyo
import matplotlib.pyplot as plt
init_notebook_mode(connected=True)
# Reading file
datafile = '../input/uspollution/pollution_us_2000_2016.csv'
data = pd.read_csv(datafile)
print('Column headers :' + str(data.columns))
print('Rows: {} Columns: {} '.format(str(data.shape[0]), str(data.shape[1])))
