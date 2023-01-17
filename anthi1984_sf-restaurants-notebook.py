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
import numpy as np
import pandas as pd
import datetime
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
init_notebook_mode(connected=True)
import os
print(os.listdir("../input"))


proc_data = pd.read_csv(r'../input/restaurant-scores-lives-standard.csv')
proc_data.head(10)

ax = proc_data.groupby('inspection_date').size().plot.line(figsize =(12,6))
ax.set_title('Inspection Dates Distribution')
ax.set_label("Inspection Date")

