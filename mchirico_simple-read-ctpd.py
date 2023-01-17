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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





from subprocess import check_output

print(check_output(["ls", "../input/"]).decode("utf8"))



if check_output(["ls", "../input/"]).decode('utf8') == 'crime.csv\n':

    DIR = "../input/crime.csv"

else:

    DIR = "../input/chtpd/crime.csv"
import pandas as pd

import numpy as np

import datetime





import plotly.plotly as py

import plotly.graph_objs as go

from plotly import tools

from plotly.offline import iplot, init_notebook_mode

init_notebook_mode()



# For Density plots

from plotly.tools import FigureFactory as FF





import datetime





import warnings

warnings.filterwarnings("ignore")

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="white", color_codes=True)









import warnings

warnings.filterwarnings("ignore")

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="white", color_codes=True)





dateparse = lambda x: datetime.datetime.strptime(x,'%m/%d/%Y %I:%M:00 %p')



# Read data 

d=pd.read_csv(DIR,parse_dates=['incident_datetime'],date_parser=dateparse)



# Adjustments

d['addr'] = d['address_1'].str.upper()

d['month'] = d['incident_datetime'].map(lambda x: x.month)

d['year'] = d['incident_datetime'].map(lambda x: x.year)



# Simple search

d[d.addr.str.match(r'.*FORES.*') & d.incident_description.str.match(r'.*BURGLARY.*')]
