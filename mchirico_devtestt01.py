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
import pandas as pd

import numpy as np

import datetime





import warnings

warnings.filterwarnings("ignore")

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="white", color_codes=True)





# Good for interactive plots

import plotly.plotly as py

import plotly.graph_objs as go

from plotly import tools

from plotly.offline import iplot, init_notebook_mode

init_notebook_mode()





# You might want to get started with accident.csv 



# Read data accident.csv



FILE="../input/globalterrorismdb_0616dist.csv"

d=pd.read_csv(FILE, encoding = "ISO-8859-1")
d.country_txt.value_counts()

d=d[d['country_txt']=='United States']
d[['eventid','latitude','longitude','attacktype1_txt','success']].head()
import random

d['latitude']=d['latitude'].apply(lambda x: x+ (random.random()*0.00001)   )

d['longitude']=d['longitude'].apply(lambda x: x+ (random.random()*0.00001)   )

#d['longitude']=d[['longitude']].astype(float)+(random.random()*0.00001)
d.attacktype1_txt.value_counts()