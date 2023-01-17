# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.express as px

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = (20,6) 
pd.set_option("display.max_columns", 100) 

pd.set_option("display.max_rows", 200)
df = pd.read_csv('/kaggle/input/uisummerschool/Online_sales.csv')
df
df.describe()
df.info()
df.hist();
grouped_age = df.groupby('Tax')['Tax'].count().reset_index(name='total_passengers')
df.summary()
from df import date
from datetime import datetime