# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
df = pd.read_excel("../input/friends.xls")
df.info()
df
freq = pd.Series(df['Friends']).value_counts()
plt.pie(freq, labels=freq.index, autopct='%.2f%%')
plt.axis('equal')
trace = go.Pie(labels=freq.index, values=freq.values, textinfo='value+percent+label')

py.iplot([trace], filename='basic_pie_chart')
cts = pd.DataFrame(freq)
cts
cts['Pct']=cts['Friends']/cts['Friends'].sum()
cts
