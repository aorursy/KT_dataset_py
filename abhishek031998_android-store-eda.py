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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df=pd.read_csv('../input/googleplaystore.csv')
df.head()
df.shape
df.drop_duplicates(subset='App',inplace=True)
df.isnull().sum()
df = df[df['Android Ver'] != np.nan]
df = df[df['Android Ver'] != 'NaN']
df.isnull().sum()
df = df[df['Installs'] != 'Free']
df = df[df['Installs'] != 'Paid']
df['Installs']=df['Installs'].apply(lambda x:x.replace('+','') if '+' in str(x) else x)
df['Installs']=df['Installs'].apply(lambda x:x.replace(',','') if ',' in str(x) else x)
df['Installs']=df['Installs'].apply(lambda x:int(x))

df.head()
df['Size'] = df['Size'].apply(lambda x: str(x).replace('Varies with device', 'NaN') if 'Varies with device' in str(x) else x)

df['Size'] = df['Size'].apply(lambda x: str(x).replace('M', '') if 'M' in str(x) else x)
df['Size'] = df['Size'].apply(lambda x: str(x).replace(',', '') if 'M' in str(x) else x)
df['Size'] = df['Size'].apply(lambda x: float(str(x).replace('k', '')) / 1000 if 'k' in str(x) else x)


df['Size'] = df['Size'].apply(lambda x: float(x))
df['Installs'] = df['Installs'].apply(lambda x: float(x))

df['Price'] = df['Price'].apply(lambda x: str(x).replace('$', '') if '$' in str(x) else str(x))
df['Price'] = df['Price'].apply(lambda x: float(x))

df['Reviews'] = df['Reviews'].apply(lambda x: int(x))
df.head()
df.isnull().sum()
total_values=df['Category'].value_counts().sort_values(ascending=True)
import plotly.graph_objs as go
import plotly
plotly.offline.init_notebook_mode(connected=True)
data=[go.Pie(labels=total_values.index,values=total_values.values,hoverinfo='label+value')]
data
plotly.offline.iplot(data, filename='pie_graph')

df['Rating'].hist()
print(np.mean(df['Rating']))
