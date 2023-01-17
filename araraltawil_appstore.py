# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

data=pd.read_csv('/kaggle/input/windows-store/msft.csv')
data.head()
data.info()
data.shape
missing_percentage=data.isna().sum()*100/data.shape[0]
missing_percentage

Category = data['Category'].value_counts()
Category = pd.DataFrame(Category)
Category=Category.reset_index()

fig = px.pie(Category, values='Category', names='index', color_discrete_sequence=px.colors.sequential.RdBu)
fig.show()
Music_data=data[data['Category']=='Music']
Music_data = Music_data['Rating'].value_counts()
Music_data = pd.DataFrame(Music_data)
Music_data=Music_data.reset_index()
fig = px.pie(Music_data, values='Rating', names='index', color_discrete_sequence=px.colors.sequential.RdBu)
fig.show()

_data=data[data['Category']=='Books']
_data = _data['Rating'].value_counts()
_data = pd.DataFrame(_data)
_data=_data.reset_index()
fig = px.pie(_data, values='Rating', names='index', color_discrete_sequence=px.colors.sequential.RdBu)
fig.show()