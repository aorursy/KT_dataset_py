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
india_import=pd.read_csv('/kaggle/input/india-trade-data/2018-2010_import.csv')

india_export=pd.read_csv('/kaggle/input/india-trade-data/2018-2010_export.csv')
print(india_import.columns)
india_import_vlu=india_import.groupby(['country']).value.sum().reset_index(name='sum')

india_export_vlu=india_export.groupby(['country']).value.sum().reset_index(name='sum')
india_import_vlu=india_import_vlu.sort_values(by='sum', ascending = False)

india_export_vlu=india_export_vlu.sort_values(by='sum', ascending = False)
import plotly.express as px

fig = px.bar(india_import_vlu, x="sum", y="country",orientation='h',color='country',height=4000,)



fig.show()

import plotly.express as px

fig = px.bar(india_export_vlu, x="sum", y="country",orientation='h',color='country',height=4000,)



fig.show()
