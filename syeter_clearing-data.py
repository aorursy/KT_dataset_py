

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
data = pd.read_csv('../input/genius_hip_hop_lyrics.csv',encoding = "ISO-8859-1")
data.head()
data.info()
data.tail()
data.columns
data.shape
print(data['sentiment'].value_counts(dropna=False))
data.describe()
data.head()
data.boxplot(column='album_release_date',by = 'sentiment') #Lower quartile/Upper quartile -+ 1.5 IQR(upper quartile - lower quartile) = outliers
data_new = data.head()
data_new
melted = pd.melt(frame=data_new,id_vars = 'line', value_vars = ['artist','song'])
melted
melted.pivot(index = 'line',columns = 'variable',values = 'value')

data1 = data.head()
data2 = data.tail()
conc_data_row = pd.concat([data1,data2], axis = 0,ignore_index = True)
conc_data_row
data1 = data['song'].head()
data2 = data['artist'].head()
conc_data_col = pd.concat([data1,data2],axis=1)
conc_data_col
data.dtypes
data['theme'] = data['theme'].astype('object')
data['sentiment'] = data['sentiment'].astype('category')
data.dtypes
data.info()
data["theme"].value_counts(dropna = False)
data1 = data
data1["theme"].dropna(inplace=True) #Inplace means changes are going to be valid for main dataframe
assert data["theme"].notnull().all()
data["theme"].fillna('empty',inplace = True) # element of dataframe must be an object.

assert data["theme"].notnull().all()
assert data.theme.dtypes == np.object
data.dtypes
