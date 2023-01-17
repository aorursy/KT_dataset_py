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
icustomer_data = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
icustomer_data.shape
icustomer_data.head()
data = icustomer_data.iloc[:, 9:11].values
print(data)
array_sum = np.sum(data)
array_has_nan = np.isnan(array_sum)

print(array_has_nan)

rows_data = data[1:100, :]
import plotly.figure_factory as ff
fig = ff.create_dendrogram(rows_data)
fig.update_layout(width=800, height=500)
fig.show()