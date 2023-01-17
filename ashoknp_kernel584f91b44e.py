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
data_export = pd.read_csv("../input/india-trade-data/2018-2010_export.csv")
data_import = pd.read_csv("../input/india-trade-data/2018-2010_import.csv")
expensive_import = data_import[data_import.value>1000]
expensive_import.head(10)

expensive_import_by_country = expensive_import.groupby(['country']).agg({'value': 'sum'})
expensive_import_by_country = expensive_import_by_country.sort_values(by='value')
import seaborn as sns 
import matplotlib.pyplot as plt
import squarify

import_value =np.array(expensive_import_by_country)
import_country=expensive_import_by_country.index

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (10.0, 10.0)

squarify.plot(sizes=import_value, label=import_country, alpha=.8 )

plt.title("Expensive Import value by each country to India")
plt.axis('off')
plt.show()
