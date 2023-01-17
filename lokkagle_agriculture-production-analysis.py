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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data1 = pd.read_csv('/kaggle/input/agricuture-crops-production-in-india/datafile.csv')
data1.head()
data2 = pd.read_csv('/kaggle/input/agricuture-crops-production-in-india/datafile (1).csv')
data2.head()
data3 = pd.read_csv('/kaggle/input/agricuture-crops-production-in-india/datafile (2).csv')
data3.head()
data31 = pd.read_csv('/kaggle/input/agricuture-crops-production-in-india/datafile (3).csv')
data31.head()
data4 = pd.read_csv('/kaggle/input/agricuture-crops-production-in-india/produce.csv')
data4.head()
data4.info()
data1.head()
data1.isna().sum()
data1.dropna(inplace= True)
data1.isna().sum()
data1.shape
data1['Crop'].unique()
# total production on each year by crop
plt.style.use('seaborn')
for i in data1.columns[1:]:
    print('Total crop on the year of {}'.format(i))
    data1.groupby('Crop')[i].sum().plot(kind = 'line', figsize = (15,5))
    plt.show()
data2.head()
data2.isna().sum()
data2['Crop'].unique()
data2['State'].unique()
data2.columns
data2.groupby(['State', 'Crop'])['Cost of Cultivation (`/Hectare) A2+FL'].sum().sort_values(ascending = True).plot(kind = 'barh',figsize = (13,13))
plt.tight_layout()
plt.title('Crop by State Based on Total Cost of Cultivation (`/Hectare) A2+FL')
plt.show()
data2.groupby(['State', 'Crop'])['Cost of Cultivation (`/Hectare) C2'].sum().sort_values(ascending = True).plot(kind = 'barh',figsize = (13,13))
plt.tight_layout()
plt.title('Crop by State Based on Total Cost of Cultivation (`/Hectare) C2')
plt.show()
data2.groupby(['State', 'Crop'])['Cost of Production (`/Quintal) C2'].sum().sort_values(ascending = True).plot(kind = 'barh',figsize = (13,13))
plt.tight_layout()
plt.title('Crop by State Based on Total Cost of Production (`/Quintal) C2')
plt.show()
data2.groupby(['State', 'Crop'])['Yield (Quintal/ Hectare) '].sum().sort_values(ascending = True).plot(kind = 'barh',figsize = (13,13))
plt.tight_layout()
plt.title('Crop by State Based on Total Yield (Quintal/ Hectare)')
plt.show()
data2.head()
# pairplot based on state
sns.pairplot(data2[['State', 'Cost of Cultivation (`/Hectare) A2+FL',
       'Cost of Cultivation (`/Hectare) C2',
       'Cost of Production (`/Quintal) C2', 'Yield (Quintal/ Hectare) ']], hue = 'State')
plt.tight_layout()
plt.show()
# pairplot based on Crop
sns.pairplot(data2[['Crop', 'Cost of Cultivation (`/Hectare) A2+FL',
       'Cost of Cultivation (`/Hectare) C2',
       'Cost of Production (`/Quintal) C2', 'Yield (Quintal/ Hectare) ']], hue = 'Crop')
plt.tight_layout()
plt.show()
data4.head()
data4.info()
data4.isna().sum()
data3 = pd.read_csv('/kaggle/input/agricuture-crops-production-in-india/datafile (2).csv')
data3.head()
data3.isna().sum()
data3.dtypes
data3.select_dtypes(include= np.float64).hist(figsize = (15,8))
plt.tight_layout()
plt.show()
data3.head()
#  Total crop production on respective years
for i in data3.columns[1:6]:
    data3.groupby('Crop             ')[i].sum().plot(kind = 'line', figsize = (15,10), legend = True)
    plt.title('Total crop production on respective years of'+' '+ i)
plt.show()
#  Total crop production on respective areas
for i in data3.columns[6:11]:
    data3.groupby('Crop             ')[i].sum().plot(kind = 'line', figsize = (15,10), legend = True)
    plt.title('Total crop production on respective areas of'+' '+ i)
plt.show()
#  Total crop production on respective yields
for i in data3.columns[-5:]:
    data3.groupby('Crop             ')[i].sum().plot(kind = 'line', figsize = (15,10), legend = True)
    plt.title('Total crop production on respective yields of'+' '+ i)
plt.show()
