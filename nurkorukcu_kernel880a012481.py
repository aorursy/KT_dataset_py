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

import os
data_path = ['/kaggle/input/covid19-in-india']
print (data_path)
import numpy as np
import pandas as pd

filepath = os.sep.join(data_path + ['StatewiseTestingDetails.csv'])
print(filepath)
data = pd.read_csv(filepath)
data.head()
import matplotlib.pyplot as plt
%matplotlib inline
ax = plt.axes()

ax.scatter(data.Date, data.Positive)

# Eksenleri isimlendirme
ax.set(xlabel='Günler',
       ylabel='Vaka Sayısı',
       title='Günlük pozitif vaka sayısı');
plt.hist(data.Positive, bins=20)

plt.show()
data.info()
data.head() 
import seaborn as sns
X = data.drop(['TotalSamples'], axis=1)
y = data['TotalSamples']
sns.distplot(y)
plt.show()