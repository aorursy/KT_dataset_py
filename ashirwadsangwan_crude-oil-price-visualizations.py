# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_excel('../input/Crude Oil Prices Daily.xlsx')
data.head()
data.tail()
import matplotlib.pyplot as plt
plt.figure(figsize = (20,10));
data['Closing Value'].plot(kind='bar');
plt.ylabel('$ Prices ',fontsize = 18);
plt.title('Variation in Crude Prices over the years', Fontsize = 24);
plt.xticks(color = 'w');
plt.figure(figsize = (20,10));
data['Closing Value'].plot(kind='line');
plt.ylabel('$ Prices ',fontsize = 18);
plt.title('Variation in Crude Prices over the years', Fontsize = 24);

data['Closing Value'].describe()
import warnings 
warnings.filterwarnings('ignore')
import seaborn as sns
plt.figure(figsize = (20,10))
sns.violinplot(data['Closing Value'], color = 'Orange');
plt.xlabel('Closing Value',fontsize = 18);

