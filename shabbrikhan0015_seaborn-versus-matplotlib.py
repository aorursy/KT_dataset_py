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
import matplotlib.pyplot as plt
plt.style.use('classic')
%matplotlib inline
import numpy as np
import pandas as pd
range = np.random.RandomState(0)
x = np.linspace(0, 10, 500)
y = np.cumsum(range.randn(100, 5))
# Plot the data with Matplotlib defaults
plt.plot(x,y)
plt.legend('A', ncol=2, loc='upper right')
import seaborn as sns
sns.set()

data = np.random.multivariate_normal([0, 0], [[5, 2], [2, 2]], size=2000)
data = pd.DataFrame(data, columns=['x', 'y'])

for col in 'xy':
    plt.hist(data[col], normed=True, alpha=0.5)


for col in 'xy':
    sns.kdeplot(data[col], shade=True)


#SEABORN distplot
sns.distplot(data['x'])
sns.distplot(data['y']);
sns.kdeplot(data);
with sns.axes_style('white'):
    sns.jointplot("x", "y", data, kind='kde');
#histogram type form of above jointplot
with sns.axes_style('white'):
    sns.jointplot("x", "y", data, kind='hex')
