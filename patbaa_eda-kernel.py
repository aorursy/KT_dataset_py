# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
labels = pd.read_csv('../input/train_labels.csv')
sns.barplot(x='isHappy',y='count', 
            data=labels.groupby('isHappy').count().reset_index().rename(columns={'ID':'count'}))
plt.show()
plt.rcParams['figure.figsize'] = (15, 3)
rollmean = labels.rolling(50, center=False, min_periods=1).mean()
plt.scatter(x=rollmean.ID.tolist(), y=rollmean.isHappy.tolist(), s=1)
plt.xlabel('ID')
plt.ylabel('avg happiness')
plt.show()