# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv(r'../input/Energy Census and Economic Data US 2010-2014.csv')
df.head()
df.StateCodes.unique()
#Want to get rid of the US row which is just an aggregation of the other data
df2 = df.ix[~(df['StateCodes'] == 'US')]
plt.scatter(df2['GDP2014'],df2['TotalPrice2014'],s=np.power(df2['CENSUS2010POP'],0.3),alpha=0.6)