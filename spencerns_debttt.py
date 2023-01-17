%matplotlib inline
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
five = pd.read_csv('../input/MERGED2005_PP.csv')
df = five[['INSTNM', 'PCIP38', 'RELAFFIL']] 
df.head()
df.sort('PCIP38', ascending=False)
df.head()
df2 = five[['INSTNM', 'AVGFACSAL']]
df2.head()
df2.dropna()
df2 = df2.sort('AVGFACSAL', ascending=False)
df2['rank'] = df2['AVGFACSAL'].rank()
print(df2['rank'])
import matplotlib.pyplot as plt
df2.fillna(0)
plt.plot(df2.rank, df2.AVGFACSAL)
