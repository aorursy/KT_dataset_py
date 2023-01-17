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
import seaborn as sns

import matplotlib.pyplot as pp

scrubbed=pd.read_csv('../input/scrubbed.csv',parse_dates=[0],sep=',',

                     dtype={'city':'category',

                           'country':'category',

                           'shape':'category'},

                    low_memory=False)

print('Scrubbed size: %s' % scrubbed.shape[0])
scrubbed.dtypes
sns.countplot(scrubbed.country)
ax=sns.countplot(scrubbed['shape'])

for tick in ax.get_xticklabels():

    tick.set_rotation(75)
scrubbed['city'].value_counts()