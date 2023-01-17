# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # Visualization

from scipy.stats import ttest_ind # T-test



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/cereal.csv')

df.describe()
# Perform the t-test

t, prob = ttest_ind(df['sugars'], df['sodium'], equal_var=False)

print(f'Got t={t}, p={prob}')
sns.distplot(df['sugars'], kde=False).set_title('Sugar values')
sns.distplot(df['sodium'], kde=False).set_title('Sodium values')