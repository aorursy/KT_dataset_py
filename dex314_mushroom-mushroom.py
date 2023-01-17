# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



import numpy as np

np.random.seed(31415)

import pandas as pd

import xgboost as xgb

import datetime

from sklearn.model_selection import train_test_split
df = pd.read_csv('../input/mushrooms.csv')

df.head(3)
import seaborn as sns

from matplotlib.pyplot import *
cnt = 0

for c in df.columns:

    if df[c].dtype == 'object':

        cnt += 1

        df[c] = pd.factorize(df[c])[0]

print(df.shape)

print(cnt)
df.head(3)
df.columns
fig, ax=subplots(1,1,figsize=(8,8))

sns.heatmap(df.corr(), ax=ax)
#sns.jointplot("bruises", "odor", data=df, kind="reg", color="r", size=7)

np.unique(df['class'])
sns.violinplot(x='bruises', y='odor', hue='gill-color', data=df,

              inner='point')

title('Odor and Bruises by Gill Color')
## lets play around with these features and multiply them together (adding 1 to deal with 0)

df['BO'] = (df.bruises+1)*(df.odor+1)

sns.violinplot(x='BO', y='spore-print-color', hue='gill-color', data=df,

              inner='point')

title('BO by Gill Color and Spore Print Color')