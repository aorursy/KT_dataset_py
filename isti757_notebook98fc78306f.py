# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.DataFrame({'p' : [1, 2, 3]*3, 'b' : [4]*3+[5]*3+[6]*3, 'n' : np.cumsum([1]*9)}).sample(frac=1)

df.set_index(['p', 'b'], inplace=True)
df1 = df.groupby(level=[0])

(df1.n.transform(np.cumsum) / df1.n.transform(np.sum)).reset_index()
df.sort_index(level=[0, 1], inplace=True)

df1 = df.groupby(level=[0])
sns.factorplot(data=(df1.n.transform(np.cumsum) / df1.n.transform(np.sum)).reset_index(),

              col='p', x='b', y='n')
sns.factorplot(data=(df1.n.transform(np.cumsum) / df1.n.transform(np.sum)).reset_index(),

              col='p', x='b', y='n')