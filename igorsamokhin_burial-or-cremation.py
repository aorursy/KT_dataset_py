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
import matplotlib.pyplot as plt
%matplotlib inline
df = pd.read_csv('../input/DeathRecords.csv')
methods = pd.read_csv('../input/MethodOfDisposition.csv')
methods
method_counts = df.MethodOfDisposition.value_counts()[:2].reset_index()
method_counts.columns = ['method','number']
other = len(df)-method_counts.number.sum()
method_counts = method_counts.append({'method': 'Other', 'number': other}, ignore_index=True)
method_counts.method = method_counts.method.replace({'B': 'Burial', 'C': 'Cremation'})
sns.set_style('white')
ax = sns.barplot(data=method_counts, x='method', y='number')
plt.ylabel('')
plt.xlabel('')
ax.set_title('How many Americans were buried or cremated in 2014?')
sns.despine()
method_by_sex = pd.crosstab(df['Sex'], df['MethodOfDisposition'])[['B','C']].apply(lambda r: r/r.sum(), axis=1)*100
ax2 = method_by_sex.plot.barh(stacked=True)
plt.ylabel('')
ax2.set_title('Almost 52% men are cremated, compared to 45% women')
sns.despine()