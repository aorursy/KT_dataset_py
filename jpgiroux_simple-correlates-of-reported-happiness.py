# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import seaborn as sns
import matplotlib.pyplot as plt
import os
import string
from scipy.stats import linregress
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/WVS_per_Country.csv')
codes = pd.read_csv('../input/Codebook.csv')
df.head()
df.describe()
codes.head()
def regplot(x_code, y_code, df):
    # make reg plot and label according to the codes in question
    data = df[~(getattr(df, x_code).isnull()|getattr(df, y_code).isnull())]
    y_label = codes[codes['VARIABLE']==y_code]['LABEL'].to_string()
    x_label = codes[codes['VARIABLE']==x_code]['LABEL'].to_string()
    ax = sns.regplot(x=x_code, y=y_code, data=data)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    plt.title(label)
    plt.show()
    

ylabel = str(codes[codes['VARIABLE']=='A008']['LABEL'].to_string())
results = []
for col in list(df.columns.values[5:]):
    if col[0] not in string.ascii_letters:
        continue
    if col[0] in ['S', 'X']:
        continue
    if col.endswith('Sd'):
        continue
    try:
        label = str(codes[codes['VARIABLE']==col]['LABEL'].to_string())
        data = df[~(df.A008.isnull()|getattr(df, col).isnull())]
        n = data.shape[0]
        print("{} rows with overlapping results for {}.".format(n, label))
        if n < 70:
            continue
        regr = linregress(data['A008'], data[col])
        if regr.rvalue in [0,1]: 
            print("R of {} returned for {}.".format(regr.rvalue, label))
            continue
        row = dict(
            code=col,
            regression=regr,
            label=label,
            n=n)
        # regplot('A008', col, df)
        results.append(row)
    except TypeError:
        continue
significant = [row for row in results if row['regression'].pvalue<0.05]
by_slope = sorted(significant, key=lambda x: x['regression'].slope, reverse=True)
top = by_slope[0:10]
bottom = by_slope[-10:]

for row in top:
    print(row)
    regplot('A008', row['code'], data)
for row in bottom:
    print(row)
    regplot('A008', row['code'], data)
