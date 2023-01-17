# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



df = pd.read_csv('../input/degrees-that-pay-back.csv')

df.head()
df.dtypes
df.columns = ['major','bgn_p50','mid_p50','delta_bgn_mid','mid_p10','mid_p25','mid_p75','mid_p90']

df
df['bgn_p50'] = df['bgn_p50'].str.replace('$','')

df['bgn_p50'] = df['bgn_p50'].str.replace(',','')
df['bgn_p50'] = pd.to_numeric(df['bgn_p50'])

df['bgn_p50'].dtype

plt.hist(df['bgn_p50'],facecolor = 'g',alpha = .75,edgecolor = 'black')

plt.xlabel('Starting Median Salaries')

plt.ylabel('Frequency')

plt.title('Starting Salaries for Major Students')

plt.show()
