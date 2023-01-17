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

import matplotlib.pyplot as plt # Python defacto plotting library

import seaborn as sns # More snazzy plotting library

%matplotlib inline
f=pd.read_csv("../input/HR_comma_sep.csv")
f.head()
f.shape
f.corr()
correlation = f.corr()

plt.figure(figsize=(10,10))

sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='cubehelix')



plt.title('Employee Leave Analysis')
f.hist()
chart = f.groupby(['sales','time_spend_company']).sum()['average_montly_hours']

chart.plot(figsize=(15,10))

chart.plot()