# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

plt.style.use('ggplot')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

%matplotlib inline
df = pd.read_csv('../input/vgsales.csv')

df.head()
grouped = df.groupby('Year').agg({'Global_Sales': np.sum})



plt.plot(grouped)

plt.ylabel('Global Sales in Millions')

plt.ylim()

plt.show()
all_sales = df.groupby('Year').agg({'Global_Sales': np.sum, 'NA_Sales': np.sum, 'EU_Sales': np.sum, 'JP_Sales': np.sum, 'Other_Sales': np.sum})

plt.plot(all_sales.Global_Sales, label='Global')

plt.plot(all_sales.NA_Sales, label='North America')

plt.plot(all_sales.EU_Sales, label='Europe')

plt.ylabel('Sales in Millions')

plt.legend(loc='upper left')

plt.show()