# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set_style('darkgrid')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



#Importing dataset

df = pd.read_csv('../input/inventory/inventory_clean.csv')
# Summary of dataframe

df.describe()
#Lead time is a good candidates for histograms

sns.distplot(df['lead_time'],bins=50, kde=False)

plt.title('Distribution of Lead Time')

plt.xlabel('Lead Time (days)')

plt.show()
#Same histogram using matplotlib

plt.hist(df['lead_time'], bins=50)

plt.title('Distribution of Lead Time')

plt.xlabel('Lead Time (days)')

plt.show()
#normalized historgram with guassian kerner density estimate

sns.distplot(df['lead_time'],norm_hist=True, bins=50, kde=True)

plt.title('Distribution of Lead Time')

plt.xlabel('Lead Time (days)')

plt.show()