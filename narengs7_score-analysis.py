# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
df = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')

df.head()
print(df.shape) # show's there are 215 samples and 15 features

df.describe() # Salary has count 148, thus it has NaN Values, which was because they are not placed, so their salary can be assigned to zero
df.corr()
#  Draw a histogram for degreep.

degree_p_vals = df["degree_p"]

w = 2

n = int((degree_p_vals.max() - degree_p_vals.min())/w)

degree_p_vals.hist(bins=n)


degree_p_vals_sp = df[df["status"]=="Placed"]["degree_p"]

degree_p_vals_snp = df[df["status"]!="Placed"]["degree_p"]

degree_p_vals_sp.hist(bins=n,color='lightgreen',alpha=0.7,label="Placed")

degree_p_vals_snp.hist(bins=n,color='firebrick',alpha=0.5,label="Not Placed")

plt.legend()
selector = 'gender'

selector_val="M"

degree_p_vals_sp = df[df[selector]==selector_val]["degree_p"]

degree_p_vals_snp = df[df[selector]!=selector_val]["degree_p"]

degree_p_vals_sp.hist(bins=n,color='lightgreen',alpha=0.7,label="Male")

degree_p_vals_snp.hist(bins=n,color='firebrick',alpha=0.5,label="Female")

plt.legend()
df["degree_p"].plot(kind='box')
selector = 'gender'

selector_val="M"

degree_p_vals_M = df[df[selector]==selector_val]["degree_p"]

degree_p_vals_FM = df[df[selector]!=selector_val]["degree_p"]



fig, axs = plt.subplots(1,2)

fig.suptitle('Box Plot for Degree % based on gender')



degree_p_vals_M.plot(kind='box',ax=axs[0],title='Male')

degree_p_vals_FM.plot(kind='box',ax=axs[1],title='Female')

plt.legend()
from sklearn.preprocessing import LabelEncoder

df_f = df[["workex","gender","degree_p"]].copy()

fig, axs = plt.subplots(1,4)

index = 0

for title,data in df_f.groupby(['workex','gender']):

    title = list(title)

    if title[1]=="F":

        title[1] = "Female"

    else:

        title[1] = "Male"

    

    if 'Y' in title[0]:

        title[0] = ' \nwith \nwork exp'

    else:

        title[0] = ' \nwithout \nwork exp'

    

    data.plot(kind='box',ax=axs[index],title=title[1]+title[0])

    index = index + 1
import seaborn as sns

sns.distplot(df['degree_p'])
# Differentiate based on status 

import seaborn as sns

fig, axs = plt.subplots(1,2)

index = 0

for title,data in df.groupby('status'):

    print(title)

    sns.distplot(data['degree_p'],ax=axs[index],axlabel=title)

    index = index+1

plt.legend()
# Differentiate based on status 

import seaborn as sns

fig, axs = plt.subplots(1,2)

index = 0

for title,data in df.groupby('status'):

    print(title)

    sns.distplot(data['degree_p'], hist=True, kde=True, 

                 bins=n, ax=axs[index],

                 hist_kws={'edgecolor':'black'},

                 kde_kws={'color': 'blue'})

    index = index+1