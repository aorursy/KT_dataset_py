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
df = pd.read_csv('../input/creditcard.csv')
df.head(10)
df.info()
cls1 = df[df['Class'] == 1]['Class'].count()

cls2 = df[df['Class'] == 0]['Class'].count()



print ("total number of fraud",cls1)

print ("total number of not fraud",cls2)
cdf.corr()
ftime = df[df['Class'] == 1]

#print (ftime)

sns.distplot(ftime['Time'], kde=False, bins=50)

sns.sc
sns.regplot(data=df, x="Time", y="Amount")
sns.regplot(data=ftime,x="Time", y="Amount")