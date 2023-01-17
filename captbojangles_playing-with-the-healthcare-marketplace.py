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
import matplotlib.pyplot as plt
df = pd.read_csv('../input/Rate.csv')
df1 = df[['Age', 'IndividualRate']]
#Replacing any non-numerics with NaN
df1.Age = df1.Age._convert(numeric = True)
#Dropping NaN values
df1 = df1.dropna()
#The convert turns our Age group into floats. Converting them back into integers
df1.Age = df1.Age.astype(int)
#We have some major outliers in the 10,000 region lets isolate the "meat" of the data
df1 = df1[df1.IndividualRate < 1600]
#Lets look at a histogram of our Individual Rate data
df1.IndividualRate.hist(bins = 300)
#It looks like we have 2 distributions. One from the 100 - 1500 range and one from the 0-100 range
#There appears to be a lot of ppl who are paying around 100. Lets ignore that for now as well
dfa = df1[df1.IndividualRate > 105]
dfa.IndividualRate.hist(bins = 200)
plt.xlim(105, 1500)
plt.ylim(0, 175000)
plt.title('Individual Rate Histogram (105-1500)')
plt.ylabel('Frequency/Count')
plt.xlabel('Individual Rate')
#Here is a good isolated plot of the 105-1500 range
#Lets look at some stats for this bad boy as well
dfa.IndividualRate.describe()

#Now lets take a look at the distribution from 0-60
dfb = df1[df1.IndividualRate < 60]
dfb.IndividualRate.hist(bins = 100)
plt.xlim(5, 60)
plt.ylim(0, 150000)
plt.title('Individual Rate Histogram (105-1500)')
plt.ylabel('Frequency/Count')
plt.xlabel('Individual Rate')
#And some stats for this bad boy as well
dfb.IndividualRate.describe()
