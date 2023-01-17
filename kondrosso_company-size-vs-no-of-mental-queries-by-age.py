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
df = pd.read_csv('../input/survey.csv')
ds = df.groupby('state')['Age']
dq = ds.describe()
from matplotlib.pyplot import plot
dq.plot(kind='barh', alpha = 0.7)
dsss
dss.pivot_table(index='no_employees', values='Age', aggfunc=['mean'])
ds = df[['no_employees', 'Age', 'Country']][df['Age'] > 0]
dsss = ds[['no_employees', 'Age', 'Country']][df['Age'] < 200]
dss1 = dsss[dsss['no_employees'] == "1-5"]
dss1c = dss1.hist(bins=5)
dss1.plot(kind ='kde', title = "Employee size: 1-5")
dss2 = dsss[dsss['no_employees'] == "6-25"]
dss2.plot(kind ='kde', title = "Employee size: 6-25")
dss2c = dss2.hist(bins=5, normed=True)
dss3 = dsss[dsss['no_employees'] == "26-100"]
dss3.plot(kind ='kde', title = "Employee size: 26-100")
dss4 = dsss[dsss['no_employees'] == "100-500"]
dss4.plot(kind ='kde', title = "Employee size: 100-500")
dss5 = dsss[dsss['no_employees'] == "500-1000"]
dss5.plot(kind ='kde', title = "Employee size: 500-1000")
dss6 = dsss[dsss['no_employees'] == "More than 1000"]
dss6.plot(kind ='kde', title = "Employee size: 1000+")
dss = df.groupby('no_employees')['Age']
dr = dss.groupby('no_employees')['Age'].mean()
du = dss.groupby('no_employees')['Age'].count()
dr.columns = ['no_employees', 'Age', 'age']
dt = pd.DataFrame(dr)
dv
dv.index_name = ['no_employees']
dv.columns = ['no_employees', 'Count']
dv
dv = pd.DataFrame(du)
dz = pd.concat([dv,dt], axis = 1)
dz
dz
dz['Count']
x = [1, 2, 3]
y = [1, 1, 1]
from matplotlib import pyplot
pyplot.scatter(dsss['no_employees'], dsss['Age'],  c='b', label = 'Count vs. Age mean')
ax = fig.add_subplot(1,1,1)
plot.scatter(dz[1:1], dz[2:2])
dt
ds.sort_values(by='Count', ascending=False)
du.plot(kind='bar')
dss.pivot_table(index='no_employees', values='Age', aggfunc=['count'])
dss
dq