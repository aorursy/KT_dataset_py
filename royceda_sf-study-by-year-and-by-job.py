%matplotlib inline
# # This Python 3 environment comes with many helpful analytics libraries installed
# # It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# # For example, here's several helpful packages to load in 
# 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #plotting lib

import scipy as sp 
from scipy.interpolate import interp1d #interpolate
# 
# # Input data files are available in the "../input/" directory.
# # For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# 
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# 
# # Any results you write to the current directory are saved as output.
# 
df = pd.read_csv("../input/Salaries.csv");

# 
#df.dtypes
#df.describe()
df


# # sort by salaries
#df.iloc[3:5,0:7]
#df.sort_index(axis=1, ascending=False);
#df.sort_values(by='TotalPay');


df2 = df.loc[:,['Year', 'TotalPay' , 'BasePay']].sort_values(by="TotalPay", ascending=False);
#df2=df2.ix[1:10]
df3=df.loc[:,['Year','TotalPay']]
df3
dfp = df3.groupby('Year').mean();
#dfp = dfp.sort_values(by='TotalPay', ascending=False)
plt1 = dfp.plot(kind='bar');

print(plt1)
df3=df.loc[:,['TotalPayBenefits', 'JobTitle']]
df3
dfp = df3.groupby('JobTitle').mean();
dfp2=dfp.ix[1:25]
plt2 = dfp2.plot(kind='bar');

print(plt2)
#dfb = df2.query('Year == 2013')
print(df2.boxplot(by='Year'))
dfb = df2.query('TotalPay < 210000')
#dfb=dfb.query('Year == 2013')
print(dfb.boxplot(by='Year'))
df3=df.loc[:,['TotalPay', 'JobTitle']].query('TotalPay > 350000')
df3
dfp1 = df3.groupby('JobTitle').mean();
dfp1 = dfp1.sort_values(by='TotalPay', ascending=False)
dfp1=dfp.ix[1:25]
plt1 = dfp1.plot(kind='bar');

print(plt1)
df12=df.loc[:,['Year','TotalPay']].query('TotalPay < 220000')
df12
dfp = df12.groupby('Year').mean();


matrix = dfp.as_matrix()
x1 = sorted([2011,2012,2013,2014])

x = np.array(x1)
y = np.array(matrix)
z = [y[0][0],y[1][0],y[2][0],y[3][0]]

new_length=60;
new_x = np.linspace(x.min(), x.max(), new_length);
new_y = sp.interpolate.interp1d(x, z, kind='cubic')(new_x);

print(plt.plot(new_x, new_y));

#print(dfp.interpolate(method='cubic', axis=1, limit=2018, inplace=False, limit_direction='forward'))

func = sp.interpolate.splrep(new_x, new_y, s=3);
pp = sp.interpolate.spltopp(func[0][1:-1],func[1],func[2])
# Print the coefficient arrays, one for cubed terms, one for squared etc
print(pp.coeffs)

df3=df.loc[:,['TotalPay', 'JobTitle']].query('TotalPay < 1000')
df3
dfp = df3.groupby('JobTitle').mean();
dfp = dfp.sort_values(by='TotalPay', ascending=False)
dfp=dfp.ix[1:25]
plt1 = dfp.plot(kind='bar');

print(plt1)

df2 = df.query('Year == 2011')
df3=df2.loc[:,['TotalPay', 'JobTitle']].query('TotalPay > 210000')
df3
dfp = df3.groupby('JobTitle').mean();
dfp = dfp.sort_values(by='TotalPay', ascending=False)
dfp=dfp.ix[1:25]
plt1 = dfp.plot(kind='bar');

print(plt1)





