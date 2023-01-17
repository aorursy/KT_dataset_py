# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/vgsales.csv')
data.info
data.corr()
f,ax = plt.subplots(figsize=(18,18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f', ax=ax)
plt.show()
data.head(10)
data.columns
#Line Plot
#color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = style of line
data.JP_Sales.plot(kind = 'line', color = 'g', label = 'JP_Sales', linewidth = 1, alpha = 0.5, grid = True, linestyle = ':')
data.Other_Sales.plot(color = 'r', label = 'Other_Sales',linewidth = 1, alpha = 0.5, grid = True, linestyle = '-.' )
plt.legend(loc='upper right')
plt.xlabel('JP_Sales')
plt.ylabel('Other_Sales')
plt.title('Years Of The NA Sales')
plt.show
plt.scatter(data.Year,data.NA_Sales, color='red',alpha=0.5)
plt.show
data.Year.plot(kind = 'hist', bins = 50, figsize = (10,10))
plt.show()
data.Year.plot(kind = 'hist',bins = 50)
plt.clf()
data = pd.read_csv('../input/vgsales.csv')
series = data['NA_Sales']
print(type(series))
data_frame = data[['NA_Sales']]
print(type(data_frame))
x = data['NA_Sales']>20
data[x]
data[np.logical_and(data['NA_Sales']>15, data['EU_Sales']>10 )]
data[(data['NA_Sales']>15) & (data['EU_Sales']>10)]





