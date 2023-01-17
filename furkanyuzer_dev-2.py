# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.""
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/Iris.csv")

data.info()
data.columns
data.describe()
data.head()
data.boxplot(column="SepalLengthCm",by="Species",figsize=(14,14))
data1=data.head()
melted_data=pd.melt(frame=data1,id_vars=("Id"),value_vars=("SepalLengthCm","SepalWidthCm"))
melted_data
data_x=data.head()
data_y=data.tail()
data_xy=pd.concat([data_x,data_y],axis=0)
data_xy
data.dtypes
data['SepalLengthCm'] = data['SepalLengthCm'].astype('int')
data.dtypes

#NaN değer var mı bakıyoruz
data["SepalLengthCm"].value_counts(dropna =False)
#eğer NaN değr bulsaydık aşşağıdaki kod ile kaldırabilirdik
#data["SepalLengthCm"].dropna(inplace = True)
assert  data['SepalLengthCm'].notnull().all()
#kontrol ediyoruz
data.SepalLengthCm.plot(kind = 'line', color = 'b',label = 'SepalLengthCm',linewidth=1,alpha = 1,grid = True,linestyle = ':',figsize=(13,13))
data.PetalLengthCm.plot(color = 'r',label = 'PetalLengthCm',linewidth=1, alpha = 1,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()