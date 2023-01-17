# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import statsmodels.formula.api as smf
%matplotlib inline
data = pd.read_csv("../input/haberman.csv")
data.shape
data.describe()
data.columns=["Age","op_year","axi_node","surv_st"]
data.head()
data.dtypes
data.surv_st.value_counts()
data.op_year.value_counts()
data.plot(kind="scatter" , x='Age', y='op_year')
plt.show()
data.plot(kind="scatter" , x='Age', y='axi_node')
plt.show()
sn.FacetGrid(data, hue="surv_st", size=6).map(plt.scatter, "Age", "axi_node")
plt.legend(['Survived > 5 yrs', 'Survived < 5 yrs'])
plt.show()
data.plot(kind="scatter" , x='op_year', y='axi_node')
plt.show()
sn.FacetGrid(data,hue='surv_st', size=8).map(sn.distplot,'Age')
plt.legend(['Survived > 5', 'Survived < 5'])
plt.show()
sn.FacetGrid(data,hue='surv_st', size=8).map(sn.distplot,'op_year')
plt.legend(['Survived > 5', 'Survived < 5'])
plt.show()
sn.FacetGrid(data,hue='surv_st', size=8).map(sn.distplot,'axi_node')
plt.legend(['Survived > 5', 'Survived < 5'])
plt.show()
sn.boxplot(x='surv_st' , y='Age', data=data)
plt.show()
sn.boxplot(x='surv_st' , y='axi_node', data=data)
plt.show()
sn.boxplot(x='surv_st' , y='op_year', data=data)
plt.show()
data_survived_more = data[data['surv_st']==1]
data_survived_less = data[data['surv_st']==2]
counts , bin_edges = np.histogram(data_survived_more['Age'], bins=10, density=True)
pdf = counts / (sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)

plt.show()

counts , bin_edges = np.histogram(data_survived_less['Age'], bins=10, density=True)
pdf = counts / (sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)

plt.show()
counts , bin_edges = np.histogram(data_survived_more['axi_node'], bins=10, density=True)
pdf = counts / (sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)

plt.show()

counts , bin_edges = np.histogram(data_survived_less['axi_node'], bins=10, density=True)
pdf = counts / (sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)

plt.show()
