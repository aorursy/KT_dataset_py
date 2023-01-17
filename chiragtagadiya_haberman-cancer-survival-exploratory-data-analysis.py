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
# EDA : Haberman's Survival Data Set
# The dataset contains cases from a study that was conducted between 1958 and 1970 at the University of Chicago's Billings Hospital on the survival of patients who had undergone surgery for breast cancer.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
haberman = pd.read_csv("/kaggle/input/habermans-survival-data-set/haberman.csv")
haberman.columns=["age","year","nodes","status"]
haberman.head()
haberman.describe()
haberman.shape

haberman["status"].value_counts()

haberman.plot(kind="scatter",x="age",y="status",legend=True)
# haberman.plot?
sns.set_style("whitegrid")
sns.FacetGrid(haberman,hue="status",size=5)\
.map(plt.scatter,"age","status")\
.add_legend();
plt.show()
## Pair Plot
plt.close()
# sns.pairplot

sns.pairplot(haberman,hue="status",vars=["age","nodes","year"],size=3)
plt.show()
class1 = haberman.loc[haberman["status"] == 1]
class2 = haberman.loc[haberman["status"]==2]
plt.plot(class1["nodes"],0+np.zeros_like(class1["nodes"]),"o",label="status1")

plt.plot(class2["nodes"],1+ np.zeros_like(class2["nodes"]),"*",label="status2")
plt.xlabel("nodes")
plt.ylabel("nodes")
plt.legend()
plt.title("1-D Scatter plot")
plt.show()





sns.FacetGrid(haberman,hue="status",size=5).map(sns.distplot,"age").add_legend()

sns.FacetGrid(haberman,hue="status",aspect=2).map(sns.distplot,"nodes").add_legend()
# person with 0 lymph nodes and status 1
haberman_data = haberman[(haberman["nodes"] == 0 ) & (haberman["status"] == 1)]
haberman["status"].value_counts()
sns.FacetGrid(haberman, hue="status", size=5) \
   .map(sns.distplot, "year") \
   .add_legend();
plt.show();

sns.FacetGrid(haberman, hue="status", size=5) \
   .map(sns.distplot, "age") \
   .add_legend();
plt.show();


status1 = haberman[haberman["status"] == 1]
status2 = haberman[haberman["status"] ==  2]
# status 1
counts,bin_edges=np.histogram(status1["nodes"],bins=10,density=True)
pdf = counts/sum(counts)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label="pdf plot status 1")
plt.plot(bin_edges[1:],cdf,label="CDF plot of status 1")
plt.title("PDF and CDF of Patient's nodes having status 1'")
plt.xlabel("Patient's lymph node count for status 1")

plt.legend()

# status 2
counts,bin_edges=np.histogram(status2["nodes"],bins=10,density=True)

pdf = counts/sum(counts)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label="pdf plot status 2")
plt.plot(bin_edges[1:],cdf,label="CDF plot of status 2")
plt.title("PDF and CDF of Patient's nodes having status 2'")
plt.xlabel("Patient's lymph node count for status 2")

plt.legend()
plt.show()

print("\nQuantiles Status 1:")
np.median(status1["nodes"])
np.median(status1["age"])
print(np.percentile(status1["age"],np.arange(0, 100, 10)))
print(np.percentile(status1["nodes"],np.arange(0, 100, 10)))


print("\nQuantiles Status 2:")
np.median(status1["nodes"])
np.median(status1["age"])
print(np.percentile(status2["age"],np.arange(0, 100, 25)))
print(np.percentile(status2["nodes"],np.arange(0, 100, 25)))
print("90th percentile status2 ")
print(np.percentile(status2["nodes"],90))

print(np.percentile(status2["age"],90))
print("90th percentile status1 ")
print(np.percentile(status1["nodes"],90))

print(np.percentile(status1["age"],90))

sns.boxplot(x="status",y="age",data=haberman)
sns.boxplot(x="status",y="nodes",data=haberman)
#print 75 the percentile of class 1 and  class 2 data

print(np.percentile(class1["nodes"],75))
print(np.percentile(class2["nodes"],75))










