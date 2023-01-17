# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.plotting.register_matplotlib_converters()





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/the-human-freedom-index/hfi_cc_2018.csv")

data.head()

data.shape
plt.figure(figsize=(16,6))

plt.title("Meaningless plot for showing how to use ")

sns.lineplot(data=data.iloc[:,5])

sns.set_style("darkgrid")

list(data.columns)
plt.figure(figsize = (10,20))

plt.title("Meaningless plot for showing how to use")

sns.lineplot(data = data["hf_score"],label = "hf_score")

plt.xlabel("index")

sns.set_style("whitegrid")
plt.figure(figsize =(25,15))

plt.title("Bar plot")

sns.barplot(x = data.countries[0:5] , y =  data.ef_regulation )

plt.xlabel("country")

plt.ylabel("regulation")
plt.figure(figsize = (10,20))

plt.title("scatter plot")

sns.scatterplot(x = data.countries , y =  data.ef_regulation )

plt.figure(figsize = (10,20))

plt.title("Meaningless plot for showing how to use")

sns.regplot(x = data.pf_religion , y =  data.ef_regulation )



sns.scatterplot(x=data.pf_religion, y=data.ef_regulation, hue=data.ef_regulation_business) # in this case we cannot see clearly what it shows. For see that hue data should be yes/no or 1/0 like gender.
#this is not work well in this data. sns.lmplot(x="pf_religion", y="ef_regulation", hue="ef_regulation_business" ,data = data)
sns.swarmplot(x=data.pf_religion, y=data.ef_regulation )
#data.pf_ss_disappearances_disap.value_counts()



sns.distplot(a = data.pf_ss_disappearances_disap,kde = False ) 

sns.kdeplot(data = data.pf_ss_disappearances_disap, shade = True)
sns.jointplot(x=data.pf_religion, y=data.ef_regulation,kind = "kde") # there are options that  “scatter” | “reg” | “resid” | “kde” | “hex”