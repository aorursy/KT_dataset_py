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
%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

pizza_data = pd.read_csv("../input/8358_1.csv")
pizza_data.head()
longi = (min(pizza_data['longitude']),max(pizza_data['longitude']))

lati = (min(pizza_data['latitude']),max(pizza_data['latitude']))
ax = plt.scatter(pizza_data['longitude'].values,pizza_data['latitude'].values,color='blue',s=10,alpha=0.5)

plt.show()
pizzas = pizza_data['menus.name'].value_counts()
pizzas
pizza_data.info()
sns.barplot(x=pizzas[0:10],y=pizzas[1:10])

plt.show()
pizzas[:10].plot.bar()
pizzas_city = pizza_data['city'].value_counts()
pizzas_city
pizzas_city[:10].plot.bar()
f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(pizza_data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)