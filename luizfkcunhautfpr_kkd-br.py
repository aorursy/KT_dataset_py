import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/train.csv", usecols=['Id','field','age','type','harvest_year','harvest_month','production'])

data.head()
#data['field']
#data.values.tolist()
data.values.tolist()
n, bins, patches = plt.hist(data['age'].values, 100, density=True, facecolor='g', alpha=0.75)
plt.xlabel('Idade')
plt.ylabel('Frequência')
plt.title('Histograma / Idade')
plt.grid(True)
plt.show()
n, bins, patches = plt.hist(data['type'].values,bins=[0,1,2,3,4,5,6,7], density=True, facecolor='g', alpha=0.75)
plt.xlabel('Tipo')
plt.ylabel('Frequência')
plt.title('Histograma / tipo')
plt.grid(True)
plt.show()
n, bins, patches = plt.hist(data['production'].values, 100, density=True, facecolor='g', alpha=0.75)
plt.xlabel('Tipo')
plt.ylabel('Produção')
plt.title('Histograma / Produção')
plt.grid(True)
plt.show()
plt.scatter(data['production'].values, data['type'].values, c="g", alpha=0.5,label="A")
plt.xlabel("Produção")
plt.ylabel("Tipo")
plt.legend(loc=2)
plt.show()
plt.scatter(data['production'].values, data['age'].values, c="g", alpha=0.5,label="A")
plt.xlabel("Produção")
plt.ylabel("Idade")
plt.legend(loc=2)
plt.show()
plt.scatter(data['production'].values, data['field'].values, c="g", alpha=0.5,label="A")
plt.xlabel("Produção")
plt.ylabel("Field")
plt.legend(loc=2)
plt.show()
plt.scatter(data['production'].values, data['harvest_year'].values, c="g", alpha=0.5,label="A")
plt.xlabel("Produção")
plt.ylabel("HarvestYear")
plt.legend(loc=2)
plt.show()
plt.scatter(data['production'].values, data['harvest_month'].values, c="g", alpha=0.5,label="A")
plt.xlabel("Produção")
plt.ylabel("HarvestMonth")
plt.legend(loc=2)
plt.show()
plt.scatter(data['field'].values, data['harvest_month'].values, c="g", alpha=0.5,label="A")
plt.xlabel("Field")
plt.ylabel("HarvestMonth")
plt.legend(loc=2)
plt.show()
plt.scatter(data['field'].values, data['harvest_year'].values, c="g", alpha=0.5,label="A")
plt.xlabel("Field")
plt.ylabel("HarvestYear")
plt.legend(loc=2)
plt.show()
plt.boxplot([data['production'].values],labels=['Produção'])
plt.show()
plt.boxplot([data['harvest_year'].values],labels=['HarvestYear'])
plt.show()
plt.boxplot([data['harvest_month'].values],labels=['HarvestMonth'])
plt.show()

