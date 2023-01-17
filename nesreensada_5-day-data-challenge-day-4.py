# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

data = pd.read_csv("../input/Health_AnimalBites.csv")
# Any results you write to the current directory are saved as output.
data.head()
data.columns
set(data.BreedIDDesc)
sns.countplot(x="GenderIDDesc", data= data);
plt.title('Gender of animals that bited humans')
plt.xlabel('Gender')
plt.ylabel('No. of bite cases reported')
plt.xticks(rotation=-45)

plt.show()
sns.countplot(x="SpeciesIDDesc", data= data);
plt.title('Species of animals that bited humans')
plt.xlabel('Species')
plt.ylabel('No. of bite cases reported')
plt.xticks(rotation=-45)

plt.show()
bitesFreq = data['GenderIDDesc'].value_counts()
plt.bar(bitesFreq.index,bitesFreq.values)
plt.title('Gender bites number')
