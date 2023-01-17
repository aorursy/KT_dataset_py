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

state_labels = pd.read_csv("../input/state_labels.csv")

state_labels.head()
state_labels.info()
import matplotlib.pyplot as plt

plt.figure(figsize = (20, 7))

plt.plot(state_labels['Population_density'], state_labels['Population'], marker='D')

plt.xlabel('Population density')

plt.ylabel('Population')

plt.title('Population density Vs population ')
import numpy as np

np.max(state_labels['Population'])
np.min(state_labels['Population'])
plt.figure(figsize=(20, 8))

plt.scatter(state_labels['Population_density'], state_labels['Population'])

plt.xlabel('Population density')

plt.ylabel('Population')

plt.title('Population density Vs population ')
import seaborn as sns

plt.figure(figsize=(15, 7))

sns.swarmplot(x='StateName', y='Population', data=state_labels)

plt.xticks(rotation=90)
plt.figure(figsize=(15, 5))

sns.swarmplot(x='StateName', y='Population_density', data=state_labels)

plt.xticks(rotation=90)
plt.figure(figsize=(15, 9))

plt.bar(state_labels['StateName'], state_labels['Population'])

plt.xticks(rotation=90)

plt.ylabel('Population')
plt.figure(figsize=(15, 6))

plt.bar(state_labels['StateName'], state_labels['Population_density'])

plt.xticks(rotation=90)
plt.figure(figsize=(15, 7))

sns.swarmplot(state_labels['StateName'], state_labels['State'])

plt.xticks(rotation=90)