# Importing Libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



dataset = pd.read_csv('../input/chopstick-effectiveness.csv')

dataset.sample(10)
#Diving the dataset into vectors for easier use

data_without_ind = dataset.iloc[:,[0,2]]

scores = dataset.iloc[:, 0].values

sizes_array = dataset.iloc[:, 2].values

sizes = np.sort(np.array(list(set(dataset.iloc[:, 2].values))))
#Starting exploratory analysis

sizes_scores=np.zeros((scores.size,sizes.size))

size_grouped = data_without_ind.groupby(['Chopstick.Length'])

sizes_for_box = []

for size in sizes:

    sizes_for_box.append(size_grouped.get_group(size)['Food.Pinching.Efficiency'])







#Plotting the mean Effectiveness of each length

bx = sns.boxplot(data =sizes_for_box)

bx.set(xlabel = 'Chopstick Length', ylabel='Chopstick Effectiveness')

sns.plt.show(block=False)



#Mean by size

mean_by_size = size_grouped.mean()

mean_by_size