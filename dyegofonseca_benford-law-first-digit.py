# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # seaborn to create grafic

import matplotlib.pylab as plt





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Collecting Argentina's data base

data = pd.read_csv('../input/argentina.csv')

data.head()
data.STREET.value_counts().head()
# Verifying data information

data.info()
data['NUMBER'].head()
data.NUMBER = data.NUMBER.astype(str)

data.NUMBER = data.NUMBER.str.strip(' ')
data['DIGITO'] = data.NUMBER.map(lambda x: x[0])

data.head()
[str(x) for x in range(10)]
index_drop = data[data.DIGITO == '0'].index.values

index_drop

data.drop(index_drop, axis=0, inplace=True)
for num in [str(x) for x in range(10)]:

    print(num, data[data['DIGITO'] == num].shape[0], data[data['DIGITO'] == num].shape[0]/ data.shape[0])
experemental_values = data.DIGITO.value_counts()

experemental_values
# Ones quantity 

data[data.DIGITO == '1'].shape[0]
# Verifying if there is 0

data['DIGITO'].loc[data['DIGITO'] == '0'].count()
data.shape
# Filtering the data frame with numbers different from 0

num_address = data[data['DIGITO'] != '0']

num_address.shape
num_address['DIGITO'].head()
len(num_address)
import math

array = [float(i) for i in range(1, 10)]

percent_bf = [math.log10(1 + 1 / d) for d in array]

percent_bf
fd_teorico = []

for i in range(len(percent_bf)):

    fd_teorico.append(len(num_address) * percent_bf[i]) 

fd_teorico
fd_plot = []

for i in range(len(percent_bf)):

    fd_plot.append(fd_teorico[i]/ len(num_address))

print(fd_plot)
size_list = [i for i in range(1, 10)]

size_list
plt.bar(size_list,fd_plot)

plt.show
sns.barplot(x= size_list, y= fd_plot)
type(num_address)
new_num_address = num_address['DIGITO'].astype(str)

print(type(new_num_address))
new_num_address.loc[new_num_address == '0'].count()
new_num_address = new_num_address[new_num_address != '0']
new_num_address.loc[new_num_address == '0'].count()
type(new_num_address[0])
new_num_address.str.contains(' ').head()
new_num_address[557750][0]
new_num_address.head()
fd_db = new_num_address.str[0]

fd_db.head()

fd_db.astype(str).head()
count = fd_db.value_counts()
len(count)
# Check len of count

print(len(count))

# Confirming there is no 0 value in the pd serie

new_num_address.loc[new_num_address == '0'].count()
fd_evidence = []

for i in range(len(count)):

    fd_evidence.append(float(count[i] / len(num_address)))

    print(i)

#del(fd_evidence[-1])

fd_evidence
import scipy.stats

# Chi square test

freq_expected = fd_plot

freq_obs = fd_evidence



qui_quadrado = ((np.array(freq_obs) - np.array(freq_expected)) ** 2 / np.array(freq_expected))



print(qui_quadrado.sum())

print(qui_quadrado)



scipy.stats.chisquare(freq_obs, freq_expected)
plt.bar(size_list,fd_plot, color ='b',width=0.8, label="Theorical velues")

plt.plot(size_list, qui_quadrado, 'b', label = "Chi square", color='y')

plt.bar(size_list,fd_evidence, color='red',width = 0.4, label="Argentina's houses first digits")

plt.xticks(size_list)

plt.xlabel("Digit")

plt.ylabel("Probability")

plt.legend()

plt.show





fig = plt.figure(figsize=(8,4), dpi=100)

fig, axes = plt.subplots()

axes.plot(size_list, fd_plot, 'b-*', label='Theorical velues')

axes.plot(size_list, fd_evidence, 'r-*', label="Argentina's houses first digits")

axes.plot(size_list, qui_quadrado, 'y-*', label="Chi squared")

axes.set_title('Benford Law - Houses numbers AR')

axes.legend(loc=0)