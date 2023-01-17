# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



data = pd.read_csv('../input/Health_AnimalBites.csv') 



colorData = data.color.value_counts()



print(colorData[0:5])



proc_data = data[data.color=='BLACK'].BreedIDDesc.value_counts()



dogSpecies = list(proc_data.keys())



corrCount = list(proc_data[proc_data.keys()])



dogSpecies = dogSpecies[0:5]

corrCount = corrCount[0:5]



y_pos = np.arange(len(dogSpecies))



plt.bar(y_pos,corrCount)



plt.xticks(y_pos,dogSpecies)



plt.show()
proc_data = data[data.color=='BROWN'].BreedIDDesc.value_counts()



dogSpecies = list(proc_data.keys())



corrCount = list(proc_data[proc_data.keys()])



dogSpecies = dogSpecies[0:5]

corrCount = corrCount[0:5]



y_pos = np.arange(len(dogSpecies))



plt.bar(y_pos,corrCount)



plt.xticks(y_pos,dogSpecies)



plt.show()
proc_data = data[data.color=='WHITE'].BreedIDDesc.value_counts()



dogSpecies = list(proc_data.keys())



corrCount = list(proc_data[proc_data.keys()])



dogSpecies = dogSpecies[0:5]

corrCount = corrCount[0:5]



y_pos = np.arange(len(dogSpecies))



plt.bar(y_pos,corrCount)



plt.xticks(y_pos,dogSpecies)



plt.show()
proc_data = data[data.color=='BLK WHT'].BreedIDDesc.value_counts()



dogSpecies = list(proc_data.keys())



corrCount = list(proc_data[proc_data.keys()])



dogSpecies = dogSpecies[0:5]

corrCount = corrCount[0:5]



y_pos = np.arange(len(dogSpecies))



plt.bar(y_pos,corrCount)



plt.xticks(y_pos,dogSpecies)



plt.show()
proc_data = data[data.color=='TAN'].BreedIDDesc.value_counts()



dogSpecies = list(proc_data.keys())



corrCount = list(proc_data[proc_data.keys()])



dogSpecies = dogSpecies[0:5]

corrCount = corrCount[0:5]



y_pos = np.arange(len(dogSpecies))



plt.bar(y_pos,corrCount)



plt.xticks(y_pos,dogSpecies)



plt.show()