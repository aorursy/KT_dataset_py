from IPython.display import display # for the multiple output

from subprocess import check_output

import pandas as pd # for data reading and pre-processing

import numpy as np # for the model building

import matplotlib.pyplot as plt # for the visualization

import seaborn as sns # for the visualization

%matplotlib inline 
# read the zillow data files.

print('all data files in zillow project:' '\n')

print(check_output(["ls","../input"]).decode("utf8"))



# read the training data file.

train16 = pd.read_csv ("../input/train_2016.csv")

# read the zillow dic file.

zdic = pd.read_excel ("../input/zillow_data_dictionary.xlsx")



print('training data shape sample:')

display(train16.head(3))

print('training data size:')

display(train16.shape)

print('The feature explanation:')

display(zdic)
# for the figure configuration

plt.figure(figsize=(20,10))

# for the plot kind,sort by the value, 

# x use the shape means the no. of the x, 

# y use the logerror values sorted

plt.scatter(range(train16.shape[0]),np.sort(train16.logerror.values))

#for the plot labels

plt.xlabel('parcels', fontsize=20)

plt.ylabel('Log Error',fontsize=20)

#hide the log of the matplotlib

plt.show()