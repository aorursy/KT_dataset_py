# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))

# after we print the code upside we see 2 files, so lets bring t dataframe

rdl = pd.read_csv('../input/deputies_dataset.csv',delimiter=',')

dds = pd.read_csv('../input/dirty_deputies_v2.csv',delimiter=',')





# after we explore the data we see some numerical e and some categorical data

# lets see the distribution of the values of the receipt

# first we recive the seaborn axis-object returned in a variable ao

ao = sns.distplot(rdl['receipt_value'], kde=True, rug=False) 

# set the x label and the y label

ao.set(xlabel='Receipt Value', ylabel='Number of Receipts')

# set the title

ao.set_title('Valores')

# finaly plot

plt.show()







# Any results you write to the current directory are saved as output.