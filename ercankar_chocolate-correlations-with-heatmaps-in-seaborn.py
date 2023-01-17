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
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
my_data= pd.read_csv('../input/flavors_of_cacao.csv')
my_data.head()
my_data.describe()
my_data.info()
my_data.columns = my_data.columns.str.lower().str.replace('\n', '_') # we used string function (str)
my_data.columns
my_data.head(10) #default is 5 rows
my_data.cocoa_percent= my_data.cocoa_percent.str.replace('%','')
my_data.head()
%timeit my_data.tail() #This is a convenient tool which runs multiple loops of the operation and reports it’s best performance time.
my_data.cocoa_percent = my_data.cocoa_percent.astype(float)
my_data.info()
my_data.dtypes
my_data.head()
my_data.corr()
plt.figure(figsize=(10, 9))

sns.heatmap(my_data.corr(),cmap='BuPu',vmin=0,vmax=1,annot=True,annot_kws={'size':15},square=True,linewidths=1, linecolor='yellow')

#cmap adjusts the colormap used.

#Adjust the lower and upper contrast bounds with vmin and vmax. 

#Label the rectangles with annot=True, which also chooses a suitable text color

#annot_kws to change the text properties of the annotations. Like font_size

#linewidths = Width of the lines that will divide each cell.

#linecolor = Color of the lines that will divide each cell.

plt.show() 
fil_data = my_data[(my_data.cocoa_percent > 70) & (my_data.cocoa_percent < 80)]
fil_data.head(10)