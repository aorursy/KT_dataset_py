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



# Any results you write to the current directory are saved as output.

dataset=pd.read_csv("../input/DigiDB_digimonlist.csv")

#dataset.describe

fig,axes=plt.subplots(nrows=2,ncols=2,figsize=(24,20))

fig.subplots_adjust(wspace=1,hspace=1)



sns.countplot(x="Attribute",data=dataset,ax=axes[0,0])

axes[0,0].set_title("Using Seaborn".upper())



dataset['Attribute'].value_counts().plot(kind='bar',ax=axes[0,1])

axes[0,1].set_title("USING DATAFRAME")



z=dataset['Attribute'].value_counts()

x=pd.DataFrame(z)

axi=x.T.plot(kind='bar',stacked='True',ax=axes[1,0],title="STACKED BAR PLOT")

axi.set_xticklabels([])

axi.set_xlabel("STACKED COMPARISON")



z.plot.pie(y=1,ax=axes[1,1])

axes[1,1].set_title("PIE REPRESENTATION")





                    