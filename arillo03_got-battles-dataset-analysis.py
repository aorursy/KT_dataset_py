# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sn#for visuals

sn.set(style="white", color_codes=True)#customizes the graphs

import matplotlib.pyplot as mp #for visuals

%matplotlib inline

import warnings #suppress certain warnings from libraries

warnings.filterwarnings("ignore")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

battles_data = pd.read_csv("../input/battles.csv")

battles_data

battles_data.shape
battles_data.head(11)
battles_data.corr()
#Building the plot

correlation=battles_data.corr()

mp.figure(figsize=(10,10))

sn.heatmap(correlation, vmax=1, square=True,annot=True,cmap='cubehelix')

mp.title('Correlation between each feature',fontsize=20,fontweight="bold")

mp.show()