# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import seaborn as sn 

import matplotlib.pyplot as mp

%matplotlib inline 

import warnings 
data = pd.read_csv("../input/movie_metadata.csv")
data

df = pd.DataFrame(data)

new =df.dropna() #remove NaN
new
data.shape

new.shape
new.head(10)
new.describe()
new.corr()
# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(12, 10))

plt.title('Pearson Correlation of Movie Features')

# Draw the heatmap using seaborn

sn.heatmap(new.corr(),linewidths=0.25,vmax=1.0, square=True, cmap="YlGnBu", linecolor='black', annot=True)