# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as mp

import numpy as np



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
HR_data=pd.read_csv("../input/HR_comma_sep.csv", sep=",",header=0) # reads in our input file
HR_data.shape # shows the makeup of the input file
HR_data.head() # shows the headers for our input file
HR_data.corr() # displays a grid of the correlation between our data
correlation = HR_data.corr()

mp.figure(figsize=(10,10))

sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='cubehelix')



mp.title('HR Data Analysis')
HR_data.hist()