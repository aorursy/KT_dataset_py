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
import numpy as np # Basic linear algebra functions

import pandas as pd # Structured data operations and manipulations

import seaborn as sn # Statistical data visualization

import matplotlib.pyplot as plt # Python defacto plotting library

%matplotlib inline 

import warnings
data = pd.read_csv("../input/HR_comma_sep.csv")
data
data.shape
data.head(10)
data['salary'].value_counts()
data['sales'].value_counts()  
data.describe()
data.columns
data.corr()
# Set up the matplotlib figure

plt.rcParams['figure.figsize']=[12.0, 10.0]

plt.title('Pearson Correlation')

# Draw the heatmap using seaborn

sn.heatmap(data.corr(),linewidths=0.5,vmax=1.0, square=True, cmap="BrBG", linecolor='black', annot=True)
plt.figure(figsize=(12,8))

sn.distplot(data["number_project"].values, bins=15, kde=True)

plt.xlabel("number_project", fontsize=12)

plt.ylabel("satisfaction_level", fontsize=12)

plt.title("Relationship Between Number of Project and Satisfaction Level")

plt.show() 