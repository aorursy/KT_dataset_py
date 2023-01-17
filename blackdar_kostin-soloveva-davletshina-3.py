# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sea



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/train.csv')

data.head(900)
sea.countplot(data['Survived'])
sea.jointplot(data['Age'], data['Survived'], kind = 'hex')
import matplotlib.pyplot as plt



sea.distplot(data[(data.Age.isnull() == False) & (data.Survived == True)]['Age'], label='Survived')

sea.distplot(data[(data.Age.isnull() == False) & (data.Survived == False)]['Age'],  label='Died')



plt.legend()
sea.distplot(data[data.Age.isnull() == False]['Survived'])