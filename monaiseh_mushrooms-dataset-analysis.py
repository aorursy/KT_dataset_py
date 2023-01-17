# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/"))
mush=pd.read_csv('../input/mushrooms.csv')

# Any results you write to the current directory are saved as output.
mush.head()
mush.describe()
mush['cap-shape'].value_counts().plot.bar()
#check the numbers of each cap-shape of mushrooms
mush['odor'].value_counts().plot.bar()
#check the number of each type of odor
((mush['population'].value_counts()/mush['population'].value_counts().sum())*100).plot.pie()
#the percentage of each population over all other populations
sns.countplot(x='population',hue='class',data=mush)
#show the number of each class in each population
sns.catplot(x='cap-shape',hue='class',col='bruises',palette='Set3',kind='count',data=mush)
#the number of each cap-shape in different classes p,e sperating by if the cap-shape has bruises or not
sns.catplot(x="veil-color",hue='class', kind='count',data=mush ,palette='rainbow')