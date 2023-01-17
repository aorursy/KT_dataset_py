# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))
data = pd.read_csv('../input/'+os.listdir('../input')[1])
data.head()
# Any results you write to the current directory are saved as output.
data.describe()
#NO MISSING VALUES FOUND
#print(data.isnull().sum())
sns.heatmap(data.corr(),annot = True)
#sns.distplot(data.CGPA,norm_hist=True,bins=np.linspace(6.5,10,20) )
sns.boxplot(x='Research',y='CGPA',data=data)

sns.boxplot(x='SOP',y='CGPA',data=data)
sns.pairplot(data,vars=['CGPA','GRE Score','TOEFL Score'])
