# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/train.csv')
age = sns.scatterplot(x="Age", y="Survived", data=data)
embarked = sns.barplot(x="Embarked", y="Survived", data=data)
pclass = sns.barplot(x="Pclass", y="Survived", data=data)
gender = sns.barplot(x="Sex", y="Survived", data=data)


