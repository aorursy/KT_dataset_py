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

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

survey = pd.read_csv('../input/Shower Survey.csv')
survey.head(8)
survey.describe()
sns.countplot(survey['How important is reducing water consumption to you?'])

x_labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

sns.set(rc={'figure.figsize':(10,10)})
g = sns.factorplot("How long is your average shower?", data=survey, aspect=1.5, kind="count", color="b")

g.set_xticklabels(rotation = 30)

sns.set(rc={'figure.figsize':(10,10)})