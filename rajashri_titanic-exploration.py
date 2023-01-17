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
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train  = pd.read_csv("../input/train.csv")
#Data exploration
train.head(5)
train.describe()
sns.heatmap(train.isnull(),cbar = False)
train.isnull().count()
#Age and Cabin has missing values,all other columns does not have any missing data.
fig1,ax1 = plt.subplots()
fare_surv = ax1.hist(train.Survived)
#Plot shows that there are more 0's compared to 1's
name_surv = train.pivot("Name","Survived","Cabin")
name_surv.head(50)
##Looks like there are more missing values in Cabin variable.
name_surv.isnull().count()
#Looks like something fishy,reaching out to kaggle forum!