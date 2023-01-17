# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

#using a file from our dataset pane
athletes = pd.read_csv("../input/athlete_events.csv", sep=",", header = 0)

print(athletes.head(4))
sn.boxplot(data=athletes)


sn.violinplot(x="Sex", y="Age", data=athletes)

co = athletes.corr()
sn.heatmap(co, annot=True, linewidths=1.0)


#RESULTS
#As the weight increases of the olympic athletes there is a high correlation in an increase in height of the olympic athlete
#Olympic athletes are split between men and women but the average age of the athlete is mid-twenties