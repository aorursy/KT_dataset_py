# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas.plotting import scatter_matrix


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print(os.listdir("../input"))
import matplotlib.pyplot as plt
import pylab as P


# Any results you write to the current directory are saved as output.
%pprint
admission_file = ("../input/Admission_Predict_Ver1.1.csv")
admission_data=pd.read_csv(admission_file)
admission_data.shape
admission_data.info()
admission_data.head()
admission_data.tail()
admission_data.columns = ['serial', 'gre', 'toefl','rating', 'sop', 'lor', 'cgpa', 'research', 'chances']
admission_data = admission_data.drop(['serial'], axis=1)
admission_data.describe()
avg_GRE= admission_data['gre'].mean()
avg_SOP = admission_data['sop'].mean()
avg_Rating = admission_data['rating'].mean()
avg_cgpa = admission_data['cgpa'].mean()
admission_data.corr()
greScores = admission_data['gre'].unique()
sorted(greScores)
adScores = admission_data['toefl'].unique()
sorted(adScores)
plot = admission_data['gre'].hist()
plot.set_xlabel("GRE Score")
plot.set_ylabel("Frequency")

plot=admission_data['toefl'].hist()
plot.set_xlabel("TOEFL Score")
plot.set_ylabel("Frequency")

plot = admission_data['cgpa'].hist()
plot.set_xlabel("CGPA Score")
plot.set_ylabel("Frequency")

plot = admission_data['sop'].hist()

plot.set_xlabel("Statement of letter strength")
plot.set_ylabel("Frequency")

admission_data['research'].hist()
P.show()
# draw a histogram (shortcut to features of matplotlib/pylab packages)
import pylab as P
admission_data['rating'].hist()
P.show()
#boxplot
admission_data.boxplot(column='gre')
#boxplot
admission_data.boxplot(column='toefl')
#boxplot
admission_data.boxplot(column='cgpa')
#boxplot
admission_data.boxplot(column='chances')
#values skew toward acceptance (average is over 70%)
# draw a histogram (shortcut to features of matplotlib/pylab packages)
import pylab as P
admission_data['chances'].hist()
P.show()
#note it skews much more towards being admitted
#boxplot one way to create it
bxplt= admission_data.boxplot(column='chances', by = 'cgpa')
xticks = [10,30,50, 70, 90, 110, 130, 150, 170, 190]
bxplt.xaxis.set_ticks(xticks)
bxplt.set_xticklabels(xticks, fontsize=16)

#boxplot using matlib
admission_data.boxplot(by=['chances'], column=['cgpa'])
# set your own proper title
plt.title('Boxplot of CGPA grouped by Admit Chance')
# get rid of the automatic 'Boxplot grouped by group_by_column_name' title
plt.suptitle("")
# Customize x tick lables
x = [1]
# create an index for each tick position

plt.xticks(range(0, 75, 10), fontsize=14)

#get first 5 rows
admission_data.iloc[0:5,:]
# scatter plot matrix
scatter_matrix(admission_data)
plt.show()