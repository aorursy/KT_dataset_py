# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sb

import matplotlib.pyplot as plt



train_data = pd.read_csv('../input/train.csv',)

train_data.head()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Remove ticket from the DataFrame(rows,colums)

print(train_data.head())

sb.set(style="ticks")

train_data.info()
g = sb.lmplot(x="Age", y="PassengerId",ci=None,data=train_data, col="Survived",

    palette="muted",col_wrap=2,scatter_kws={"s": 100,"alpha":.5},

    line_kws={"lw":4,"alpha":0.5},hue="Survived",x_jitter=1.0,y_jitter=1.0,size=6)



# remove the top and right line in graph

sb.despine()

# Additional line to adjust some appearance issue

plt.subplots_adjust(top=0.9)



# Set the Title of the graph from here

g.fig.suptitle('Age vs. PassengerId', fontsize=10,color="b",alpha=0.5)



# Set the xlabel of the graph from here

g.set_xlabels("Age",size = 10,color="b",alpha=0.5)



# Set the ylabel of the graph from here

g.set_ylabels("PassengerId",size = 10,color="b",alpha=0.5)
# Replace values in the DataFrame which are not identified

def random_age():

    for age in train_data["Age"]:

        if age == 'NaN':

            age = rand(sum(train_data["Age"])/train_data().size())

    return age



def random_passengerId():

    for passengerId in train_data["PassengerId"]:

        if passengerId == 'NaN':

            passengerId = rand(sum(train_data["PassengerId"])/train_data().size())

    return passengerId



train_data["Age"][np.isnan(train_data["Age"])] = random_age()

train_data["PassengerId"][np.isnan(train_data["PassengerId"])] = random_passengerId()



# Transform Type Object into valid Type for the DataFrame 

train_data['Fare'] = train_data['Fare'].astype(int)





# Create a Pairplot

g = sb.pairplot(train_data,hue="Survived",palette="muted",size=5, 

    vars=["Age", "PassengerId", "SibSp", "Parch", "Pclass", "Fare"],kind='reg')



# To change the size of the scatterpoints in graph

g = g.map_offdiag(plt.scatter,  s=150, alpha=0.5)