# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # Visualize the data

import seaborn as sns # Visualize the data



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        os.chdir(dirname)

        dataset = pd.read_csv(filename)



# Any results you write to the current directory are saved as output.
# Explarotory Data Analysis

print("The type of Data set would be : ", type(dataset))

print(dataset.shape)

dataset.describe()
# Create a New Dataset with pivoting

new_dataset = pd.pivot_table(dataset, index = ["year_award","nominee"], values = "win", aggfunc = lambda x: len(x.unique()))



# Identifying the nominee who has received the most awards single year

winner = new_dataset[new_dataset["win"]==2]

print("The number winner based on a single year are =", winner.count()[0])

# Identifying the awards based on a overall

from collections import Counter

new_dataset = Counter(dataset["nominee"])

highlights  = new_dataset.most_common()[1:10]



# Define the function of separating

def separate(data_input):

    # Visualize the tupple

    X = []

    y = []

    for i,j in enumerate(data_input) :

        X.append(j[0])

        y.append(j[1])

    return(X,y)



# Defint the function of visualization

def visualize_sns(X,y, title):

    # Visualize using Seaborn

    plt.figure(figsize=(10,5))

    chart = sns.barplot(x = X, y = y)

    chart.set_xticklabels(labels = X, rotation=89)

    chart.set_title(title)

    chart.set(ylabel = "The total number of awards")

# Running the function

X, y = separate(highlights)

visualize_sns(X,y,"The 10 most award nominees")



from tabulate import tabulate

print(tabulate(highlights, headers = ["Most Award","Values"]))
# Identifying the awards based on a overall

from collections import Counter

New_Dataset = Counter(dataset["category"])

highlights  = New_Dataset.most_common()[1:10]



# Define the function of separating

def separate(data_input):

    # Visualize the tupple

    X = []

    y = []

    for i,j in enumerate(data_input) :

        X.append(j[0])

        y.append(j[1])

    return(X,y)



# Defint the function of visualization

def visualize_sns(X,y, title):

    # Visualize using Seaborn

    plt.figure(figsize=(10,5))

    chart = sns.barplot(x = X, y = y)

    chart.set_xticklabels(labels = X, rotation=75)

    chart.set_title(title)

    chart.set(ylabel = "The total number of awards")

# Running the function

X, y = separate(highlights)

visualize_sns(X,y,"The 10 received job most awards in a event")



# Create a Table

from tabulate import tabulate

print(tabulate(highlights, headers = ["Most Work Award","Values"]))

# Identifying the awards based on a overall

from collections import Counter

new_dataset = Counter(dataset["nominee"])

highlights  = new_dataset.most_common()[1:10]



X, y = separate(highlights)

print("It is",X[0],"&",X[1],"with total number of awards", y[0], "&", y[1], "respectively")