#First lets import all the required libraries.

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



#Using “Heart Disease UCI” dataset as all its columns are numeric.

data = pd.read_csv("../input/heart.csv")



#Evaluating data texture (data type, null count & unique values) of dataset.

index = list(data.columns)

new_df = pd.DataFrame(index=index)

for ix in data.columns:

    new_df.at[ix,'data_types'] = data[ix].dtypes

    new_df.at[ix,'null_counts'] = data[ix].isnull().sum()

    new_df.at[ix,'unique_values'] = data[ix].nunique()

new_df
#Below function will cluster each values in a numeric column based on minimum and maximum value in a given group of range.

def rng_to_grp(min_n, max_n, grp, val):

    min_n = min_n if min_n % 2 == 0 else min_n - 1

    max_n = max_n if max_n % 2 == 0 else max_n + 1

    arr = np.linspace(round(min_n,0), round(max_n,0), grp)

    for i in arr:

        if val <= i:

            to_n = i

            break

    diff = arr[1] - arr[0]

    from_n = to_n - diff

    return str(round(from_n,0))+' to '+str(round(to_n,0))



#Using the "rng_to_grp" function for clustering those 5 columns which has good spread of unique values.

for ix in data.index:

    data.at[ix,'age_grp'] = rng_to_grp(data['age'].min(),data['age'].max(),10,data.at[ix,'age'])

    data.at[ix,'trestbps_grp'] = rng_to_grp(data['trestbps'].min(),data['trestbps'].max(),10,data.at[ix,'trestbps'])

    data.at[ix,'chol_grp'] = rng_to_grp(data['chol'].min(),data['chol'].max(),10,data.at[ix,'chol'])

    data.at[ix,'thalach_grp'] = rng_to_grp(data['thalach'].min(),data['thalach'].max(),10,data.at[ix,'thalach'])

    data.at[ix,'oldpeak_grp'] = rng_to_grp(data['oldpeak'].min(),data['oldpeak'].max(),10,data.at[ix,'oldpeak'])



#Post clustering again evaluating data texture of dataset.

index = list(data.columns)

new_df = pd.DataFrame(index=index)

for ix in data.columns:

    new_df.at[ix,'data_types'] = data[ix].dtypes

    new_df.at[ix,'null_counts'] = data[ix].isnull().sum()

    new_df.at[ix,'unique_values'] = data[ix].nunique()

new_df
#Adding a column “Count” with value 1 to dataset, in order to use as value in pivot_table.

data['Count'] = 1



#Creating a function to plot a graph for clustered columns.

def plot_graph(x):

    table = pd.pivot_table(data, values='Count', index=[x], columns=['target'], aggfunc=np.sum).fillna(0).reset_index()

    table['find'] = table[x].apply(lambda x: x[:x.find('to',0)]).astype(float)

    table = table.sort_values(['find'],ascending=True)

    fig, ax = plt.subplots()

    ax.stackplot(table[x], table[0], table[1], labels=["0", "1"], colors=['g','r'], zorder=2)

    ax.legend()

    plt.grid(zorder=1)

    plt.xticks(rotation=90)

    plt.title(x)

    plt.show()



#Running “plot_graph” function for data visualization of clustered columns.

plot_graph('age_grp')

plot_graph('trestbps_grp')

plot_graph('chol_grp')

plot_graph('thalach_grp')

plot_graph('oldpeak_grp')