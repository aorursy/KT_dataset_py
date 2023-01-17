# Importing libs

import pandas as pd

import numpy as np

from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import warnings

warnings.filterwarnings("ignore")
def format_spines(ax, right_border=True):

    """docstring for format_spines:

    this function sets up borders from an axis and personalize colors

    input:

        ax: figure axis

        right_border: flag to determine if the right border will be visible or not"""

    

    # Setting up colors

    ax.spines['bottom'].set_color('#CCCCCC')

    ax.spines['left'].set_color('#CCCCCC')

    ax.spines['top'].set_color('#FFFFFF')

    if right_border:

        ax.spines['right'].set_color('#CCCCCC')

    else:

        ax.spines['right'].set_color('#FFFFFF')

    ax.patch.set_facecolor('#FFFFFF')





def count_plot(feature, df, colors='Blues_d', hue=False):

    """docstring for count_plot:

    this function plots data setting up frequency and percentage. This algo sets up borders

    and personalization

    input:

        feature: feature to be plotted

        df: dataframe

        colors = color palette (default=Blues_d)

        hue = second feature analysis (default=False)"""

    

    # Preparing variables

    ncount = len(df)

    fig, ax = plt.subplots()

    if hue != False:

        ax = sns.countplot(x=feature, data=df, palette=colors, hue=hue)

    else:

        ax = sns.countplot(x=feature, data=df, palette=colors)



    # Make twin axis

    ax2=ax.twinx()



    # Switch so count axis is on right, frequency on left

    ax2.yaxis.tick_left()

    ax.yaxis.tick_right()



    # Also switch the labels over

    ax.yaxis.set_label_position('right')

    ax2.yaxis.set_label_position('left')



    ax2.set_ylabel('Frequency [%]')



    # Setting borders

    format_spines(ax)

    format_spines(ax2)



    # Setting percentage

    for p in ax.patches:

        x=p.get_bbox().get_points()[:,0]

        y=p.get_bbox().get_points()[1,1]

        ax.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), 

                ha='center', va='bottom') # set the alignment of the text

    if not hue:

        ax.set_title(df[feature].describe().name + ' Analysis', size=13, pad=15)

    else:

        ax.set_title(df[feature].describe().name + ' Analysis by ' + hue, size=13, pad=15)

        

    plt.show()



def compute_square_distances(df, Kmin=1, Kmax=12):

    """docstring for compute_square_distances

    this function computes the square distance of KMeans algorithm through the number of

    clusters in range Kmin and Kmax

    input:

        df: dataframe

        Kmin: min index of K analysis

        Kmax: max index of K analysis"""

    

    square_dist = []

    K = range(Kmin, Kmax)

    for k in K:

        km = KMeans(n_clusters=k)

        km.fit(df)

        square_dist.append(km.inertia_)

    return K, square_dist



def plot_elbow_method(df, Kmin=1, Kmax=12):

    """docstring for plot_elbow_method

    this function computes the square distances and plots the elbow method for best cluster

    number analysis

    input:

        df: dataframe

        Kmin: min index of K analysis

        Kmax: max index of K analysis"""

    

    # Computing distances

    K, square_dist = compute_square_distances(df, Kmin, Kmax)

    

    # Plotting elbow method

    fig, ax = plt.subplots()

    ax.plot(K, square_dist, 'bo-')

    format_spines(ax, right_border=False)

    plt.xlabel('Number of Clusters')

    plt.ylabel('Sum of Square Dist')

    plt.title(f'Elbow Method - {df.columns[0]} and {df.columns[1]}', size=14)

    plt.show()

    

def plot_kmeans(df, y_kmeans, centers):

    """docstring for plotKMeans

    this function plots the result of a KMeans training

    input:

        df: dataframe

        y_kmeans: kmeans prediction

        centers: cluster centroids"""

    

    # Setting up and plotting

    X = df.values

    sns.set(style='white', palette='muted', color_codes=True)

    fix, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='plasma')

    ax.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)

    ax.set_title('KMeans Applied', size=14)

    ax.set_xlabel(f'{df.columns[0]}', size=12, labelpad=5)

    ax.set_ylabel(f'{df.columns[1]}', size=12, labelpad=5)

    format_spines(ax, right_border=False)

    plt.show()
# Reading data

df = pd.read_csv(r'../input/Mall_Customers.csv')

df.head()
# Dims

df.shape
# Communication

print(f'This dataset has {df.shape[0]} rows and {df.shape[1]} columns.')
# Null data

df.isnull().sum()
# Dataset info

df.info()
# Some statistics

df.describe()
# Numerical features distribution

sns.set(style='white', palette='muted', color_codes=True)

fig, axs = plt.subplots(1, 3, figsize=(12, 3))

sns.despine(left=True)

axs[0] = sns.distplot(df['Age'], bins=20, ax=axs[0])

axs[1] = sns.distplot(df['Annual Income (k$)'], bins=20, ax=axs[1], color='g')

axs[2] = sns.distplot(df['Spending Score (1-100)'], bins=20, ax=axs[2], color='r')



fig.suptitle('Numerical Feature Distribution')

plt.setp(axs, yticks=[])

plt.tight_layout()

plt.show()
# Counting gender

custom_colors = ["#3498db", "#C8391A"]

count_plot(feature='Gender', df=df, colors=custom_colors)
# Looking at age values

df['Age'].describe()
# Creating new category

bins = [18, 22, 50, 70]

labels = ['Young', 'Adult', 'Senior']

df['Age Range'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)



df.head()
# Result

count_plot(feature='Age Range', df=df, colors='YlGnBu')
# Gender by Age Range

count_plot(feature='Gender',df=df, hue='Age Range')
# Maybe the inverse would be more clear

count_plot(feature='Age Range', df=df, colors=custom_colors, hue='Gender')
# Spending Score Distribution

fig, ax = plt.subplots(figsize=(10, 4), sharex=True)

female = df.loc[df['Gender'] == 'Female']

male = df.loc[df['Gender'] == 'Male']

ax = sns.distplot(female['Spending Score (1-100)'], bins=20, label='female', 

                  color='r')

ax = sns.distplot(male['Spending Score (1-100)'], bins=20, label='male')

ax.set_title('Spending Score Distribution by Gender', size=14)

format_spines(ax, right_border=False)

plt.legend()

plt.show()
# Annual Income Distribution

fig, ax = plt.subplots(figsize=(10, 4), sharex=True)

female = df.loc[df['Gender'] == 'Female']

male = df.loc[df['Gender'] == 'Male']

ax = sns.distplot(female['Annual Income (k$)'], bins=20, label='female', 

                  color='r', hist=True)

ax = sns.distplot(male['Annual Income (k$)'], bins=20, label='male')

ax.set_title('Annual Income Distribution by Gender', size=14)

format_spines(ax, right_border=False)

plt.legend()

plt.show()
# Configuration

sns.set(style='white', palette='muted', color_codes=True)

fig, axs = plt.subplots(1, 3, figsize=(12, 3))

sns.despine(left=True)



# Dataframe indexing

young = df.loc[df['Age Range'] == 'Young']

adult = df.loc[df['Age Range'] == 'Adult']

senior = df.loc[df['Age Range'] == 'Senior']

titles = ['Young', 'Adult', 'Senior']

age_range_dataframes = [young, adult, senior]



for idx in range(3):

    age_range = age_range_dataframes[idx]

    axs[idx] = sns.distplot(age_range[age_range['Gender']=='Male']['Spending Score (1-100)'], 

                          bins=20, ax=axs[idx], label='male', color='b', hist=False)

    axs[idx] = sns.distplot(age_range[age_range['Gender']=='Female']['Spending Score (1-100)'], 

                          bins=20, ax=axs[idx], label='female', color='r', hist=False)

    axs[idx].set_title(titles[idx], size=13)



fig.suptitle('Spending Score Distribution by Gender and Age Range')

plt.setp(axs, yticks=[])

plt.tight_layout()

plt.subplots_adjust(top=0.75)

plt.show()
# Spending Score Distribution by Age Range

fig, ax = plt.subplots(figsize=(10, 4), sharex=True)

young = df.loc[df['Age Range'] == 'Young']

adult = df.loc[df['Age Range'] == 'Adult']

senior = df.loc[df['Age Range'] == 'Senior']

ax = sns.distplot(young['Spending Score (1-100)'], bins=10, label='Young', color='b')

ax = sns.distplot(adult['Spending Score (1-100)'], bins=10, label='Adult', color='g')

ax = sns.distplot(senior['Spending Score (1-100)'], bins=10, label='Senior', color='grey')

ax.set_title('Spending Score Distribution by Gender', size=14)

format_spines(ax, right_border=False)

plt.legend()

plt.show()
# Indexing dataframe

df_1 = df.loc[:, ['Age', 'Spending Score (1-100)']]



# Searching for optimun K

plot_elbow_method(df_1)
# Training KMeans

k_means = KMeans(n_clusters=4)

k_means.fit(df_1)

y_kmeans = k_means.predict(df_1)

centers = k_means.cluster_centers_

plot_kmeans(df_1, y_kmeans, centers)
# Dataframe indexing

df_2 = df.loc[:, ['Annual Income (k$)', 'Spending Score (1-100)']]



# Optimum K

plot_elbow_method(df_2)
# Training

k_means = KMeans(n_clusters=5)

k_means.fit(df_2)

y_kmeans = k_means.predict(df_2)

centers = k_means.cluster_centers_

plot_kmeans(df_2, y_kmeans, centers)