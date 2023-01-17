import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
%%time

df = pd.read_csv("../input/train.csv")
%%time

features_list = list(df.columns[2:])
%%time

features_list_log = []



for feature in features_list:

    features_list_log.append(np.log(df[feature]))
def visualize_features_log(list_of_features, list_of_logs):

    figsize_row = int(len(list_of_features) + (len(list_of_features)*0.2))

    figsize_col = int(len(list_of_features)/2)

    fig, axes = plt.subplots(nrows=1, ncols=10, figsize=(figsize_row, figsize_col))



    for i in range(len(list_of_features)):

        plt.subplot(5, 10, i+1)

        plt.scatter(df[list_of_features[i]], list_of_logs[i])

        plt.title(list_of_features[i], fontdict={'fontsize': 36})

    plt.show()
%%time

visualize_features_log(features_list[:50], features_list_log[:50])
def density_plot():

    

    sns.set_style('whitegrid')

    plt.figure()

    fig, ax = plt.subplots(5, 10, figsize=(25, 10))

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,

                    wspace=0.35)

    

    for i, feature in enumerate(df.columns[2:][:50]):

       

        target_0 = df[feature].loc[df['target'] == 0]

        target_1 = df[feature].loc[df['target'] == 1]

        

        plt.subplot(5, 10, i+1)

        sns.kdeplot(target_0, bw=0.5, label='0')



        plt.subplot(5, 10, i+1)

        plt.title(feature)

        plt.yticks([])

        plt.xticks([])

        sns.kdeplot(target_1, bw=0.5, label='1')

    plt.show()
%%time

density_plot()