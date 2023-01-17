import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

import math

import seaborn as sns

import datetime

print(os.listdir("../input"));
df = pd.read_csv('../input/environmental-remediation-sites.csv')
def mosaic_plot(df):

    for i in df.columns:

        df['_' + i] = df[i].map({pd.unique(df[i])[j]: j for j in range(len(pd.unique(df[i])))})

        df = df.drop(i, axis = 1)

    plt.rcParams['figure.figsize']=(20,20)

    g=sns.heatmap(df.corr(method='pearson'))

mosaic_plot(df)