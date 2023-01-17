import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.style as style 

style.use('ggplot')



train_df = pd.read_csv('../input/X_train.csv')

target = pd.read_csv("../input/y_train.csv")

test_df = pd.read_csv('../input/X_test.csv')

sample_submission = pd.read_csv("../input/sample_submission.csv")



series_dict = {}

for series in (train_df['series_id'].unique()):

    series_dict[series] = train_df[train_df['series_id'] == series]

    

def plotSeries(series_id):

    style.use('ggplot')

    plt.figure(figsize=(28, 16))

    print(target[target['series_id'] == series_id]['surface'].values[0].title())

    for i, col in enumerate(series_dict[series_id].columns[3:]):

        if col.startswith("o"):

            color = 'red'

        elif col.startswith("a"):

            color = 'green'

        else:

            color = 'blue'

        if i >= 7:

            i+=1

        plt.subplot(3, 4, i + 1)

        plt.plot(series_dict[series_id][col], color=color, linewidth=3)

        plt.title(col)
id_series = 13

plotSeries(id_series)
id_series = 134

plotSeries(id_series)