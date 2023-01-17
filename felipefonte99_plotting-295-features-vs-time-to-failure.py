import pandas as pd

import matplotlib.pyplot as plt
features_df = pd.read_csv('../input/features/features.csv', index_col=0)
features_df.head()
def plot_feature_ttf(feature):

    fig, ax1 = plt.subplots(figsize=(12, 5))

    plt.title('{} vs time to failure'.format(feature))

    plt.plot(features_df[feature], color='r')

    ax1.set_xlabel('training samples')

    ax1.set_ylabel('{}'.format(feature), color='r')

    plt.legend(['{}'.format(feature)], loc=(0.01, 0.95))



    ax2 = ax1.twinx()

    plt.plot(features_df['ttf'], color='b')

    ax2.set_ylabel('time to failure', color='b')

    plt.legend(['time to failure'], loc=(0.01, 0.9))

    

    plt.grid(True)

    plt.show()
features = [col for col in features_df.columns if col not in ['ttf']]

for feature in features:

    plot_feature_ttf(feature)