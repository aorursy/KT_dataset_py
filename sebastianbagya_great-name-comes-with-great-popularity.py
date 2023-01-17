import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer





def draw_plot(data, start_range, end_range, total_data, bar_color, chart_title):

    plt.rcdefaults()

    fig, ax = plt.subplots()

    data_filtered = data['title'][(data['score'] > start_range) & (data['score'] < end_range)].drop_duplicates()

    cv = CountVectorizer(stop_words='english')

    cv_fit = cv.fit_transform(data_filtered)

    data_frame = {'Name': cv.get_feature_names(), 'Freq': cv_fit.toarray().sum(axis=0)}

    data_graph = pd.DataFrame(data_frame).sort_values(by=['Freq'], ascending=False)[0:total_data]

    objects = data_graph['Name'].values.tolist()

    y_pos = np.arange(len(data_graph['Name']))

    frequency = data_graph['Freq'].values.tolist()

    ax.barh(y_pos, frequency, align='center',

            color=bar_color, ecolor='black', alpha=0.5)

    ax.set_yticks(y_pos)

    ax.set_yticklabels(objects)

    ax.invert_yaxis()

    ax.set_xlabel('Frequency')

    ax.set_title(chart_title)

    plt.show()
file = pd.read_csv('../input/ign.csv')

file.head()
draw_plot(file, 0, 10.1, 20, 'black', 'What is the popular words?')
draw_plot(file, 9.4, 10.1, 20, 'blue', 'What made the masterpieces?')
draw_plot(file, 5.9, 9.5, 20, 'green', 'What is the Okay and above?')
draw_plot(file, 0, 6.0, 20, 'red', 'The Worst')