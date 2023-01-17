data = []

with open('../input/SBW-vectors-300-min5.txt', 'r') as fopen:

    for n, line in enumerate(fopen):

        data.append(line)

        if (n + 1) % 100000 == 0:

            print('done processed: ', n + 1)

            

print('done!')
# i will remove first element in data list

del data[0]



# i shuffle because i just want to take first 100k words in the data

import random

random.shuffle(data)
data = filter(None, data)

data_vector, words = [], []

for n, i in enumerate(data):

    vec = i.split(' ')

    words.append(vec[0])

    vec = vec[1:]

    vec = filter(None, vec)

    vec = [float(number) for number in vec]

    data_vector.append(vec)

    # break if 100k

    if (n + 1) == 100000:

        break
import numpy as np

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import Normalizer



# i will change it into numpy matrix

X = np.array(data_vector)



# normalize into positive value, std deviation

X = StandardScaler().fit_transform(X)



# normalize it between 0 and 1

X = Normalizer().fit_transform(X)
# to split 30% to visualize later using PCA

from sklearn.cross_validation import train_test_split

from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



kmeans = KMeans(n_clusters = 6).fit(X)

y = kmeans.labels_

unique_label = np.unique(y)
# create a dictionary for our words clustering

dict_word = {0: [], 1:[], 2:[], 3:[], 4:[], 5:[]}

for i in range(len(words)):

    dict_word[y[i]].append(words[i])
for key, value in dict_word.items():

    print('class: ', key)

    # print at least 10

    print(value[:10], '\n')
_, X_, _, y_, = train_test_split(X, y, test_size = 0.3)

data_visual = PCA(n_components = 2).fit_transform(X_)

plt.rcParams["figure.figsize"] = [21, 21]

ax = plt.subplot(111)

current_palette = sns.color_palette()

for no, _ in enumerate(np.unique(y_)):

    ax.scatter(data_visual[y_ == no, 0], data_visual[y_ == no, 1], c = current_palette[no], label = unique_label[no], alpha = 0.5)

    

box = ax.get_position()

ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 1])

ax.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.05), fancybox = True, shadow = True, ncol = 4)

plt.show()
words = np.array(words)

words_
import plotly.offline as py

py.init_notebook_mode(connected = True)

import plotly.graph_objs as go



_, X_, _, y_, _, words_ = train_test_split(X, y, words, test_size = 0.05)



data_visual = PCA(n_components = 3).fit_transform(X_)

current_palette = sns.color_palette()

data_graph = []

for no, _ in enumerate(np.unique(y_)):

    temp_text = []

    for i in range(words_.shape[0]):

        if y_[i] == no:

            temp_text.append(words_[i])

    graph = go.Scatter3d(

    x = data_visual[y_ == no, 0],

    y = data_visual[y_ == no, 1],

    z = data_visual[y_ == no, 2],

    text = temp_text,

    name = unique_label[no],

    mode = 'markers',

    marker = dict(

        size = 12,

        line = dict(

            color = current_palette[no],

            width = 0.5

            ),

        opacity = 0.5

        )

    )

    data_graph.append(graph)

    

layout = go.Layout(

    scene = dict(

        camera = dict(

            eye = dict(

            x = 0.5,

            y = 0.5,

            z = 0.5

            )

        )

    ),

    margin = dict(

        l = 0,

        r = 0,

        b = 0,

        t = 0

    )

)

fig = go.Figure(data = data_graph, layout = layout)

py.iplot(fig, filename = '3d-scatter')