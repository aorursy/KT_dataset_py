# use numpy and pandas

import numpy as np

import pandas as pd



# We need sklearn for preprocessing and for the TSNE Algorithm.

import sklearn

from sklearn.preprocessing import Imputer, scale

from sklearn.manifold import TSNE



# WE employ a random state.

RS = 20150101



# We'll use matplotlib for graphics.

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

%matplotlib inline



# We import seaborn to make nice plots.

import seaborn as sns

# palette so that each person is having a color

palette = np.array(sns.color_palette("hls", 6))
data = pd.read_csv("../input/dataset.csv")

data.head()
data.info()
sns.countplot(x='ClassLabel', data=data, palette=palette[1:6])
data.describe()
X = data.copy()



# now we sort for the target

X.sort_values(by='ClassLabel', inplace=True)



# We split the target off the features and store it separately

y = X['ClassLabel']

X.drop('ClassLabel', inplace=True, axis=1)



# make sure the target is not part of the input data any more

assert 'ClassLabel' not in X.columns



# make sure the target is as expected and turn it into an array

assert set(y.unique()) == {1, 2, 3, 4, 5}

y = np.array(y)



# we scale the data

X = scale(X) 
# run the Algorithm

handtremor_proj = TSNE(random_state=RS).fit_transform(X)

handtremor_proj.shape
# choose the palette

palette = np.array(sns.color_palette("hls", 6))



# plot the result

def scatter_plot(x, colors, ax):

    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,

                    c=palette[colors.astype(np.int)])

    ax.axis('off')

    ax.axis('tight')    

    return sc



# plot the legend

def legend_plot(font_size=14):

    patch1 = mpatches.Patch(color=palette[1], label='Person 1')

    patch2 = mpatches.Patch(color=palette[2], label='Person 2')

    patch3 = mpatches.Patch(color=palette[3], label='Person 3')

    patch4 = mpatches.Patch(color=palette[4], label='Person 4')

    patch5 = mpatches.Patch(color=palette[5], label='Person 5')

    plt.legend(handles=[patch1, patch2, patch3, patch4, patch5], fontsize=font_size, loc=4)
f = plt.figure(figsize=(8, 8))

f.suptitle('Geometry of Handtremor for 5 persons', fontsize=20)

ax = plt.subplot(aspect='equal')

scatter_plot(handtremor_proj, y, ax)

legend_plot()
# finding the indexes for each person

persons = {}

for i in range(1, 6):

    persons[i] = np.where(y == i)    
# now we make a separate subfigure for each person

f, axs = plt.subplots(2, 3, figsize=(12,8))

axs[0,0] = scatter_plot(handtremor_proj[persons[1]], y[persons[1]], axs[0,0])

axs[1,0] = scatter_plot(handtremor_proj[persons[2]], y[persons[2]], axs[1,0])

axs[0,1] = scatter_plot(handtremor_proj[persons[3]], y[persons[3]], axs[0,1])

axs[1,1] = scatter_plot(handtremor_proj[persons[4]], y[persons[4]], axs[1,1])

axs[0,2] = scatter_plot(handtremor_proj[persons[5]], y[persons[5]], axs[0,2])

axs[-1, -1].axis('off')

legend_plot(font_size=20)