# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import warnings

warnings.filterwarnings('ignore')
!pip install pydotplus
from sklearn import tree

from sklearn.tree import DecisionTreeClassifier, export_graphviz

from sklearn.model_selection import train_test_split



from matplotlib import pyplot as plt

import seaborn as sns



import graphviz

import pydotplus

import io

from scipy import misc

import imageio
songs = pd.read_csv('../input/spotifyclassification/data.csv')

type(songs)
songs.describe()
songs.head()
train, test = train_test_split(songs, test_size = 0.15)

print("Training size:", len(train), "Test Size:", len(test))

train.shape
pos_tempo = songs[songs['target'] == 1]['tempo']

neg_tempo = songs[songs['target'] == 0]['tempo']

fig = plt.figure(figsize=(12,8))

plt.title("Song Tempo Like/deslike distribution")

pos_tempo.hist(alpha = 0.7, bins = 25, label='positive')

neg_tempo.hist(alpha = 0.7, bins = 25, label='negative')

plt.legend(loc = 'upper rigth')
# Loudness

pos_loud = songs[songs['target'] == 1]['loudness']

neg_loud = songs[songs['target'] == 0]['loudness']



fig2 = plt.figure(figsize=(15,15))





ax3 = fig2.add_subplot(331)

ax3.set_xlabel('loudness')

ax3.set_ylabel('Count')

ax3.set_title('Song loudness Like/deslike distribution')



pos_loud.hist(alpha = 0.5, bins=30)

ax4 = fig2.add_subplot(331)

neg_loud.hist(alpha = 0.5, bins=30)



# energy

pos_energy = songs[songs['target'] == 1]['energy']

neg_energy = songs[songs['target'] == 0]['energy']



ax5 = fig2.add_subplot(332)

ax5.set_xlabel('energy')

ax5.set_ylabel('Count')

ax5.set_title('Song energy Like/deslike distribution')



pos_energy.hist(alpha = 0.5, bins=30)

ax6 = fig2.add_subplot(332)

neg_energy.hist(alpha = 0.5, bins=30)





# instrumentalness



pos_inst = songs[songs['target'] == 1]['instrumentalness']

neg_inst = songs[songs['target'] == 0]['instrumentalness']



ax7 = fig2.add_subplot(333)

ax7.set_xlabel('instrumentalness')

ax7.set_ylabel('Count')

ax7.set_title('Song instrumentalness Like/deslike distribution')



pos_inst.hist(alpha = 0.5, bins=30)

ax8 = fig2.add_subplot(333)

neg_inst.hist(alpha = 0.5, bins=30)



#artist



pos_duration = songs[songs['target'] == 1]['duration_ms']

neg_duration = songs[songs['target'] == 0]['duration_ms']



ax9 = fig2.add_subplot(334)

ax9.set_xlabel('duration_ms')

ax9.set_ylabel('Count')

ax9.set_title('Song duration_ms Like/deslike distribution')



pos_duration.hist(alpha = 0.5, bins=30)

ax10 = fig2.add_subplot(334)

neg_duration.hist(alpha = 0.5, bins=30)
t = DecisionTreeClassifier(min_samples_split=110)

features = ['danceability',"loudness","valence","energy", "instrumentalness", "acousticness","key","speechiness","duration_ms"]
x_train = train[features]

y_train = train['target']



x_test = test[features]

y_test = test['target']
dt = t.fit(x_train, y_train)
# show decision tree

def show_tree(tree, features, path):

    f = io.StringIO()

    export_graphviz(tree, out_file=f, feature_names= features)

    pydotplus.graph_from_dot_data(f.getvalue()).write_png(path)

    img = imageio.imread(path)

    plt.rcParams["figure.figsize"] = (20,20)

    plt.imshow(img)

#show_tree(dt,features,'dec_tree_01.png')
dt.score(x_test,y_test)
from sklearn.ensemble import RandomForestClassifier
#random forest

rf = RandomForestClassifier(n_estimators=20)

rf.fit(x_train, y_train)
rf.score(x_test,y_test)
from sklearn.ensemble import BaggingClassifier
bg = BaggingClassifier(DecisionTreeClassifier(), max_samples=0.5, max_features = 1.0, n_estimators = 20 )

bg.fit(x_train, y_train)
bg.score(x_test,y_test)