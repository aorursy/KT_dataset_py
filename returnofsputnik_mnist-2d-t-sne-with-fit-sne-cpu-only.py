# First we install the fast part in C. http://www.fftw.org/download.html

!wget http://www.fftw.org/fftw-3.3.8.tar.gz
# Extract the contents. Do not use verbose or else you get spammed with lots of files it unpacked.

!tar xzf fftw-3.3.8.tar.gz
# We follow the installation instructions from here:

# http://www.fftw.org/fftw3_doc/Installation-on-Unix.html#Installation-on-Unix

!./fftw-3.3.8/configure --silent
# Instead of running !make, we do this to silence all output.

# The output is really long and can make the web page crash because of all the text needed to display.

!make >/dev/null || make
!make install --silent
# Now we just need to download the t-SNE part.

!git clone https://github.com/KlugerLab/FIt-SNE.git
# Compile the program

!g++ -std=c++11 -O3 FIt-SNE/src/sptree.cpp FIt-SNE/src/tsne.cpp FIt-SNE/src/nbodyfft.cpp -I fftw-3.3.8/api/ -L .libs/ -o FIt-SNE/bin/fast_tsne -pthread -lfftw3 -lm
import sys; sys.path.append('./FIt-SNE/') # Append it so we can import fast_tsne from that directory
import pandas as pd

import numpy as np

from fast_tsne import fast_tsne

import matplotlib.pyplot as plt

%matplotlib inline
train = pd.read_csv('../input/digit-recognizer/train.csv')

test = pd.read_csv('../input/digit-recognizer/test.csv')
y = train['label'].values

train = train[test.columns].values

test = test[test.columns].values
train_test = np.vstack([train, test])

train_test.shape
%%time

# We have to have 'random' initialization instead of default 'pca', because RAPIDS also uses random initialization.

# This is so we can have a fair comparison.

train_test_2D = fast_tsne(train_test, map_dims=2, initialization='random', seed=1337)
%%time

# We have to have 'random' initialization instead of default 'pca', because RAPIDS also uses random initialization.

# This is so we can have a fair comparison.

train_2D = fast_tsne(train, map_dims=2, initialization='random', seed=1337)
plt.scatter(train_2D[:,0], train_2D[:,1], c = y, s = 0.5)
train_2D = train_test_2D[:train.shape[0]]

test_2D = train_test_2D[train.shape[0]:]



np.save('train_2D', train_2D)

np.save('test_2D', test_2D)