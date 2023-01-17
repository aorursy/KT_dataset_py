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
import numpy as np

import pandas as pd

from tqdm.notebook import tqdm
def manhattan_distance_single(i1, i2, size=21):

    """Gets the distance in one dimension between two columns or 

    two rows, including wraparound."""

    iMin = min(i1, i2)

    iMax = max(i1, i2)

    return min(iMax - iMin, iMin + size - iMax)





def manhattan_distance(pos1, pos2, size=21):

    """Gets the Manhattan distance between two positions, i.e.,

    how many moves it would take a ship to move between them."""

    # E.g. for 17-size board, 0 and 17 are actually 1 apart

    dx = manhattan_distance_single(pos1 % size, pos2 % size)

    dy = manhattan_distance_single(pos1 // size, pos2 // size)

    return dx + dy
%%time



for i in tqdm(range(1_000_000)):

    a = manhattan_distance(0, 220)

print(a)
%%time 



# do this only once per game



def dist_1d(a1, a2):

    amin = np.fmin(a1, a2)

    amax = np.fmax(a1, a2)

    adiff = amax-amin

    adist = np.fmin(adiff, 21-adiff)

    return adist





def make_dist_matrix():

    base = np.arange(21**2)

    idx1 = np.repeat(base, 21**2)

    idx2 = np.tile(base, 21**2)



    rowdist = dist_1d(idx1 // 21, idx2 // 21)

    coldist = dist_1d(idx1 % 21, idx2 % 21)



    dist_matrix = (rowdist + coldist).reshape(21**2, -1)

    return dist_matrix



distance_list = make_dist_matrix().tolist()

%%time

for i in tqdm(range(1_000_000)):

    a = distance_list[0][220]

print(a)