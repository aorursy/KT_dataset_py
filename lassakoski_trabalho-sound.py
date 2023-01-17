# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

#kick_signals = [librosa.load(p)[0] for p in Path().glob('../input/aula3-dataset/dataset/train/kick_*.wav')]

#snare_signals = [librosa.load(p)[0] for p in Path().glob('../input/aula3-dataset/dataset/train/snare_*.wav')]
classe = pd.read_csv("../input/urbansound8k/UrbanSound8K.csv")

classe_0 =classe[classe['classID']== 0]
classe_1 =classe[classe['classID']== 1]
classe_2 =classe[classe['classID']== 2]
classe_3 =classe[classe['classID']== 3]
classe_4 =classe[classe['classID']== 4]
classe_5 =classe[classe['classID']== 5]
classe_6 =classe[classe['classID']== 6]
classe_7 =classe[classe['classID']== 7]
classe_8 =classe[classe['classID']== 8]
classe_9 =classe[classe['classID']== 9]

os.makedirs('/kaggle/working/train/')

from distutils.dir_util import copy_tree

fromDirectory1 = '../input/urbansound8k/fold1'
fromDirectory2 = '../input/urbansound8k/fold2'
fromDirectory3 = '../input/urbansound8k/fold3'
fromDirectory4 = '../input/urbansound8k/fold4'
fromDirectory5 = '../input/urbansound8k/fold5'
fromDirectory6 = '../input/urbansound8k/fold6'
fromDirectory7 = '../input/urbansound8k/fold7'

toDirectory = '/kaggle/working/train/'

copy_tree(fromDirectory1,toDirectory)
copy_tree(fromDirectory2,toDirectory)
copy_tree(fromDirectory3,toDirectory)
copy_tree(fromDirectory4,toDirectory)
copy_tree(fromDirectory5,toDirectory)
copy_tree(fromDirectory6,toDirectory)
copy_tree(fromDirectory7,toDirectory)
classe_0.slice_file_name.iloc[999]

signals_0 = [librosa.load(p)[0] for p in Path().glob('../input/aula3-dataset/dataset/train/kick_*.wav')]