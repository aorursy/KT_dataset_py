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
from __future__ import print_function

from keras.callbacks import LambdaCallback

from keras.models import Model, load_model, Sequential

from keras.layers import Dense, Activation, Dropout, Input, Masking

from keras.layers import LSTM

from keras.utils.data_utils import get_file

from keras.preprocessing.sequence import pad_sequences

from shakes.shakespeare_utils import *

import sys

import io
import os

os.chdir("../input")

import shutil

shutil.copy("../input/shakes/shakespeare.txt", "../input/shakespeare.txt")

shutil.copy("../input/shakes/shakespeare_utils.py", "../input/shakespeare_utils.py")
print_callback = LambdaCallback(on_epoch_end=on_epoch_end)



model.fit(x, y, batch_size=128, epochs=10, callbacks=[print_callback])
generate_output()