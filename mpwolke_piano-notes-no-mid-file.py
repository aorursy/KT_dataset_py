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
 #pip install mido
#import mido

#output = mido.open_output()

#output.send(mido.Message('note_on', note=60, velocity=64))
#import mido

#msg = mido.Message('note_on', note=60)

#msg.type

#   'note_on'

#msg.note

#60

#msg.bytes()

#[144, 60, 64]

#msg.copy(channel=2)

#message note_on channel=2 note=60 velocity=64 time=0>
# with input as mido.open_input('SH-201'):

#...     for message in input:

#...         print(message)