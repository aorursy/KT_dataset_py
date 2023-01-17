# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
ss = pd.read_csv('../input/sample_submission.csv')

ss["surface"].value_counts()
ss.to_csv('concrete.csv', index=False)

ss.head(10)
ss['surface'] = "hard_tiles"

ss.to_csv('hard_tiles.csv', index=False)

ss.head(10)
ss['surface'] = "carpet"

ss.to_csv('carpet.csv', index=False)

ss.head(10)
ss['surface'] = "soft_tiles"

ss.to_csv('soft_tiles.csv', index=False)

ss.head(10)
ss['surface'] = "hard_tiles_large_space"

ss.to_csv('hard_tiles_large_space.csv', index=False)

ss.head(10)
ss['surface'] = "fine_concrete"

ss.to_csv('fine_concrete.csv', index=False)

ss.head(10)
ss['surface'] = "tiled"

ss.to_csv('tiled.csv', index=False)

ss.head(10)
ss['surface'] = "wood"

ss.to_csv('wood.csv', index=False)

ss.head(10)
ss['surface'] = "soft_pvc"

ss.to_csv('soft_pvc.csv', index=False)

ss.head(10)