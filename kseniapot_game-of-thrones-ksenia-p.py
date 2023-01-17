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
import pandas as pd
import numpy as np
import seaborn as sn
sn.set(color_codes=True, style="white")
import matplotlib.pyplot as ml
import warnings
warnings.filterwarnings("ignore")
death=pd.read_csv("../input/character-deaths.csv",sep=",",header=0)

print(death.shape)
print(death.head(11))
death_cor=death.corr()
print(death_cor)
sn.heatmap(death_cor, 
        xticklabels=death_cor.columns,
        yticklabels=death_cor.columns)
