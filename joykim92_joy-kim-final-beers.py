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
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
beers = pd.read_csv("../input/beers.csv")
beers
beers.shape
beers.head(10)
beers.corr()
sns_heatmap = sns.heatmap(beers.corr(), xticklabels=beers.corr().columns, yticklabels=beers.corr().columns, annot=True)
sns_heatmap.set_title('Beers Correlation Heatmap')
