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
import seaborn as sns
import matplotlib.pyplot as plt
happiness = pd.read_csv("../input/2017.csv")
happiness
happiness.shape
happiness.head(10)
happiness.corr()
sns_heatmap = sns.heatmap(happiness.corr(), xticklabels=happiness.corr().columns, yticklabels=happiness.corr().columns, annot=True)
sns_heatmap.set_title('Happiness Correlation Heatmap')
