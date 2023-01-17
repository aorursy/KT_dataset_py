# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

from scipy.stats import ttest_ind

import matplotlib.pyplot as plt



# Any results you write to the current directory are saved as output.
cereal_df = pd.read_csv("../input/cereal.csv")

sodium_hot = cereal_df["sodium"][cereal_df['type'] == 'H']

sodium_cold = cereal_df["sodium"][cereal_df['type'] == 'C']
ttest_ind(sodium_hot,sodium_cold,equal_var='False')
plt.hist(sodium_cold,alpha=0.5,label='cold')

plt.hist(sodium_hot,label = 'hot')

plt.legend(loc='upper right')
