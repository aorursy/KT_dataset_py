# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from scipy.stats import ttest_ind

data = pd.read_csv('../input/cereal.csv')
data.columns
data
hot = data[data.type == 'H'].sodium

cold = data[data.type == 'C'].sodium



s_h = np.std(hot)

s_c = np.std(cold)

print(s_h, s_c)



n_h = len(hot)

n_c = len(cold)

print(n_h, n_c)
# Standard t-test (equal variance assumption)

ttest_ind(hot, cold, equal_var=True)
# Welch's t-test (unequal variance assumption)

ttest_ind(hot, cold, equal_var=False)
import seaborn as sns

import matplotlib.pyplot as plt



sns.distplot(cold, label='Cold')

sns.distplot(hot, label='Hot')



plt.title('Histograms for sodium content in cerials')

plt.xlabel('Sodium (mg)')

plt.legend()

plt.show()