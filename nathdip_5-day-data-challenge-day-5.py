# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

data_cereal = pd.read_csv('../input/cereal.csv')

data_cereal.describe()



# Any results you write to the current directory are saved as output.
manufacturers = data_cereal['mfr']

from collections import Counter

mfr_freqs = Counter(manufacturers)

mfr_freqs = list(mfr_freqs.values())

from scipy.stats import chisquare

chisquare(mfr_freqs)

import matplotlib.pyplot as plt

import seaborn as sns

plt.figure(figsize = (12, 9))

sns.countplot(x = 'mfr', hue = 'type', data = data_cereal).set_title('Bar plot visualization of cereal type across manufacturers')
list(mfr_freqs)