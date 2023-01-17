import pandas as pd

import numpy as np

from pandas import DataFrame,Series

from collections import Counter

import matplotlib.pyplot as plt

import csv

e=pd.read_csv('../input/cities_r2.csv')

data=DataFrame(e)

literacy_rate=data[data['effective_literacy_rate_total']>90]

literacy_rate_states=literacy_rate['state_name']

literacy_rate_states=Counter(literacy_rate_states)

plt.figure(figsize=(11,8))

plt.bar(range(len(literacy_rate_states)), literacy_rate_states.values(),color='b')

plt.xticks(range(len(literacy_rate_states)),literacy_rate_states.keys(),rotation=35)

plt.show()
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