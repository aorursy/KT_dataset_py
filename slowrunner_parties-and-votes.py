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
data = pd.read_csv('../input/up_res.csv')
data
import matplotlib.pyplot as plt

%matplotlib inline
array = data['party'].unique()

array
parties = {}

for party in array:

    parties[party] = 0
for index, row in data.iterrows():

    party = row['party']

    prev = parties[party]

    parties[party] = row['votes'] + prev
parties
plt.figure(figsize=(13,6))

plt.bar(range(len(parties)), parties.values(), align='center')

plt.xticks(range(len(parties)), parties.keys())

plt.show()