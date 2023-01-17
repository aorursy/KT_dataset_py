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
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline
train_df = pd.read_csv("../input/train.csv")
train_df.head()
train_df = train_df.drop('PassengerId', axis=1)
train_df[['Fare']].max()
train_df[['Fare', 'Survived']].sort_values('Fare').head()
fare_bins = []

bin_init = 0

num_bins = 50.00

bin_diff = train_df[['Fare']].max() / num_bins

for i in range(0, int(num_bins) + 1):

    tmp_dict = {i: [bin_init + bin_diff * i, bin_init + bin_diff * (i + 1)]}

    fare_bins.append(tmp_dict)

fare_bins
fare_col = train_df['Fare']

bands = []

for row in fare_col.iteritems():

    fare = row[1]

    for band in fare_bins:

        bin = list(band.values())[0]

        band_key = list(band.keys())[0]

        print(bin, band_key)

#         if fare >= bin[0] and fare < bin[1]:

#             bands.add(list(band.keys())[0])

#             break