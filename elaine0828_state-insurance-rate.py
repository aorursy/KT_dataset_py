# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





# Any results you write to the current directory are saved as output.



csv_chunks = pd.read_csv("../input/Rate.csv",iterator=True,chunksize = 1000)

Rate = pd.concat(chunk for chunk in csv_chunks)
Rates = Rate[[ 'StateCode',

               'BusinessYear',

               'Age',

               'IndividualRate']]
# Outlier detect

def get_median_filtered(signal, threshold=5):

    signal = signal.copy()

    difference = np.abs(signal - np.median(signal))

    median_difference = np.median(difference)

    if median_difference == 0:

        s = 0

    else:

        s = difference / float(median_difference)

    mask = s > threshold

    signal[mask] = np.median(signal)

    return signal
import matplotlib.pyplot as plt



figsize = (10, 10)

kw = dict(marker='o', linestyle='none', color='r', alpha=0.3)



Rates ['aveRate'] = get_median_filtered(Rates['IndividualRate'].values, threshold=5)



outlier_idx = np.where(Rates['aveRate'].values != Rates['IndividualRate'].values)[0]



fig1, ax1 = plt.subplots(figsize=figsize)

Rates['IndividualRate'].plot()

Rates['IndividualRate'][outlier_idx].plot(**kw)

plt.show()
normal = Rates['IndividualRate'] < 9997
Rate_Cleaned = Rates[normal]
RateAverage = Rate_Cleaned.groupby(['StateCode', 'BusinessYear'])['IndividualRate'].mean()

RateAverage = pd.DataFrame(RateAverage)

RateAverage.reset_index(inplace=True)  
import seaborn as sns



fig, ax = plt.subplots(figsize=(10, 40))

sns.barplot(ax=ax,y="StateCode", x="IndividualRate", hue="BusinessYear",data=RateAverage,palette="Paired")