import numpy as np 

import pandas as pd

cols = ["y","technical_20","fundamental_53","technical_30","technical_27","derived_0","fundamental_42","fundamental_48"]

cols_to_read = cols + ["timestamp"]

df = pd.read_hdf("../input/train.h5")[cols_to_read]

print("Data shape: {}".format(df.shape))
import matplotlib.pyplot as plt

# idea of timediff periods taken from anokas notebook: https://www.kaggle.com/anokas/two-sigma-financial-modeling/two-sigma-time-travel-eda/notebook

timediff = df.groupby('timestamp')['timestamp'].count().diff()

for col in cols:

    data = df.groupby('timestamp')[col].mean()

    plt.figure(figsize=(9, 5))

    plt.plot(data)

    plt.xlabel('Timestamp')

    plt.ylabel('Mean of %s' % col)

    plt.title('Mean value of %s over time' % col)

    for i in timediff[timediff > 5].index:

        plt.axvline(x=i, linewidth=0.25, color='red')
for col in cols:

    data = df.dropna().groupby('timestamp')[col].median()

    plt.figure(figsize=(9, 5))

    plt.plot(data)

    plt.xlabel('Timestamp')

    plt.ylabel('Median of %s' % col)

    plt.title('Median value of %s over time' % col)

    for i in timediff[timediff > 5].index:

        plt.axvline(x=i, linewidth=0.25, color='red')
for col in cols:

    nancnt = df[col].isnull().sum()

    print("%16s %s nan" % (col, str(nancnt)))