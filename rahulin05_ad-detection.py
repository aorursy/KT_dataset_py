import numpy as np

import pandas as pd

data = pd.read_csv("../input/add.csv",low_memory=False)
data.shape
data.info()
# Reading first 20 rows

data.head(20)
# Check whether a given value is a missing value, if yes change it to NaN

def toNum(cell):

    try:

        return np.float(cell)

    except:

        return np.nan

    

# Apply missing value check to a column / Pandas series

def seriestoNum(series):

    return series.apply(toNum)
# Missing  values have been replaced by Nan

train_data=data.iloc[0:,0:-1].apply(seriestoNum)

train_data.head(20)
# Apply pandas dropna function for removing all the missing value rows

train_data=train_data.dropna()

train_data.head(20)
# Converting Training label to 0 or 1

def toLabel(str):

    if str=="ad.":

        return 1

    else:

        return 0

    

train_labels=data.iloc[train_data.index,-1].apply(toLabel)

train_labels
# using row 50 t0 2200 for training

from sklearn.svm import LinearSVC 

clf = LinearSVC()

clf.fit(train_data[50:2200],train_labels[50:2200])
clf.predict(train_data.iloc[12].values.reshape(1,-1))
clf.predict(train_data.iloc[-1].values.reshape(1,-1))