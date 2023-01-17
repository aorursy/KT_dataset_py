import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from sklearn import metrics
def metric(df, preds):

    df["prediction"] = preds

    maes = []

    for t in df.type.unique():

        y_true = df[df.type==t].scalar_coupling_constant.values

        y_pred = df[df.type==t].prediction.values

        mae = np.log(metrics.mean_absolute_error(y_true, y_pred))

        maes.append(mae)

    return np.mean(maes)
train = pd.read_csv("../input/train.csv")
metric(train, np.zeros(len(train)))