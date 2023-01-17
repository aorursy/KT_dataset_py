import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline



with pd.HDFStore("../input/train.h5", "r") as train:

    df = train.get("train")
timemax = max(df["timestamp"])

timemin = min(df["timestamp"])

xlim = [timemin, timemax]



for asset in df["id"].unique() :

    #print(df["id"=asset])

    x = df[df["id"]==asset]["timestamp"]

    diffy = df[df["id"]==asset]["y"]

    y = np.cumsum(diffy)

    

    plt.figure(figsize=(9,1))

    plt.plot(x, y, 'k-')

    plt.plot(x, diffy, 'b-')

    plt.xlim(xlim)

    plt.title("ID # %s" %(asset),size=10)

    

    tmax = max(x)

    ax = plt.subplot()

    ax.axvline(tmax, color='r', linestyle='--')



    if asset > 50 :

        break;