# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

from sklearn.cluster import DBSCAN

import matplotlib.pyplot as plt

from collections import Counter

import warnings

warnings.filterwarnings('ignore')
hbinint = 10

vbinint = 0.3

label_dict = {

    0: "normal",

    1: "rotor < 5",

    2: "power neg",

    3: "h outlier",

    5: "v low stack",

    6: "v dbscan",

}
# laod data

df = pd.read_csv("../input/wind-cluster/dataset/1.csv")

df["label"] = 0
def plot(df, num = 1, no_v_move = True):

    # arg "no_v_move": 

    #     whether the data has been passed to "v_move" function



    # take windnumber 1 fof example by default

    subdf = df[df.WindNumber == num]

    if no_v_move: subdf["OldWindSpeed"] = subdf.WindSpeed



    fig, ax = plt.subplots(2, 2, figsize = (8, 8))

    # title

    _ = ax[0][0].set_title(num); _ = ax[1][0].set_title(num)

    _ = ax[0][1].set_title(num); _ = ax[1][1].set_title(num)



    # scatter of original windspeed ~ power (with outlier labels)

    scatter = ax[0][0].scatter(subdf.OldWindSpeed, subdf.Power, s = 0.1, c = subdf.label, cmap = "tab10")

    _ = ax[0][0].legend(handles=scatter.legend_elements()[0], labels=label_dict.values())



    # scatter of original windspeed ~ power (nomal data only)

    _ = ax[0][1].scatter(subdf.OldWindSpeed[subdf.label == 0], subdf.Power[subdf.label == 0], s = 0.1)



    # scatter of v_moved windspeed ~ power (with outlier labels)

    scatter = ax[1][0].scatter(subdf.WindSpeed, subdf.Power, s = 0.1, c = subdf.label, cmap = "tab10")

    _ = ax[1][0].legend(handles=scatter.legend_elements()[0], labels=label_dict.values())



    # scatter of v_moved windspeed ~ power (nomal data only)

    _ = ax[1][1].scatter(subdf.WindSpeed[subdf.label == 0], subdf.Power[subdf.label == 0], s = 0.1) 

    fig.show()

    # plt.close(fig)



# a sub-df for demo

demo_df = df[df.WindNumber == 1]
# remove negative power (label 2) and low rotorspeed (label 1)

def remove_neg(df):

    keep = df[df.label != 0]

    df = df[df.label == 0]

    df.loc[df.RotorSpeed < 4, "label"] = 1

    df.loc[df.Power < 0, "label"] = 2

    return pd.concat([df, keep], sort = False).reset_index(drop = True)

demo_df = remove_neg(demo_df)

plot(demo_df)
# slice power by binwidth (hbinint)

# remove 3IQR outlier of each bin (label 3)

def remove_h_out(df):

    keep = df[df.label != 0]

    df = df[df.label == 0]

    bins = pd.cut(df.Power, np.arange(-1000, 3000, hbinint))

    df["hbins"] = bins

    groups = []

    for name, group in df.groupby("hbins"):

        if group.shape[0] == 0:

            continue

        q1 = np.quantile(group.WindSpeed, 0.25)

        q3 = np.quantile(group.WindSpeed, 0.75)

        iqr = q3 - q1

        fl = q1 - 1.5 * iqr

        fu = q3 + 1.5 * iqr

        group.loc[np.bitwise_or(group.WindSpeed < fl, group.WindSpeed > fu),"label"] = 3

        groups.append(group)

    df = pd.concat(groups, sort = False).reset_index(drop = True)

    return pd.concat([df, keep], sort = False).reset_index(drop = True)



demo_df = remove_h_out(demo_df)

plot(demo_df)
# slice windspeed by binwidth (vbinint)

# remove 3IQR outlier of each bin (label 4)

def remove_v_out(df):

    keep = df[df.label != 0]

    df = df[df.label == 0]

    bins = pd.cut(df.WindSpeed, np.arange(-15, 30, vbinint))

    df["vbins"] = bins

    groups = []

    for name, group in df.groupby("vbins"):

        if group.shape[0] == 0:

            continue

        q1 = np.quantile(group.Power, 0.25)

        q3 = np.quantile(group.Power, 0.75)

        fl = 2.5 * q1 - 1.5 * q3

        fu = 2.5 * q3 - 1.5 * q1

        group.loc[group.Power > fu,"label"] = 4

        groups.append(group)

    df = pd.concat(groups, sort = False).reset_index(drop = True)

    return pd.concat([df, keep], sort = False).reset_index(drop = True)



demo_df = remove_v_out(demo_df)

plot(demo_df)
# move data to be vertivcal aligned

# save the original WindSpeed to OldWindSpeed column

# the new WindSpeed columns is aligned

def v_move(df):

    num = np.unique(df.WindNumber)[0]

    df["OldWindSpeed"] = df.WindSpeed

    bins = pd.cut(df.Power, np.arange(-1000, 3000, hbinint))

    df["hbins"] = bins

    groups = []

    for name, group in df.groupby("hbins"):

        if np.sum(group.label == 0) == 0:

            groups.append(group)

            continue

        group.WindSpeed -= np.quantile(group.WindSpeed[group.label == 0], 0.05)

        groups.append(group)

    df = pd.concat(groups, sort = False).reset_index(drop = True)

    return df.reset_index(drop = True)



demo_df = v_move(demo_df)

plot(demo_df, no_v_move = False)
# remove some lower stack points except the highest stack (where the power reaches the rated power ~ 2000)

# this kind of outlier is labeled with 5

# the lower stack is defined as lower than 0.9 *max(power) ~= 1900

def remove_v_out_low(df):

    num = np.unique(df.WindNumber)[0]

    keep = df[df.label != 0]

    df = df[df.label == 0]

    bins = pd.cut(df.WindSpeed, np.arange(-15, 30, vbinint))

    df["vbins"] = bins

    groups = []

    for name, group in df.groupby("vbins"):

        if group.shape[0] == 0:

            continue

        if name.left > 2:

            group.loc[group.Power < np.max(df.Power) * 0.9, "label"] = 5

        groups.append(group)

    df = pd.concat(groups, sort = False).reset_index(drop = True)

    return pd.concat([df, keep], sort = False).reset_index(drop = True)



demo_df = remove_v_out_low(demo_df)

plot(demo_df, no_v_move = False)
# to improve the accuracy of removing low stack points

# I use DBSCAN clustering the find the low stack clusters and remove them

# this kind of outliers are labeled with 6

# SORRY for the confusing codes to distinguish normal points cluster from abnormal clusters ...

def remove_v_dbscan(df):

    num = np.unique(df.WindNumber)[0]

    keep = df[df.label != 0]

    df = df[df.label == 0]

    bins = pd.cut(df.WindSpeed, np.arange(-15, 30, vbinint))

    df["vbins"] = bins

    groups = []

    dbscan = DBSCAN(40, min_samples = 5)

    for name, group in df.groupby("vbins"):

        if group.shape[0] < 5 or name.left < 1.5:

            groups.append(group)

            continue

        label = dbscan.fit_predict(group[["Power"]])

        group["cls"] = label

        clsmean = group.groupby("cls").apply(lambda x: np.mean(x.Power))

        if len(clsmean) != 1:

            clsmean = clsmean[clsmean.index != -1]

        maxcls = clsmean.sort_values().index[-1]

        group["label"] = [0 if l == maxcls else 6 for l in label]

        group = group.drop(columns = ["cls"])

        groups.append(group)

    df = pd.concat(groups, sort = False).reset_index(drop = True)

    return pd.concat([df, keep], sort = False).reset_index(drop = True)



demo_df = remove_v_dbscan(demo_df)

plot(demo_df, no_v_move = False)
# some functions to reassign the label

def withdraw_v_out(df):

    df.loc[df.label == 4, "label"] = 0

    return df



def withdraw_h_out(df):

    df.loc[df.label == 3, "label"] = 0

    return df
# a combine of several remove functions

# you can custom define the order of these functions

def remove(df):

    df = remove_neg(df)

    df = remove_h_out(df)

    df = remove_v_out(df)

    df = v_move(df)

    df = remove_v_out_low(df)

    df = remove_v_dbscan(df)

    df = withdraw_v_out(df)

    df = withdraw_h_out(df)

    df = remove_h_out(df)

    return df



demo_df = df[df.WindNumber == 1]

demo_df = remove(demo_df)

plot(demo_df, no_v_move = False)
# do this for all windnumber

sub = []

for windnumber in np.unique(df.WindNumber):

    subdf = remove(df[df.WindNumber == windnumber])

    sub.append(subdf)



sub = pd.concat(sub)
sub = sub[["WindNumber", "Time","label"]]

print(Counter(sub.label))

sub.label = (sub.label != 0).astype(np.int)

sub