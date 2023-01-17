import numpy as np

import pandas as pd

import os

import matplotlib as mpl

from matplotlib import pyplot as plt

import sys



df_train = pd.read_csv("../input/train.csv")

df_train_contr = pd.read_csv("../input/train_controls.csv").drop(labels="well_type", axis=1)

df_test = pd.read_csv("../input/test.csv")

df_test_contr = pd.read_csv("../input/test_controls.csv").drop(labels="well_type", axis=1)
df_train.head(2)
df_train_contr.head(2)
# extract row and column number from well

df_train["row"] = df_train["well"].apply(lambda x: ord(x[0].lower()) - 96)

df_train["col"] = df_train["well"].apply(lambda x: int(x[1:]))

df_train_contr["row"] = df_train_contr["well"].apply(lambda x: ord(x[0].lower()) - 96)

df_train_contr["col"] = df_train_contr["well"].apply(lambda x: int(x[1:]))

df_test_contr["row"] = df_test_contr["well"].apply(lambda x: ord(x[0].lower()) - 96)

df_test_contr["col"] = df_test_contr["well"].apply(lambda x: int(x[1:]))

df_train.sample(n=5)
# create ordered list of treatment sirnas with group1+group2+...+group4

sirnas = []

exp = "HEPG2-03" # select experiment that has all sirnas

df_exp = df_train.groupby("experiment").get_group(exp)

for plate, df_exp_pl in df_exp.groupby("plate"):

    ss = sorted(df_exp_pl["sirna"].unique())

    sirnas += ss

    print("Plate {} has {} sirnas.".format(plate, df_exp_pl["sirna"].nunique()))

    print("   First 10 in ordered group:", ss[:10])
# write sirna groups to a dataframe to save as output

pd.DataFrame(data={"sirna" : sirnas, 

                   "group" : [i for i in range(1,5) for j in range(277)]}).to_csv("sirna_groups.csv", index=False)
# assign unique colors to treatment sirnas

sirnas_colormaps = {1 : "Blues", 2 : "Greens", 3 : "Purples", 4 : "Reds"}

colors = []

for plate in [1,2,3,4]:

    colors += [mpl.cm.get_cmap(sirnas_colormaps[plate])(i) for i in np.linspace(0., 1., 277)]

    

sirnas_colors_dict = dict(zip(sirnas, colors))

df_train["color"] = df_train["sirna"].map(sirnas_colors_dict)

df_train.head()
# create ordered list of control sirnas

contr_sirnas = sorted(df_train_contr["sirna"].unique())

print(f"{len(contr_sirnas)} control sirnas.")
# assign unique colors to control sirnas

colors = [mpl.cm.get_cmap("hsv")(i) for i in np.linspace(0., 1., 31)]

contr_sirnas_colors_dict = dict(zip(contr_sirnas, colors))

df_train_contr["color"] = df_train_contr["sirna"].map(contr_sirnas_colors_dict)

df_test_contr["color"] = df_test_contr["sirna"].map(contr_sirnas_colors_dict)

df_train_contr.head()
df_pattern = pd.DataFrame(index=df_train["experiment"].unique())

df_pattern["pattern"] = ""

for exp, df_exp in df_train.groupby("experiment"):

    pattern = ""

    for plate, df_exp_pl in df_exp.groupby("plate"):

        sirna_sample = df_exp_pl["sirna"].values[0]

        group_sirna_sample = sirnas.index(sirna_sample) // 277 + 1

        pattern += str(group_sirna_sample)

    df_pattern.loc[exp, "pattern"] = pattern



df_pattern.reset_index(inplace=True)

df_pattern.columns = ["experiment", "pattern"]

df_pattern
df_pattern.groupby("pattern").size()
for pattern, df_pattern_pattern in df_pattern.groupby("pattern"):

    print("====================================================================")

    print(f"Pattern {pattern}\n")

    experiments = df_pattern_pattern["experiment"]

    for exp in experiments:

        df_exp = df_train.groupby("experiment").get_group(exp)

        fig, axs = plt.subplots(1, 8, figsize=(16,3))

        for plate, df_exp_pl in df_exp.groupby("plate"):

            if plate == 1:

                axs[plate-1].set_ylabel(exp)

            axs[plate-1].scatter(df_exp_pl["row"], df_exp_pl["col"], color=df_exp_pl["color"], s=30)

            axs[plate-1].set_title(f"PL. {plate} treat")

            df_exp_pl_contr = df_train_contr[(df_train_contr["experiment"]==exp) & (df_train_contr["plate"]==plate)]

            axs[plate-1+4].scatter(df_exp_pl_contr["row"], df_exp_pl_contr["col"], color=df_exp_pl_contr["color"], s=30)

            axs[plate-1+4].set_title(f"PL. {plate} contr")

        plt.show()
# controls seem to appear in a random order in the same "scheme" of wells

# sometimes negative controls appear in a well that is normally dedicated to a treatment (it's a failed treatment)

# pick "real" control wells from a plate that seems not to have failed treatments or anomalies

control_wells = df_train_contr.loc[(df_train_contr["experiment"]=="RPE-06") & (df_train_contr["plate"]==1), "well"].values

control_wells
experiment_plate_contr_list = []

pattern_contr_list = []

for exp_pl, df_exp_pl in df_train_contr[df_train_contr["well"].isin(control_wells)].groupby(["experiment", "plate"]):

    experiment_plate_contr_list.append(exp_pl)

    # df_exp_pl is already sorted by row and column

    pattern_contr_list.append("_".join(df_exp_pl["sirna"].astype("str").values.tolist()))

df_pattern_contr = pd.DataFrame(data={"experiment_plate" : experiment_plate_contr_list,

                                      "pattern" : pattern_contr_list})

df_pattern_contr.head()
df_pattern_contr.groupby("pattern").size().sort_values(ascending=False).head(5)
p1 = "1138_1108_1109_1110_1111_1112_1113_1114_1115_1116_1117_1118_1119_1120_1121_1122_1123_1124_1125_1126_1127_1128_1129_1130_1131_1132_1133_1134_1135_1136_1137"

p2 = "1138_1108_1109_1110_1111_1112_1113_1114_1115_1116_1117_1118_1138_1120_1121_1122_1123_1124_1125_1126_1127_1128_1129_1130_1131_1132_1133_1134_1135_1136_1137"

df_pattern_contr[df_pattern_contr["pattern"].isin([p1,p2])]
# visualize controls in test

for exp, df_exp in df_test_contr.groupby("experiment"):

    fig, axs = plt.subplots(1, 4, figsize=(8,3))

    for plate, df_exp_pl in df_exp.groupby("plate"):

        if plate == 1:

            axs[plate-1].set_ylabel(exp)

        axs[plate-1].scatter(df_exp_pl["row"], df_exp_pl["col"], color=df_exp_pl["color"], s=30)

        axs[plate-1].set_title(f"PL. {plate} contr")

    plt.show()
experiment_plate_contr_test_list = []

pattern_contr_test_list = []

for exp_pl, df_exp_pl in df_test_contr[df_test_contr["well"].isin(control_wells)].groupby(["experiment", "plate"]):

    experiment_plate_contr_test_list.append(exp_pl)

    # df_exp_pl is already sorted by row and column

    pattern_contr_test_list.append("_".join(df_exp_pl["sirna"].astype("str").values.tolist()))

df_pattern_contr_test = pd.DataFrame(data={"experiment_plate" : experiment_plate_contr_test_list,

                                           "pattern" : pattern_contr_test_list})

df_pattern_contr_test.head()
df_pattern_contr_test.groupby("pattern").size().sort_values(ascending=False).head(5)
p1 = "1138_1108_1109_1110_1111_1112_1113_1114_1115_1116_1117_1118_1119_1120_1121_1122_1123_1124_1125_1126_1127_1128_1129_1130_1131_1132_1133_1134_1135_1136_1137"

df_pattern_contr_test[df_pattern_contr_test["pattern"].isin([p1])]
experiment_plate_treat_list = []

pattern_treat_list = []

for exp_pl, df_exp_pl in df_train.groupby(["experiment", "plate"]):

    experiment_plate_treat_list.append(exp_pl)

    # df_exp_pl is already sorted by row and column

    pattern_treat_list.append("_".join(df_exp_pl["sirna"].astype("str").values.tolist()))

df_pattern_treat = pd.DataFrame(data={"experiment_plate" : experiment_plate_treat_list,

                                      "pattern" : pattern_treat_list})

df_pattern_treat.head()
df_pattern_treat.groupby("pattern").size().sort_values(ascending=False).head(5)
from difflib import SequenceMatcher



def similar(a, b):

    return SequenceMatcher(None, a, b).ratio()



m = np.zeros((len(pattern_treat_list), len(pattern_treat_list)))

for i, p1 in enumerate(pattern_treat_list):

    for j, p2 in enumerate(pattern_treat_list):

        if i<j:

            s = similar(p1, p2)

            m[i,j] = s

            if s>0.1:

                print(f"- Match found at {experiment_plate_treat_list[i]} and {experiment_plate_treat_list[j]}")

                print(f"     p1 = {p1}")

                print(f"     p2 = {p2}")

                print(f"     simil = {s}")