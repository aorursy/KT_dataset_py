import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



sns.set_style('darkgrid')

sns.set_palette('bone')



train = pd.read_csv('../input/pubg-finish-placement-prediction/train_V2.csv')



fig, ax = plt.subplots(1, 2, figsize=(12, 4))



train.groupby('matchId')['matchType'].first().value_counts().plot.bar(ax=ax[0])



'''

solo  <-- solo,solo-fpp,normal-solo,normal-solo-fpp

duo   <-- duo,duo-fpp,normal-duo,normal-duo-fpp,crashfpp,crashtpp

squad <-- squad,squad-fpp,normal-squad,normal-squad-fpp,flarefpp,flaretpp

'''

mapper = lambda x: 'solo' if ('solo' in x) else 'duo' if ('duo' in x) or ('crash' in x) else 'squad'

train['matchType'] = train['matchType'].apply(mapper)

train.groupby('matchId')['matchType'].first().value_counts().plot.bar(ax=ax[1])
import os

import numpy as np

import pandas as pd



print(os.listdir("../input/"))



# Any results you write to the current directory are saved as output.

df_sub = pd.read_csv("../input/lightgbm-baseline/submission_adjusted.csv")

df_sub2 = pd.read_csv("../input/pubg-lgb-ensamble-lvl-1/submission_v8.csv")

df_sub3 = pd.read_csv("../input/pubg-nn-ensamble-lvl-1/submission_nn_ensamble_v5.csv")

df_test = pd.read_csv("../input/pubg-finish-placement-prediction/test_V2.csv")



# STACK

df_sub["winPlacePerc"] = (df_sub2["winPlacePerc"] + df_sub3["winPlacePerc"]) / 2



df_sub = df_sub[["Id", "winPlacePerc"]]

df_test = pd.read_csv("../input/pubg-finish-placement-prediction/test_V2.csv")





# Restore some columns

df_sub = df_sub.merge(df_test[["Id", "matchId", "groupId", "maxPlace", "numGroups"]], on="Id", how="left")



# Sort, rank, and assign adjusted ratio

df_sub_group = df_sub.groupby(["matchId", "groupId"]).first().reset_index()

df_sub_group["rank"] = df_sub_group.groupby(["matchId"])["winPlacePerc"].rank()

df_sub_group = df_sub_group.merge(

    df_sub_group.groupby("matchId")["rank"].max().to_frame("max_rank").reset_index(), 

    on="matchId", how="left")

df_sub_group["adjusted_perc"] = (df_sub_group["rank"] - 1) / (df_sub_group["numGroups"] - 1)



df_sub = df_sub.merge(df_sub_group[["adjusted_perc", "matchId", "groupId"]], on=["matchId", "groupId"], how="left")

df_sub["winPlacePerc"] = df_sub["adjusted_perc"]



# Deal with edge cases

df_sub.loc[df_sub.maxPlace == 0, "winPlacePerc"] = 0

df_sub.loc[df_sub.maxPlace == 1, "winPlacePerc"] = 1



# Align with maxPlace

# Credit: https://www.kaggle.com/anycode/simple-nn-baseline-4

subset = df_sub.loc[df_sub.maxPlace > 1]

gap = 1.0 / (subset.maxPlace.values - 1)

new_perc = np.around(subset.winPlacePerc.values / gap) * gap

df_sub.loc[df_sub.maxPlace > 1, "winPlacePerc"] = new_perc



# Edge case

df_sub.loc[(df_sub.maxPlace > 1) & (df_sub.numGroups == 1), "winPlacePerc"] = 0

assert df_sub["winPlacePerc"].isnull().sum() == 0



df_sub["winPlacePerc"] = df_sub["winPlacePerc"]





df_sub[["Id", "winPlacePerc"]].to_csv("submission_ensemble_lvl2_v4.csv", index=False)