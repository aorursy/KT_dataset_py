import numpy as np

import pandas as pd 

import gc

import json

import matplotlib.pyplot as plt
train_df = pd.read_csv("/kaggle/input/data-science-bowl-2019/train.csv")

# specs_df = pd.read_csv("/kaggle/input/data-science-bowl-2019/specs.csv")

test_df = pd.read_csv("/kaggle/input/data-science-bowl-2019/test.csv")

train_label_df = pd.read_csv("/kaggle/input/data-science-bowl-2019/train_labels.csv")
train_df
plt.hist(list(train_df["game_session"].value_counts()))
train_df["game_time"].value_counts()
gc.collect()
train_label_df[train_label_df.installation_id=="0006a69f"]
train_df[(train_df.installation_id=="0006a69f") & (train_df.title=="Mushroom Sorter (Assessment)") & (train_df.event_code==4100)]
train_df[(train_df.installation_id=="0006a69f") & (train_df.title=="Bird Measurer (Assessment)") & (train_df.event_code==4110)]
train_df_clear = train_df[((train_df.event_code==4100)|(train_df.event_code==4110))

                          &(train_df.event_data.str.contains("true"))]

train_df_fail = train_df[((train_df.event_code==4100)|(train_df.event_code==4110))

                         &(train_df.event_data.str.contains("false"))]
train_df_clear_g = train_df_clear.groupby(["installation_id"]).count()["event_id"]

train_df_fail_g = train_df_fail.groupby(["installation_id"]).count()["event_id"]
train_df_clear_g
sample_submission = pd.read_csv("/kaggle/input/data-science-bowl-2019/sample_submission.csv")
test_df
test_df_clear = test_df[((test_df.event_code==4100)|(test_df.event_code==4110))

                          &(test_df.event_data.str.contains("true"))]

test_df_fail = test_df[((test_df.event_code==4100)|(test_df.event_code==4110))

                         &(test_df.event_data.str.contains("false"))]



test_df_clear_g = test_df_clear.groupby(["installation_id"]).count()["event_id"]

test_df_fail_g = test_df_fail.groupby(["installation_id"]).count()["event_id"]
test_clear_dic=dict(zip(test_df_clear_g.index,list(test_df_clear_g)))

test_fail_dic=dict(zip(test_df_fail_g.index,list(test_df_fail_g)))
for i in range(len(sample_submission)):

    id = sample_submission["installation_id"][i]

    fail = test_fail_dic[id] if id in test_fail_dic else 0

    clear = test_clear_dic[id] if id in test_clear_dic else 0

    if fail+clear!=0:

        score = clear/(fail+clear)

        if score>0.9:

            sample_submission["accuracy_group"][i]=3

        elif score>0.6:

            sample_submission["accuracy_group"][i]=2

        elif score>0.3:

            sample_submission["accuracy_group"][i]=1

        else:

            sample_submission["accuracy_group"][i]=0

    else:

        sample_submission["accuracy_group"][i]=1
sample_submission
sample_submission.to_csv("submission.csv",index=False)