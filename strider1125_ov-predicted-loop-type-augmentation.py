import pandas as pd

import json

import numpy as np
train = pd.read_json('../input/stanford-covid-vaccine/train.json', lines=True)

test = pd.read_json('../input/stanford-covid-vaccine/test.json', lines=True)
pd_loop = json.load(open(r"/kaggle/input/ov-predicted-loop-type-augmentation/predicted_loop.json", "r"))
use_cols = ['id','sequence', 'structure']

all_samples = pd.concat([train[use_cols], test[use_cols]], ignore_index=True, sort=False)
p1 = json.load(open("/kaggle/input/ov-arnie-augmentation-structure-part1/part1.json"))

p2 = json.load(open("/kaggle/input/ov-arnie-augmentation-structure-part2/part2.json"))



result = []

for key, item in p1.items():

    for r in item:

        result.append([key, r[2], r[4]])

for key, item in p2.items():

    for r in item:

        result.append([key, r[2], r[4]])
df = pd.DataFrame(data=result, columns=['id', 'probability', 'exp_seq'])

df.drop_duplicates(inplace=True)

df = df[~df['exp_seq'].isin(['.'*107, '.'*130])]

# df.to_csv("possible_structures_arnie.csv", index=False)
t = pd.merge(df, all_samples[use_cols], how='left', on='id')

t = t[t['exp_seq']!=t['structure']]
tm = pd.merge(t, t[['id','probability']].groupby('id').max().reset_index().rename(columns={"probability": "mcc"}), how='left', on='id')

tm = tm[tm['probability']==tm['mcc']]
print(tm.shape)
tm['aug_loop_type'] = tm['id'].apply(lambda x: pd_loop.get(x))
tm.rename(columns={'exp_seq': "aug_structure", "sequence":"original_sequence", "structure":"original_structure", "mcc":"aug_mcc"}, inplace=True)
tm = tm[['id', 'original_sequence', 'original_structure', 'aug_mcc', 'aug_structure', 'aug_loop_type']]
tm.to_csv("addition_structure_loop_type_by_id.csv", index=False)