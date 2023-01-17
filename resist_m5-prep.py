import tensorflow as tf

import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split, KFold

import itertools

import random

import os
UNITS_PATH = os.path.join("..", "input", "m5-forecasting-uncertainty", "sales_train_evaluation.csv")

WEIGHTS_PATH = os.path.join("..", "input", "m5git", "validation", "weights_validation.csv")

SEQ_LEN = 1941
store_agg = pd.read_csv(

    UNITS_PATH

).astype({f"d_{n}": np.float32 for n in range(1, SEQ_LEN)})



# print(store_agg[store_agg.isna().apply(any, axis='columns')])
CORRECT_COL_ORDER = store_agg.columns

context_cols = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]



hier_seqs = store_agg.sum(

).to_frame(

).transpose(

).assign(

    id="Total_X", **{col: "agg" for col in context_cols}

)[CORRECT_COL_ORDER]
def agg_and_adjust(levels):

    agg_cols = [col for col in context_cols if col not in levels]

    return store_agg.groupby(

        levels,

        as_index=False

    ).sum(

    ).assign(

        id=lambda x: x[levels[0]] + "_" + (x[levels[1]] if len(levels) == 2 else "X"),

        **{col: "agg" for col in agg_cols}

    )
levels = [["state_id", "cat_id"], ["state_id", "dept_id"],

          ["store_id", "cat_id"], ["store_id", "dept_id"],

          ["state_id"], ["store_id"], ["cat_id"], ["dept_id"]]

hier_seqs = hier_seqs.append([agg_and_adjust(level) for level in levels],

                             ignore_index=True, sort=False)    

frames = [hier_seqs[CORRECT_COL_ORDER]]
store_state_agg = agg_and_adjust(["item_id"])[CORRECT_COL_ORDER]

frames.extend([store_state_agg[store_state_agg.item_id.map(lambda x: x[:-4]) == dept]

                  for dept in store_agg.dept_id.unique()])
def get_strata(df, levels):

    cuts = itertools.product(store_agg[levels[0]].unique(),

                             store_agg[levels[1]].unique())

    return [(df.item_id.map(lambda x: x[:-4]) == dept)

                & (df[levels[0]] == level_0) for level_0, dept in cuts]
state_agg = agg_and_adjust(["state_id", "item_id"])[CORRECT_COL_ORDER]

frames.extend([state_agg[strata] for strata in get_strata(state_agg, ["state_id", "dept_id"])])
store_agg.id = store_agg.id.map(lambda x: x[:-11])

frames.extend([store_agg[strata] for strata in get_strata(store_agg, ["store_id", "dept_id"])])
weights = pd.read_csv(WEIGHTS_PATH

).assign(

    id=lambda x: x.Agg_Level_1 + "_" + x.Agg_Level_2

).drop(

    columns=["Agg_Level_1", "Agg_Level_2", "Level_id"]

).rename(

    columns={"Weight": "weights"}

)
col_vals = {

        "item_id": ["agg".encode("utf-8")],

        "dept_id": list(range(3)),

        "cat_id": list(range(3)),

        "store_id": list(range(4)),

        "state_id": list(range(3))

    }



vals = {

    "_1": [0], "_2": [1], "_3": [2],

    "_4": [3], "WI": [0], "CA": [1],

    "TX": [2], "HOUSEHOLD": [0],

    "HOBBIES": [1], "FOODS": [2]        

}



def parse(col, row):

    val = row[col]

    if val == "agg":

        out = col_vals[col]

        if col == "dept_id" and row["cat_id"] not in ["FOODS", "agg"]:

            out = out[:-1]

        elif col == "store_id" and row["state_id"] in ["TX", "WI"]:

            out = out[:-1]

            

        return out

    else:

        if col in {"dept_id", "store_id"}:

            val = val[-2:]

        

        if col in {"id", "item_id"}:

            return [val.encode("utf-8")]

        else:

            return vals[val]
def write_to_tfrecord(row, file):

    feature = {col: tf.train.Feature(

        bytes_list=tf.train.BytesList(value=parse(col, row))

    ) for col in ["id", "item_id"]}

    

    feature.update({col: tf.train.Feature(

        int64_list=tf.train.Int64List(value=parse(col, row))

    ) for col in ["dept_id", "cat_id", "store_id", "state_id"]})

    

    feature["weights"] = tf.train.Feature(

        float_list=tf.train.FloatList(value=[row["weights"]]))

    feature["units"] =  tf.train.Feature(

        float_list=tf.train.FloatList(value=list(row.loc["d_1":f"d_{SEQ_LEN}"])))

    

    example = tf.train.Example(

        features=tf.train.Features(feature=feature))

    file.write(example.SerializeToString())
def stratify_uniformly(in_frames):

    random.shuffle(in_frames)

    train, test_frame = train_test_split(in_frames.pop(), test_size=0.2)

    len_train, len_test = train.shape[0], test_frame.shape[0]

    k_fold = KFold(shuffle=True)

    out_frames = [train.iloc[idx]

                      for _, idx in k_fold.split(train)]

    for frame in in_frames:

        train, test = train_test_split(frame, test_size=0.2)

        len_train += train.shape[0]

        len_test += test.shape[0]

        if len_test * 4 - len_train >= 5:

            train = train.append(test.iloc[-1], ignore_index=True)

            test = test.iloc[:-1]

            len_test -= 1

            len_train += 1

        

        test_frame = test_frame.append(test, ignore_index=True)

        out_frames.sort(key=lambda x: x.shape[0])

        new_frames = [train.iloc[idx]

                          for _, idx in k_fold.split(train)]

        new_frames.sort(key=lambda x: x.shape[0], reverse=True) 

        out_frames = [pd.concat(frames, ignore_index=True)

                          for frames in zip(out_frames, new_frames)]

        

    out_frames.append(test_frame)

    return out_frames
stratified_frames = stratify_uniformly(frames)

for n, frame in enumerate(stratified_frames):

    frame = weights[["weights", "id"]].join(

        frame.set_index("id"),

        how="right", on="id"

    )

    if n < 5:

        path = f"train_{n + 1}.tfrecord"

    else:

        path = "test.tfrecord"

    

    with tf.io.TFRecordWriter(path) as writer:

        frame.apply(write_to_tfrecord, axis='columns', args=[writer])