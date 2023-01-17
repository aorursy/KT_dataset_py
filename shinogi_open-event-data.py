import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import sklearn as skl

from sklearn import preprocessing

import lightgbm as lgbm



import json

import pickle

import gc
dl_train_df          = pd.read_csv("/kaggle/input/data-science-bowl-2019/train.csv", sep=",", header=0, quotechar="\"")

dl_test_df           = pd.read_csv("/kaggle/input/data-science-bowl-2019/test.csv", sep=",", header=0, quotechar="\"")
dl_train_df["data_type"] = "train"

dl_test_df["data_type"] = "test"

base_event_data_df = pd.concat([dl_train_df[["data_type", "installation_id", "event_data"]], dl_test_df[["data_type", "installation_id","event_data"]]], axis=0)

base_event_data_df = base_event_data_df.reset_index()

base_event_data_df = base_event_data_df.rename(columns={"index": "event_data_id"})

del dl_train_df, dl_test_df
base_event_data_df.shape
event_data_set = set()

for x in base_event_data_df.event_data.values:

    event_data_set = event_data_set | set(json.loads(x).keys())

event_data_columns = list(event_data_set)
len(event_data_columns)
event_data_columns
base_event_data_df["event_data_cnt"] = base_event_data_df.event_data.apply(lambda x : len(json.loads(x)))
base_event_data_df.event_data_cnt.unique()
base_event_data_df[["event_data_cnt", "event_data_id"]].groupby(["event_data_cnt"]).count().T
base_event_data_df.head()
gc.collect
# event_data_seri = base_event_data_df.event_data.apply(lambda x:json.loads(x))

event_data_seri = base_event_data_df.sample(n=10000).event_data.apply(lambda x:json.loads(x))
# event_data_seri.to_pickle("./kaggle/input/data-science-bowl-2019/event_data_seri.pickle")
event_data_open_df = pd.DataFrame(index=event_data_seri.index)

# -------------------

event_data_open_df["bottle"] = event_data_seri.apply(lambda x:x.get("bottle"))

event_data_open_df["level"] = event_data_seri.apply(lambda x:x.get("level"))

event_data_open_df["total_duration"] = event_data_seri.apply(lambda x:x.get("total_duration"))

event_data_open_df["cloud_size"] = event_data_seri.apply(lambda x:x.get("cloud_size"))

event_data_open_df["chests"] = event_data_seri.apply(lambda x:x.get("chests"))

event_data_open_df["dinosaurs"] = event_data_seri.apply(lambda x:x.get("dinosaurs"))

event_data_open_df["target_containers"] = event_data_seri.apply(lambda x:x.get("target_containers"))

event_data_open_df["media_type"] = event_data_seri.apply(lambda x:x.get("media_type"))

event_data_open_df["starting_weights"] = event_data_seri.apply(lambda x:x.get("starting_weights"))

event_data_open_df["caterpillar"] = event_data_seri.apply(lambda x:x.get("caterpillar"))

event_data_open_df["crystals"] = event_data_seri.apply(lambda x:x.get("crystals"))

event_data_open_df["total_containers"] = event_data_seri.apply(lambda x:x.get("total_containers"))

event_data_open_df["cauldron"] = event_data_seri.apply(lambda x:x.get("cauldron"))

event_data_open_df["shell_size"] = event_data_seri.apply(lambda x:x.get("shell_size"))

event_data_open_df["castles_placed"] = event_data_seri.apply(lambda x:x.get("castles_placed"))

event_data_open_df["dinosaurs_placed"] = event_data_seri.apply(lambda x:x.get("dinosaurs_placed"))

event_data_open_df["target_bucket"] = event_data_seri.apply(lambda x:x.get("target_bucket"))

event_data_open_df["previous_jars"] = event_data_seri.apply(lambda x:x.get("previous_jars"))

event_data_open_df["scale_contents"] = event_data_seri.apply(lambda x:x.get("scale_contents"))

event_data_open_df["has_water"] = event_data_seri.apply(lambda x:x.get("has_water"))

event_data_open_df["correct"] = event_data_seri.apply(lambda x:x.get("correct"))

event_data_open_df["pillars"] = event_data_seri.apply(lambda x:x.get("pillars"))

event_data_open_df["object_type"] = event_data_seri.apply(lambda x:x.get("object_type"))

event_data_open_df["destination"] = event_data_seri.apply(lambda x:x.get("destination"))

event_data_open_df["max_position"] = event_data_seri.apply(lambda x:x.get("max_position"))

event_data_open_df["duration"] = event_data_seri.apply(lambda x:x.get("duration"))

event_data_open_df["water_level"] = event_data_seri.apply(lambda x:x.get("water_level"))

event_data_open_df["holding_shell"] = event_data_seri.apply(lambda x:x.get("holding_shell"))

event_data_open_df["source"] = event_data_seri.apply(lambda x:x.get("source"))

event_data_open_df["bird_height"] = event_data_seri.apply(lambda x:x.get("bird_height"))

event_data_open_df["containers"] = event_data_seri.apply(lambda x:x.get("containers"))

event_data_open_df["sand"] = event_data_seri.apply(lambda x:x.get("sand"))

event_data_open_df["animals"] = event_data_seri.apply(lambda x:x.get("animals"))

event_data_open_df["round_prompt"] = event_data_seri.apply(lambda x:x.get("round_prompt"))

event_data_open_df["gate"] = event_data_seri.apply(lambda x:x.get("gate"))

event_data_open_df["mode"] = event_data_seri.apply(lambda x:x.get("mode"))

event_data_open_df["nest"] = event_data_seri.apply(lambda x:x.get("nest"))

event_data_open_df["tape_length"] = event_data_seri.apply(lambda x:x.get("tape_length"))

event_data_open_df["side"] = event_data_seri.apply(lambda x:x.get("side"))

event_data_open_df["buckets"] = event_data_seri.apply(lambda x:x.get("buckets"))

event_data_open_df["growth"] = event_data_seri.apply(lambda x:x.get("growth"))

event_data_open_df["cloud"] = event_data_seri.apply(lambda x:x.get("cloud"))

event_data_open_df["total_bowls"] = event_data_seri.apply(lambda x:x.get("total_bowls"))

event_data_open_df["options"] = event_data_seri.apply(lambda x:x.get("options"))

event_data_open_df["holes"] = event_data_seri.apply(lambda x:x.get("holes"))

event_data_open_df["filled"] = event_data_seri.apply(lambda x:x.get("filled"))

event_data_open_df["bucket"] = event_data_seri.apply(lambda x:x.get("bucket"))

event_data_open_df["position"] = event_data_seri.apply(lambda x:x.get("position"))

event_data_open_df["time_played"] = event_data_seri.apply(lambda x:x.get("time_played"))

event_data_open_df["bug"] = event_data_seri.apply(lambda x:x.get("bug"))

event_data_open_df["flower"] = event_data_seri.apply(lambda x:x.get("flower"))

event_data_open_df["toy"] = event_data_seri.apply(lambda x:x.get("toy"))

event_data_open_df["diet"] = event_data_seri.apply(lambda x:x.get("diet"))

event_data_open_df["hats_placed"] = event_data_seri.apply(lambda x:x.get("hats_placed"))

event_data_open_df["identifier"] = event_data_seri.apply(lambda x:x.get("identifier"))

event_data_open_df["dinosaur"] = event_data_seri.apply(lambda x:x.get("dinosaur"))

event_data_open_df["left"] = event_data_seri.apply(lambda x:x.get("left"))

event_data_open_df["round"] = event_data_seri.apply(lambda x:x.get("round"))

event_data_open_df["misses"] = event_data_seri.apply(lambda x:x.get("misses"))

event_data_open_df["houses"] = event_data_seri.apply(lambda x:x.get("houses"))

event_data_open_df["container_type"] = event_data_seri.apply(lambda x:x.get("container_type"))

event_data_open_df["target_size"] = event_data_seri.apply(lambda x:x.get("target_size"))

event_data_open_df["caterpillars"] = event_data_seri.apply(lambda x:x.get("caterpillars"))

event_data_open_df["round_number"] = event_data_seri.apply(lambda x:x.get("round_number"))

event_data_open_df["location"] = event_data_seri.apply(lambda x:x.get("location"))

event_data_open_df["scale_weights"] = event_data_seri.apply(lambda x:x.get("scale_weights"))

event_data_open_df["round_target"] = event_data_seri.apply(lambda x:x.get("round_target"))

# event_data_open_df["coordinates"] = event_data_seri.apply(lambda x:x.get("coordinates"))

event_data_open_df["dinosaur_weight"] = event_data_seri.apply(lambda x:x.get("dinosaur_weight"))

event_data_open_df["end_position"] = event_data_seri.apply(lambda x:x.get("end_position"))

event_data_open_df["bowl_id"] = event_data_seri.apply(lambda x:x.get("bowl_id"))

event_data_open_df["jar_filled"] = event_data_seri.apply(lambda x:x.get("jar_filled"))

event_data_open_df["stage_number"] = event_data_seri.apply(lambda x:x.get("stage_number"))

event_data_open_df["buglength"] = event_data_seri.apply(lambda x:x.get("buglength"))

event_data_open_df["prompt"] = event_data_seri.apply(lambda x:x.get("prompt"))

event_data_open_df["stumps"] = event_data_seri.apply(lambda x:x.get("stumps"))

event_data_open_df["target_distances"] = event_data_seri.apply(lambda x:x.get("target_distances"))

event_data_open_df["hat"] = event_data_seri.apply(lambda x:x.get("hat"))

event_data_open_df["dinosaur_count"] = event_data_seri.apply(lambda x:x.get("dinosaur_count"))

event_data_open_df["target_weight"] = event_data_seri.apply(lambda x:x.get("target_weight"))

event_data_open_df["rocket"] = event_data_seri.apply(lambda x:x.get("rocket"))

event_data_open_df["flowers"] = event_data_seri.apply(lambda x:x.get("flowers"))

event_data_open_df["animal"] = event_data_seri.apply(lambda x:x.get("animal"))

# event_data_open_df["jar"] = event_data_seri.apply(lambda x:x.get("jar"))

event_data_open_df["target_water_level"] = event_data_seri.apply(lambda x:x.get("target_water_level"))

event_data_open_df["bowls"] = event_data_seri.apply(lambda x:x.get("bowls"))

event_data_open_df["event_code"] = event_data_seri.apply(lambda x:x.get("event_code"))

event_data_open_df["description"] = event_data_seri.apply(lambda x:x.get("description"))

event_data_open_df["shells"] = event_data_seri.apply(lambda x:x.get("shells"))

event_data_open_df["weight"] = event_data_seri.apply(lambda x:x.get("weight"))

event_data_open_df["bug_length"] = event_data_seri.apply(lambda x:x.get("bug_length"))

event_data_open_df["hole_position"] = event_data_seri.apply(lambda x:x.get("hole_position"))

event_data_open_df["toy_earned"] = event_data_seri.apply(lambda x:x.get("toy_earned"))

event_data_open_df["table_weights"] = event_data_seri.apply(lambda x:x.get("table_weights"))

event_data_open_df["distance"] = event_data_seri.apply(lambda x:x.get("distance"))

event_data_open_df["session_duration"] = event_data_seri.apply(lambda x:x.get("session_duration"))

event_data_open_df["group"] = event_data_seri.apply(lambda x:x.get("group"))

event_data_open_df["bottles"] = event_data_seri.apply(lambda x:x.get("bottles"))

event_data_open_df["launched"] = event_data_seri.apply(lambda x:x.get("launched"))

event_data_open_df["game_time"] = event_data_seri.apply(lambda x:x.get("game_time"))

event_data_open_df["event_count"] = event_data_seri.apply(lambda x:x.get("event_count"))

event_data_open_df["size"] = event_data_seri.apply(lambda x:x.get("size"))

event_data_open_df["dwell_time"] = event_data_seri.apply(lambda x:x.get("dwell_time"))

event_data_open_df["hats"] = event_data_seri.apply(lambda x:x.get("hats"))

event_data_open_df["tutorial_step"] = event_data_seri.apply(lambda x:x.get("tutorial_step"))

event_data_open_df["current_containers"] = event_data_seri.apply(lambda x:x.get("current_containers"))

event_data_open_df["layout"] = event_data_seri.apply(lambda x:x.get("layout"))

event_data_open_df["exit_type"] = event_data_seri.apply(lambda x:x.get("exit_type"))

event_data_open_df["molds"] = event_data_seri.apply(lambda x:x.get("molds"))

event_data_open_df["crystal_id"] = event_data_seri.apply(lambda x:x.get("crystal_id"))

event_data_open_df["house"] = event_data_seri.apply(lambda x:x.get("house"))

event_data_open_df["movie_id"] = event_data_seri.apply(lambda x:x.get("movie_id"))

event_data_open_df["right"] = event_data_seri.apply(lambda x:x.get("right"))

event_data_open_df["has_toy"] = event_data_seri.apply(lambda x:x.get("has_toy"))

event_data_open_df["weights"] = event_data_seri.apply(lambda x:x.get("weights"))

event_data_open_df["buckets_placed"] = event_data_seri.apply(lambda x:x.get("buckets_placed"))

event_data_open_df["version"] = event_data_seri.apply(lambda x:x.get("version"))

event_data_open_df["object"] = event_data_seri.apply(lambda x:x.get("object"))

event_data_open_df["scale_weight"] = event_data_seri.apply(lambda x:x.get("scale_weight"))

event_data_open_df["item_type"] = event_data_seri.apply(lambda x:x.get("item_type"))

event_data_open_df["height"] = event_data_seri.apply(lambda x:x.get("height"))

event_data_open_df["resources"] = event_data_seri.apply(lambda x:x.get("resources"))
# event_data_coordinates_seri = event_data_seri.apply(lambda x:x.get("coordinates")).apply(lambda x:json.loads(x) if x is not None else None)
# event_data_coordinates_seri
# event_data_open_coordinates_df = pd.DataFrame(index=event_data_coordinates_seri.index)

# event_data_open_coordinates_df["coordinates_stage_height"] = event_data_coordinates_seri.apply(lambda x:x.get("stage_height"))

# event_data_open_coordinates_df["coordinates_stage_width"] = event_data_coordinates_seri.apply(lambda x:x.get("stage_width"))

# event_data_open_coordinates_df["coordinates_x"] = event_data_coordinates_seri.apply(lambda x:x.get("x"))

# event_data_open_coordinates_df["coordinates_y"] = event_data_coordinates_seri.apply(lambda x:x.get("y"))
# event_data_jar_seri = event_data_seri.apply(lambda x:x.get("jar")).apply(lambda x:json.loads(x))