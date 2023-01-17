import pandas as pd

import numpy as np



import json



from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from sklearn.metrics import confusion_matrix

# from sklearn.feature_selection import RFE

from sklearn.model_selection import train_test_split

from sklearn.utils import class_weight

# from sklearn.svm import SVC

# from catboost import CatBoostClassifier

# from sklearn.linear_model import LogisticRegression

# from sklearn.ensemble import RandomForestClassifier

# from sklearn.ensemble import AdaBoostClassifier



import keras

import tensorflow as tf



# import pprint

import gc

import os

import tqdm

import matplotlib.pyplot as plt

import seaborn as sns



# pandas display option

pd.set_option('display.max_columns', 500)

pd.set_option('display.max_row', 1500)

pd.set_option('max_colwidth', 150)

pd.set_option('display.float_format', '{:.2f}'.format)



# data load option

dtypes = {"event_id":"object", "game_session":"object", "timestamp":"object", 

          "event_data":"object", "installation_id":"object", "event_count":"int16", 

          "event_code":"int16", "game_time":"int32", "title":"category", 

          "type":"category", "world":"category"}

label = {"game_session":"object", "installation_id":"object", "title":"category", 

         "num_correct":"int8", "num_incorrect":"int8", 

         "accuracy":"float16", "accuracy_group":"int8"}



# hyper parameter

loss_type = "category" # mse/category

dp_log = True

# window = 70

batch_sizes = 1

validation = True

scale_type = "minmax" # minmax/robust/standard
train = pd.read_csv("/kaggle/input/data-science-bowl-2019/train.csv", dtype=dtypes)

test = pd.read_csv("/kaggle/input/data-science-bowl-2019/test.csv", dtype=dtypes)

label_ = pd.read_csv("/kaggle/input/data-science-bowl-2019/train_labels.csv", dtype=label)

# sample = pd.read_csv("/kaggle/input/data-science-bowl-2019/sample_submission.csv")

# specs = pd.read_csv("/kaggle/input/data-science-bowl-2019/specs.csv")
# calculating accuracy

class accuracy:

    def __init__(self, df):

        self.df = df



        

    # Assessment evaluation-Cart Balancer (Assessment)

    def cart_assessment(self):

        _ = self.df.query("title=='Cart Balancer (Assessment)' and event_id=='d122731b'")

        _ = _.loc[:, ["game_session", "installation_id", "event_data"]]

        _["correct"] = _.event_data.apply(lambda x:(json.loads(x)["correct"] if "correct" in json.loads(x).keys() else -999))

        _["num_correct_"] = 0

        _["num_incorrect_"] = 0

        _.loc[_.correct==True, "num_correct_"] = 1

        _.loc[_.correct==False, "num_incorrect_"] = 1

        _ = _.groupby(["installation_id", "game_session"]).sum().reset_index()

        _["accuracy_"] = _["num_correct_"]/(_["num_correct_"]+_["num_incorrect_"])

        _["accuracy_group"] = _["num_incorrect_"].apply(lambda x : 3 if x==0 else (2 if x==1 else 1))*_["num_correct_"]



#         return _.loc[:, ["installation_id", "game_session", "num_correct_", "num_incorrect_", "accuracy_", "accuracy_group"]]

        return _.loc[:, ["installation_id", "game_session", "accuracy_group"]]



    def cart_assessment_2(self):

        _ = self.df.query("title=='Cart Balancer (Assessment)' and event_id=='b74258a0'")

        _ = _.loc[:, ["game_session", "installation_id", "event_data"]]

        _["misses"] = _.event_data.apply(lambda x:(json.loads(x)["misses"] if "misses" in json.loads(x).keys() else -999))

        _["num_correct_"]=1

        _ = _.groupby("game_session").sum().reset_index()

        _["accuracy_"] = _["num_correct_"]/(_["num_correct_"]+_["misses"])



        return _.loc[:, ["game_session", "num_correct_", "misses", "accuracy_"]]

    

    

    # Assessment evaluation-Chest Sorter (Assessment)

    def chest_assessment(self):

        _ = self.df.query("title=='Chest Sorter (Assessment)' and event_id=='93b353f2'")

        _ = _.loc[:, ["game_session", "installation_id", "event_data"]]

        _["correct"] = _.event_data.apply(lambda x:(json.loads(x)["correct"] if "correct" in json.loads(x).keys() else -999))

        _["num_correct_"] = 0

        _["num_incorrect_"] = 0

        _.loc[_.correct==True, "num_correct_"] = 1

        _.loc[_.correct==False, "num_incorrect_"] = 1

        _ = _.groupby(["installation_id", "game_session"]).sum().reset_index()

        _["accuracy_"] = _["num_correct_"]/(_["num_correct_"]+_["num_incorrect_"])

        _["accuracy_group"] = _["num_incorrect_"].apply(lambda x : 3 if x==0 else (2 if x==1 else 1))*_["num_correct_"]



#         return _.loc[:, ["installation_id", "game_session", "num_correct_", "num_incorrect_", "accuracy_", "accuracy_group"]]

        return _.loc[:, ["installation_id", "game_session", "accuracy_group"]]

    

    def chest_assessment_2(self):

        _ = self.df.query("title=='Chest Sorter (Assessment)' and event_id=='38074c54'")

        _ = _.loc[:, ["game_session", "installation_id", "event_data"]]

        _["misses"] = _.event_data.apply(lambda x:(json.loads(x)["misses"] if "misses" in json.loads(x).keys() else -999))

        _["num_correct_"]=1

        _ = _.groupby("game_session").sum().reset_index()

        _["accuracy_"] = _["num_correct_"]/(_["num_correct_"]+_["misses"])



        return _.loc[:, ["game_session", "num_correct_", "misses", "accuracy_"]]

    

    

    # Assessment evaluation-Cauldron Filler (Assessment)

    def cauldron_assessment(self):

        _ = self.df.query("title=='Cauldron Filler (Assessment)' and event_id=='392e14df'")

        _ = _.loc[:, ["game_session", "installation_id", "event_data"]]

        _["correct"] = _.event_data.apply(lambda x:(json.loads(x)["correct"] if "correct" in json.loads(x).keys() else -999))

        _["num_correct_"] = 0

        _["num_incorrect_"] = 0

        _.loc[_.correct==True, "num_correct_"] = 1

        _.loc[_.correct==False, "num_incorrect_"] = 1

        _ = _.groupby(["installation_id", "game_session"]).sum().reset_index()

        _["accuracy_"] = _["num_correct_"]/(_["num_correct_"]+_["num_incorrect_"])

        _["accuracy_group"] = _["num_incorrect_"].apply(lambda x : 3 if x==0 else (2 if x==1 else 1))*_["num_correct_"]



#         return _.loc[:, ["installation_id", "game_session", "num_correct_", "num_incorrect_", "accuracy_", "accuracy_group"]]

        return _.loc[:, ["installation_id", "game_session", "accuracy_group"]]



    def cauldron_assessment_2(self):

        _ = self.df.query("title=='Cauldron Filler (Assessment)' and event_id=='28520915'")

        _ = _.loc[:, ["game_session", "installation_id", "event_data"]]

        _["misses"] = _.event_data.apply(lambda x:(json.loads(x)["misses"] if "misses" in json.loads(x).keys() else -999))

        _["num_correct_"] = 1

        _ = _.groupby("game_session").sum().reset_index()

        _["accuracy_"] = _["num_correct_"]/(_["num_correct_"]+_["misses"])



        return _.loc[:, ["game_session", "num_correct_", "misses", "accuracy_"]]

    

    

    # Assessment evaluation-Mushroom Sorter (Assessment)

    def mushroom_assessment(self):

        _ = self.df.query("title=='Mushroom Sorter (Assessment)' and event_id=='25fa8af4'")

        _ = _.loc[:, ["game_session", "installation_id", "event_data"]]

        _["correct"] = _.event_data.apply(lambda x:(json.loads(x)["correct"] if "correct" in json.loads(x).keys() else -999))

        _["num_correct_"] = 0

        _["num_incorrect_"] = 0

        _.loc[_.correct==True, "num_correct_"] = 1

        _.loc[_.correct==False, "num_incorrect_"] = 1

        _ = _.groupby(["installation_id", "game_session"]).sum().reset_index()

        _["accuracy_"] = _["num_correct_"]/(_["num_correct_"]+_["num_incorrect_"])

        _["accuracy_group"] = _["num_incorrect_"].apply(lambda x : 3 if x==0 else (2 if x==1 else 1))*_["num_correct_"]



#         return _.loc[:, ["installation_id", "game_session", "num_correct_", "num_incorrect_", "accuracy_", "accuracy_group"]]

        return _.loc[:, ["installation_id", "game_session", "accuracy_group"]]

    

    def mushroom_assessment_2(self):

        _ = self.df.query("title=='Mushroom Sorter (Assessment)' and event_id=='6c930e6e'")

        _ = _.loc[:, ["game_session", "installation_id", "event_data"]]

        _["misses"] = _.event_data.apply(lambda x:(json.loads(x)["misses"] if "misses" in json.loads(x).keys() else -999))

        _["num_correct_"] = 1

        _ = _.groupby("game_session").sum().reset_index()

        _["accuracy_"] = _["num_correct_"]/(_["num_correct_"]+_["misses"])



        return _.loc[:, ["game_session", "num_correct_", "misses", "accuracy_"]]

    

    

    # Assessment evaluation-Bird Measurer (Assessment)

    def bird_assessment(self):

        _ = self.df.query("title=='Bird Measurer (Assessment)' and event_id=='17113b36'")

        _ = _.loc[:, ["game_session", "installation_id", "event_data"]]

        _["correct"] = _.event_data.apply(lambda x:(json.loads(x)["correct"] if "correct" in json.loads(x).keys() else -999))

        _["num_correct_"] = 0

        _["num_incorrect_"] = 0

        _.loc[_.correct==True, "num_correct_"] = 1

        _.loc[_.correct==False, "num_incorrect_"] = 1

        _ = _.groupby(["installation_id", "game_session"]).sum().reset_index()

        _["accuracy_"] = _["num_correct_"]/(_["num_correct_"]+_["num_incorrect_"])

        _["accuracy_group"] = _["num_incorrect_"].apply(lambda x : 3 if x==0 else (2 if x==1 else 1))*_["num_correct_"]



#         return _.loc[:, ["installation_id", "game_session", "num_correct_", "num_incorrect_", "accuracy_", "accuracy_group"]]

        return _.loc[:, ["installation_id", "game_session", "accuracy_group"]]

    

    def bird_assessment_2(self):

        _ = self.df.query("title=='Bird Measurer (Assessment)' and event_id=='f6947f54'")

        _ = _.loc[:, ["game_session", "installation_id", "event_data"]]

        _["misses"] = _.event_data.apply(lambda x:(json.loads(x)["misses"] if "misses" in json.loads(x).keys() else -999))

        _["num_correct_"] = 1

        _ = _.groupby("game_session").sum().reset_index()

        _["accuracy_"] = _["num_correct_"]/(_["num_correct_"]+_["misses"])



        return _.loc[:, ["game_session", "num_correct_", "misses", "accuracy_"]]



# quadratic kappa

def quadratic_kappa(actuals, preds, N=4):

    w = np.zeros((N,N))

    O = confusion_matrix(actuals, preds)

    for i in range(len(w)): 

        for j in range(len(w)):

            w[i][j] = float(((i-j)**2)/(N-1)**2)

    

    act_hist=np.zeros([N])

    for item in actuals: 

        act_hist[item]+=1

    

    pred_hist=np.zeros([N])

    for item in preds: 

        pred_hist[item]+=1

                         

    E = np.outer(act_hist, pred_hist);

    E = E/E.sum();

    O = O/O.sum();

    

    num=0

    den=0

    for i in range(len(w)):

        for j in range(len(w)):

            num+=w[i][j]*O[i][j]

            den+=w[i][j]*E[i][j]

    return (1 - (num/den))
test["timestamp"] = pd.to_datetime(test.timestamp)

test.sort_values(["timestamp", "event_count"], ascending=True, inplace=True)



_ = accuracy(test).cart_assessment()

_ = _.append(accuracy(test).chest_assessment(), ignore_index=True)

_ = _.append(accuracy(test).cauldron_assessment(), ignore_index=True)

_ = _.append(accuracy(test).mushroom_assessment(), ignore_index=True)

_ = _.append(accuracy(test).bird_assessment(), ignore_index=True)



test = test[test.installation_id.isin(pd.unique(_.installation_id))]

test = test.merge(_, how="left", on=["installation_id", "game_session"])
df_test = []

idx = 0

for _, val in tqdm.tqdm_notebook(test.groupby("installation_id", sort=False)):

# for _, val in tqdm.notebook.tqdm(test.groupby("installation_id", sort=False)):

    val.reset_index(drop=True, inplace=True)

    _ = val.query("type=='Assessment'")

    _ = _[~_.accuracy_group.isnull()]

    session = _.reset_index().groupby("game_session", sort=False).index.first().values

    for j in session:

        sample = val[:j+1]

        sample["ID"] = idx

        idx += 1

        df_test.append(sample)
label = pd.DataFrame(columns=["ID", "accuracy_group"])

for i in tqdm.tqdm_notebook(df_test):

# for i in tqdm.notebook.tqdm(df_test):

    label = pd.concat([label, i.iloc[-1:, -2:]], sort=False)



label.reset_index(drop=True, inplace=True)

label.accuracy_group = label.accuracy_group.astype("int8")
df = train[train.installation_id.isin(pd.unique(label_.installation_id))]

del train

df = df.merge(label_.loc[:, ["installation_id", "game_session", "title", "accuracy_group"]], 

              on=["installation_id", "game_session", "title"], how="left")

df["timestamp"] = pd.to_datetime(df.timestamp)

df.sort_values(["timestamp", "event_count"], ascending=True, inplace=True)

df.reset_index(drop=True, inplace=True)
df_train = []

idx = max(label.ID)+1

for _, val in tqdm.tqdm_notebook(df.groupby("installation_id", sort=False)):

# for _, val in tqdm.notebook.tqdm(df.groupby("installation_id", sort=False)):

    val.reset_index(drop=True, inplace=True)

    session = val.query("type=='Assessment'").reset_index().groupby("game_session", sort=False).index.first().values

    for j in session:

        if ~np.isnan(val.iat[j, -1]):

            sample = val[:j+1]

            sample["ID"] = idx

            idx += 1

            df_train.append(sample)
for i in tqdm.tqdm_notebook(df_train):

# for i in tqdm.notebook.tqdm(df_train):

    label = pd.concat([label, i.iloc[-1:, -2:]], sort=False)



label.reset_index(drop=True, inplace=True)

label.accuracy_group = label.accuracy_group.astype("int8")

label = label.merge(pd.get_dummies(label.accuracy_group, prefix="y"), left_on=["ID"], right_index=True)



df_test.extend(df_train)

df_train = df_test

del df_test
display(df_train[0].head()), display(label.head())
col = {}
ID = []

world = []

size = []



for i in tqdm.tqdm_notebook(df_train):

    # world_log

    _ = i.groupby(["ID", "world"]).size().reset_index()

    ID.extend(_.ID)

    world.extend(_.world)

    size.extend(_[0])



world_log = pd.DataFrame(data={"ID":ID, "world":world, "size":size})

world_log = world_log.pivot_table(index="ID", columns="world", values="size")

world_log = world_log.fillna(0)

world_log.columns.name = None

world_log.reset_index(inplace=True)

world_log = world_log.loc[:, ["ID", "CRYSTALCAVES", "MAGMAPEAK", "TREETOPCITY", "NONE"]]
plt.figure(figsize=(30, 10))

for idx, val in enumerate(["CRYSTALCAVES", "MAGMAPEAK", "TREETOPCITY", "NONE"]):

    plt.subplot(2, 4, idx+1)

    for i in [0, 1, 2, 3]:

        sns.distplot(world_log.merge(label).query("accuracy_group==@i")[val], label=i)

    plt.legend()

    

    plt.subplot(2, 4, idx+5)

    for i in [0, 1, 2, 3]:

        sns.distplot(np.log(world_log.merge(label).query("accuracy_group==@i")[val]+1), label=i)

    plt.legend()

plt.show()
world_log = world_log.add_suffix("_l")

world_log.rename(columns={"ID_l":"ID"}, inplace=True)

if dp_log==True:

    world_log.iloc[:, 1:] = np.log(world_log.iloc[:, 1:]+1)

gc.collect()
world_log.head()
ID = []

world = []

game_time = []



for i in tqdm.tqdm_notebook(df_train):

    # world_time

    _ = i.groupby(["ID", "world", "game_session"]).game_time.max().reset_index()

    ID.extend(_.ID)

    world.extend(_.world)

    game_time.extend(_.game_time)



world_time = pd.DataFrame(data={"ID":ID, "world":world, "game_time":game_time})

world_time = world_time.groupby(["ID", "world"]).sum().reset_index()

world_time = world_time.pivot_table(index="ID", columns="world", values="game_time")

world_time = world_time.fillna(-1)

world_time.columns.name = None

world_time["ID"] = world_time.index

world_time.reset_index(drop=True, inplace=True)

world_time = world_time.loc[:, ["ID", "CRYSTALCAVES", "MAGMAPEAK", "TREETOPCITY", "NONE"]]
plt.figure(figsize=(30, 10))

for idx, val in enumerate(["CRYSTALCAVES", "MAGMAPEAK", "TREETOPCITY", "NONE"]):

    plt.subplot(2, 4, idx+1)

    for i in [0, 1, 2, 3]:

        sns.distplot(world_time.merge(label).query("accuracy_group==@i")[val], label=i)

    plt.legend()

    

    plt.subplot(2, 4, idx+5)

    for i in [0, 1, 2, 3]:

        sns.distplot(np.log(world_time.merge(label).query("accuracy_group==@i")[val]+2), label=i)

    plt.legend()

plt.show()
plt.figure(figsize=(30, 10))

for idx, val in enumerate(["CRYSTALCAVES", "MAGMAPEAK", "TREETOPCITY", "NONE"]):

    plt.subplot(2, 4, idx+1)

    sns.distplot(world_time[val])

#     plt.title(val)

    plt.subplot(2, 4, idx+5)

    sns.distplot(np.log(world_time[val]+2))

#     plt.title(val)

plt.show()
world_time.drop(columns=["NONE"], inplace=True)

world_time = world_time.add_suffix("_t")

world_time.rename(columns={"ID_t":"ID"}, inplace=True)

if dp_log==True:

    world_time.iloc[:, 1:] = np.log(world_time.iloc[:, 1:]+2)

gc.collect()
world_time.head()
ID = []

world = []

game_session = []



for i in tqdm.tqdm_notebook(df_train):

    # world_session

    _ = i.groupby(["ID", "world"]).game_session.nunique().reset_index()

    ID.extend(_.ID)

    world.extend(_.world)

    game_session.extend(_.game_session)



world_session = pd.DataFrame(data={"ID":ID, "world":world, "game_session":game_session})

world_session = world_session.pivot_table(index="ID", columns="world", values="game_session")

world_session = world_session.fillna(0)

world_session.columns.name = None

world_session["ID"] = world_session.index

world_session.reset_index(drop=True, inplace=True)

world_session = world_session.loc[:, ["ID", "CRYSTALCAVES", "MAGMAPEAK", "TREETOPCITY", "NONE"]]
plt.figure(figsize=(30, 10))

for idx, val in enumerate(["CRYSTALCAVES", "MAGMAPEAK", "TREETOPCITY", "NONE"]):

    plt.subplot(2, 4, idx+1)

    for i in [0, 1, 2, 3]:

        sns.distplot(world_session.merge(label).query("accuracy_group==@i")[val], label=i)

    plt.legend()

    

    plt.subplot(2, 4, idx+5)

    for i in [0, 1, 2, 3]:

        sns.distplot(np.log(world_session.merge(label).query("accuracy_group==@i")[val]+1), label=i)

    plt.legend()

plt.show()
plt.figure(figsize=(30, 10))

for idx, val in enumerate(["CRYSTALCAVES", "MAGMAPEAK", "TREETOPCITY", "NONE"]):

    plt.subplot(2, 4, idx+1)

    sns.distplot(world_session[val])

#     plt.title(val)

    plt.subplot(2, 4, idx+5)

    sns.distplot(np.log(world_session[val]+1))

#     plt.title(val)

plt.show()
world_session = world_session.add_suffix("_s")

world_session.rename(columns={"ID_s":"ID"}, inplace=True)

if dp_log==True:

    world_session.iloc[:, 1:] = np.log(world_session.iloc[:, 1:]+1)

gc.collect()
world_session.head()
ID = []

event_id = []

size = []



for i in tqdm.tqdm_notebook(df_train):

    # event_id

    _ = i.groupby(["ID", "event_id"]).size().reset_index()

    ID.extend(_.ID)

    event_id.extend(_.event_id)

    size.extend(_[0])



event_id = pd.DataFrame(data={"ID":ID, "event_id":event_id, "size":size})

event_id = event_id.pivot_table(index="ID", columns="event_id", values="size")

event_id = event_id.fillna(0)

event_id.columns.name = None

event_id.index.name = None

event_id["ID"] = event_id.index

event_id.reset_index(drop=True, inplace=True)
if dp_log==True:

    event_id.iloc[:, :-1] = np.log(event_id.iloc[:, :-1]+1)

#     event_id.iloc[:, 1:] = np.log(event_id.iloc[:, 1:]+1)

gc.collect()
event_id.head()
None
ID = []

game_time = []



for i in tqdm.tqdm_notebook(df_train):

    # play_time

    _ = i.groupby(["ID", "game_session"]).game_time.max().reset_index()

    ID.extend(_.ID)

    game_time.extend(_.game_time)



play_time = pd.DataFrame(data={"ID":ID, "game_time":game_time})

play_time = play_time.groupby(["ID"]).sum().reset_index()

play_time.reset_index(drop=True, inplace=True)

play_time = play_time.fillna(0)
plt.figure(figsize=(20, 5))

plt.subplot(1, 2, 1)

for i in [0, 1, 2, 3]:

    sns.distplot(play_time.merge(label).query("accuracy_group==@i")["game_time"], label=i)

plt.legend()



plt.subplot(1, 2, 2)

for i in [0, 1, 2, 3]:

    sns.distplot(np.log(play_time.merge(label).query("accuracy_group==@i")["game_time"]+1), label=i)

plt.legend()

plt.show()
if dp_log==True:

    play_time.iloc[:, 1:] = np.log(play_time.iloc[:, 1:]+1)

gc.collect()
play_time.head()
gap_time = pd.DataFrame()

for i in tqdm.tqdm_notebook(df_train):

    # gap_time

    _ = i.groupby(["ID"]).timestamp.agg(["min", "max"])

    _.columns.name = None

    gap_time = pd.concat([gap_time, _], sort=True)



gap_time.reset_index(inplace=True)

gap_time["gap"] = gap_time["max"]-gap_time["min"]

gap_time["gap"] = gap_time["gap"].astype("int")
plt.figure(figsize=(20, 5))

plt.subplot(1, 2, 1)

for i in [0, 1, 2, 3]:

    sns.distplot(gap_time.merge(label).query("accuracy_group==@i")["gap"], label=i)

plt.legend()



plt.subplot(1, 2, 2)

for i in [0, 1, 2, 3]:

    sns.distplot(np.log(gap_time.merge(label).query("accuracy_group==@i")["gap"]+1), label=i)

plt.legend()

plt.show()
gap_time.drop(columns=["max", "min"], inplace=True)

if dp_log==True:

    gap_time.iloc[:, 1:] = np.log(gap_time.iloc[:, 1:]+1)

gc.collect()
gap_time.head()
session_count = pd.DataFrame()

for i in tqdm.tqdm_notebook(df_train):

    # session_count

    _ = i.groupby(["ID"]).game_session.nunique().reset_index()

    _.columns.name = None

    session_count = pd.concat([session_count, _], sort=True)



session_count.reset_index(drop=True, inplace=True)
plt.figure(figsize=(20, 5))

plt.subplot(1, 2, 1)

for i in [0, 1, 2, 3]:

    sns.distplot(session_count.merge(label).query("accuracy_group==@i")["game_session"], bins=50, label=i)

plt.legend()



plt.subplot(1, 2, 2)

for i in [0, 1, 2, 3]:

    sns.distplot(np.log(session_count.merge(label).query("accuracy_group==@i")["game_session"]), bins=50, label=i)

plt.legend()

plt.show()
if dp_log==True:

    session_count.iloc[:, 1:] = np.log(session_count.iloc[:, 1:])

gc.collect()
session_count.head()
session_length = pd.DataFrame()

for i in tqdm.tqdm_notebook(df_train):

    # session_length

#     _ = i.query("type!='Clip'").groupby(["ID", "game_session"]).size().groupby(["ID"]).mean().reset_index().rename(columns={0:"session_length"})

    _ = i.groupby(["ID", "game_session"]).size().groupby(["ID"]).mean().reset_index().rename(columns={0:"session_length"})

    _.columns.name = None

    session_length = pd.concat([session_length, _], sort=True)



session_length.reset_index(drop=True, inplace=True)
plt.figure(figsize=(20, 5))

plt.subplot(1, 2, 1)

for i in [0, 1, 2, 3]:

    sns.distplot(session_length.merge(label).query("accuracy_group==@i")["session_length"], bins=50, label=i)

plt.legend()



plt.subplot(1, 2, 2)

for i in [0, 1, 2, 3]:

    sns.distplot(np.log(session_length.merge(label).query("accuracy_group==@i")["session_length"]), bins=50, label=i)

plt.legend()

plt.show()
if dp_log==True:

    session_length.iloc[:, 1:] = np.log(session_length.iloc[:, 1:])

gc.collect()
session_length.head()
session_time = pd.DataFrame()

for i in tqdm.tqdm_notebook(df_train):

    # session_time

    _ = i.groupby(["ID", "game_session"]).game_time.max().groupby(["ID"]).mean().reset_index()

    _.columns.name = None

    session_time = pd.concat([session_time, _], sort=True)



session_time.reset_index(drop=True, inplace=True)
plt.figure(figsize=(20, 5))

plt.subplot(1, 2, 1)

for i in [0, 1, 2, 3]:

    sns.distplot(session_time.merge(label).query("accuracy_group==@i")["game_time"], bins=50, label=i)

plt.legend()



plt.subplot(1, 2, 2)

for i in [0, 1, 2, 3]:

    sns.distplot(np.log(session_time.merge(label).query("accuracy_group==@i")["game_time"]+1), bins=50, label=i)

plt.legend()

plt.show()
if dp_log==True:

    session_time.iloc[:, 1:] = np.log(session_time.iloc[:, 1:]+1)

gc.collect()
session_time.head()
ID = []

types = []

size = []



for i in tqdm.tqdm_notebook(df_train):

    # types

    _ = i.groupby(["ID", "type"]).size().reset_index()

    ID.extend(_.ID)

    types.extend(_.type)

    size.extend(_[0])



types = pd.DataFrame(data={"ID":ID, "type":types, "size":size})

types = types.pivot_table(index="ID", columns="type", values="size")

types.columns.name = None

types.index.name = None

types = types.fillna(0)

types["ID"] = types.index

types = types.loc[:, ["ID", "Activity", "Assessment", "Clip", "Game"]]
plt.figure(figsize=(30, 10))

for idx, val in enumerate(["Activity", "Assessment", "Clip", "Game"]):

    plt.subplot(2, 4, idx+1)

    for i in [0, 1, 2, 3]:

        sns.distplot(types.merge(label).query("accuracy_group==@i")[val], label=i)

    plt.legend()

    

    plt.subplot(2, 4, idx+5)

    for i in [0, 1, 2, 3]:

        sns.distplot(np.log(types.merge(label).query("accuracy_group==@i")[val]+1), label=i)

    plt.legend()

plt.show()
if dp_log==True:

    types.iloc[:, 1:] = np.log(types.iloc[:, 1:]+1)

gc.collect()
types.head()
ID = []

title = []

size = []



for i in tqdm.tqdm_notebook(df_train):

    # title

    _ = i.groupby(["ID", "title"]).size().reset_index()

    ID.extend(_.ID)

    title.extend(_.title)

    size.extend(_[0])



title = pd.DataFrame(data={"ID":ID, "title":title, "size":size})

title = title.pivot_table(index="ID", columns="title", values="size")

title.columns.name = None

title.index.name = None

title = title.fillna(0)

title["ID"] = title.index
if dp_log==True:

    title.iloc[:, :-1] = np.log(title.iloc[:, :-1]+1)

gc.collect()
title.head()
assessment = pd.DataFrame(columns=["ID", "title"])

for i in tqdm.tqdm_notebook(df_train):

    # assessment

    _ = i.tail(1).loc[:, ["ID", "title"]].reset_index(drop=True)

    assessment = pd.concat([assessment, _], sort=False)



assessment['Assessment_1'] = 0

assessment['Assessment_2'] = 0

assessment['Assessment_3'] = 0

assessment['Assessment_4'] = 0

assessment['Assessment_5'] = 0



assessment.loc[assessment.title=='Mushroom Sorter (Assessment)', 'Assessment_1'] = 1

assessment.loc[assessment.title=='Cauldron Filler (Assessment)', 'Assessment_2'] = 1

assessment.loc[assessment.title=='Chest Sorter (Assessment)', 'Assessment_3'] = 1

assessment.loc[assessment.title=='Cart Balancer (Assessment)', 'Assessment_4'] = 1

assessment.loc[assessment.title=='Bird Measurer (Assessment)', 'Assessment_5'] = 1
_ = assessment.merge(label).groupby(["title", "accuracy_group"]).size().reset_index()

_.accuracy_group = _.accuracy_group.astype("object")

plt.figure(figsize=(20, 10))

sns.barplot(x="title", y=0, hue="accuracy_group", data=_, dodge=True, alpha=.7)

plt.show()
plt.figure(figsize=(20, 10))

plt.bar("title", height="count", 

        data=assessment.groupby("title").size().reset_index().rename(columns={0:"count"}))

plt.show()
del assessment["title"]

# assessment = assessment.loc[:, ["ID", "Assessment_1", "Assessment_2", "Assessment_3", "Assessment_4", "Assessment_5"]]
assessment.head()
time = pd.DataFrame(columns=["ID", "timestamp"])

for i in tqdm.tqdm_notebook(df_train):

    # time

    _ = i.tail(1).loc[:, ["ID", "timestamp"]]

    time = pd.concat([time, _], sort=False)



time.reset_index(drop=True, inplace=True)

time["hour"] = time.timestamp.dt.hour

time["hour"] = time.hour.astype("object")

time = time.merge(pd.get_dummies(time.hour, prefix="hour"), how="left", 

                  left_index=True, right_index=True)

time.drop(columns=["timestamp", "hour"], inplace=True)
time.head()
ID = []

game_title = []

game_round = []



for i in tqdm.tqdm_notebook(df_train):

    if "Game" in i.type.unique():

        _ = i.query("type=='Game'").loc[:, ["ID", "title", "event_data"]].set_index(["ID", "title"]).event_data.apply(lambda x: json.loads(x)["round"]).groupby(["ID", "title"]).max().reset_index()

        ID.extend(list(_.ID))

        game_title.extend(_.title)

        game_round.extend(_.event_data)

        

game = pd.DataFrame(data={"ID":ID, "game_title":game_title, "round":game_round})

game = game.pivot_table(index="ID", columns="game_title", values="round")

game.reset_index(inplace=True)

game.columns.name = None

game = game.fillna(-1)



ID = pd.DataFrame(data={"ID":range(0, len(df_train))})

game = ID.merge(game, how="left")

game = game.fillna(-1)



game = game.add_suffix("_r")

game.rename(columns={"ID_r":"ID"}, inplace=True)
game.head()
# data_set = [world_log, world_time, world_session, event_id, play_time, gap_time, session_count, session_length, session_time, types, title, assessment, time, game]

# _ = pd.concat(data_set, axis=1, keys=["ID"])
_ = world_log.merge(world_time, how="left", on=["ID"])

_ = _.merge(world_session, how="left", on=["ID"])

_ = _.merge(event_id, how="left", on=["ID"])

_ = _.merge(play_time, how="left", on=["ID"])

_ = _.merge(gap_time, how="left", on=["ID"])

_ = _.merge(session_count, how="left", on=["ID"])

_ = _.merge(session_length, how="left", on=["ID"])

_ = _.merge(session_time, how="left", on=["ID"])

_ = _.merge(types, how="left", on=["ID"])

_ = _.merge(title, how="left", on=["ID"])

_ = _.merge(assessment, how="left", on=["ID"])

_ = _.merge(time, how="left", on=["ID"])

_ = _.merge(game, how="left", on=["ID"])
train_x_col = list(_.columns)

train_y_col = ["accuracy_group", "y_0", "y_1", "y_2", "y_3"]
_.to_csv("train.csv", index=False)

label.to_csv("label.csv", index=False)
if loss_type=="mse":

    if scale_type=="minmax":

        scaler = MinMaxScaler()

    elif scale_type=="robust":

        scaler = RobustScaler()

    elif scale_type=="standard":

        scaler = StandardScaler()

    scaler_y = MinMaxScaler()

    train_x = scaler.fit_transform(_.loc[:, train_x_col[1:]])

#     train_y = scaler_y.fit_transform([_.loc[:, "accuracy_group"]])

    train_y = label.loc[:, train_y_col]

    print(train_x[0])

    print(train_y.iloc[0, :])

elif loss_type=="category":

    if scale_type=="minmax":

        scaler = MinMaxScaler()

    elif scale_type=="robust":

        scaler = RobustScaler()

    elif scale_type=="standard":

        scaler = StandardScaler()

    train_x = scaler.fit_transform(_.loc[:, train_x_col[1:]])

    train_y = label.loc[:, train_y_col]

    print(train_x[0])

    print(train_y.iloc[0, :])
class_weights = class_weight.compute_class_weight('balanced', np.unique(label.accuracy_group),

                                                  label.accuracy_group)

np.unique(label.accuracy_group), class_weights
train_y["class_weight"] = 0

train_y.loc[train_y.accuracy_group==0, "class_weight"] = class_weights[0]

train_y.loc[train_y.accuracy_group==1, "class_weight"] = class_weights[1]

train_y.loc[train_y.accuracy_group==2, "class_weight"] = class_weights[2]

train_y.loc[train_y.accuracy_group==3, "class_weight"] = class_weights[3]
if validation:

    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, random_state=1228)

    display(train_x.shape, train_y.shape)    
for i in range(len(train_x[0])):

    print(i, min(train_x[:, i]), max(train_x[:, i]))
# leakyrelu = keras.layers.LeakyReLU(alpha=0.3)

leakyrelu = tf.nn.leaky_relu
model = keras.models.Sequential()



model.add(keras.layers.Dense(128, activation=leakyrelu, kernel_initializer="he_normal"))

model.add(keras.layers.Dropout(.3))



model.add(keras.layers.Dense(256, activation=leakyrelu, kernel_initializer="he_normal"))

model.add(keras.layers.Dropout(.3))



model.add(keras.layers.Dense(256, activation=leakyrelu, kernel_initializer="he_normal"))

model.add(keras.layers.Dropout(.3))



model.add(keras.layers.Dense(128, activation=leakyrelu, kernel_initializer="he_normal"))

model.add(keras.layers.Dropout(.3))



model.add(keras.layers.Dense(64, activation=leakyrelu, kernel_initializer="he_normal"))

model.add(keras.layers.Dropout(.3))



model.add(keras.layers.Dense(32, activation=leakyrelu, kernel_initializer="he_normal"))

model.add(keras.layers.Dropout(.3))



model.add(keras.layers.Dense(16, activation=leakyrelu, kernel_initializer="he_normal"))

model.add(keras.layers.Dropout(.3))



if loss_type=="mse":

    model.add(keras.layers.Dense(1, activation="linear"))

    model.compile(loss="mse", optimizer="Adam")

elif loss_type=="category":

    model.add(keras.layers.Dense(4, activation="softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=['categorical_accuracy'])
# keras.backend.reset_uids()
if not os.path.exists("model"):

    os.mkdir("model")
if validation:

    if loss_type=="mse":

        model.fit(x=train_x, y=train_y.loc[:, ["accuracy_group"]], 

                  validation_data=[val_x, val_y.loc[:, ["accuracy_group"]]], 

                  epochs=50, batch_size=batch_sizes, shuffle=True, class_weight=class_weight)

    elif loss_type=="category":

        model.fit(x=train_x, y=train_y.loc[:, ["y_0", "y_1", "y_2", "y_3"]].values, 

                  validation_data=[val_x, val_y.loc[:, ["y_0", "y_1", "y_2", "y_3"]].values], 

                  epochs=1000, batch_size=batch_sizes, shuffle=True, 

                  sample_weight=train_y.loc[:, ["class_weight"]].values.flatten(), 

                  callbacks=[keras.callbacks.EarlyStopping(monitor="val_categorical_accuracy", 

                                                           patience=100, mode="auto"), 

                             keras.callbacks.ModelCheckpoint("model/weights.{epoch:02d}-{val_categorical_accuracy:.3f}.hdf5", 

                                                             monitor='val_categorical_accuracy', 

                                                             verbose=0, save_best_only=True, save_weights_only=False, 

                                                             mode="auto", period=1)])
if validation==False:

    if loss_type=="mse":

        model.fit(train_x, train_y.values, epochs=150, batch_size=batch_sizes, verbose=1, validation_split=.1, shuffle=True)

    elif loss_type=="category":

        model.fit(train_x, train_y.values, epochs=100, batch_size=batch_sizes, verbose=1, validation_split=.1, shuffle=True)
# model.fit(train_x, _.accuracy_group.values, epochs=20, batch_size=10, verbose=1, validation_split=.1, shuffle=True)
if loss_type=="mse":

    plt.figure(figsize=(40, 20))

    plt.subplot(2, 1, 1)

    plt.plot(model.history.history["loss"], "o-", alpha=.4, label="loss")

    plt.plot(model.history.history["val_loss"], "o-", alpha=.4, label="val_loss")

    plt.axhline(1.2, linestyle="--", c="C2")

    plt.legend()

    plt.subplot(2, 1, 2)

    plt.plot(model.history.history["loss"][3:], "o-", alpha=.4, label="loss")

    plt.plot(model.history.history["val_loss"][3:], "o-", alpha=.4, label="val_loss")

    plt.axhline(1.1, linestyle="--", c="C2")

    plt.legend()

    plt.show()



elif loss_type=="category":

    plt.figure(figsize=(40, 20))

    plt.subplot(2, 1, 1)

    plt.plot(model.history.history["loss"], "o-", alpha=.4, label="loss")

    plt.plot(model.history.history["val_loss"], "o-", alpha=.4, label="val_loss")

    plt.axhline(1.05, linestyle="--", c="C2")

    plt.legend()

    plt.subplot(2, 1, 2)

    plt.plot(model.history.history["categorical_accuracy"], "o-", alpha=.4, label="categorical_accuracy")

    plt.plot(model.history.history["val_categorical_accuracy"], "o-", alpha=.4, label="val_categorical_accuracy")

    plt.axhline(.65, linestyle="--", c="C2")

    plt.legend()

    plt.show()
np.sort(os.listdir("model"))
model = keras.models.load_model("model/"+np.sort(os.listdir("model"))[-1], custom_objects={'leaky_relu': tf.nn.leaky_relu})

np.sort(os.listdir("model"))[-1]
if validation:

    if loss_type=="mse":

        result = model.predict(val_x)

        result[result <= 1.12232214] = 0

        result[np.where(np.logical_and(result > 1.12232214, result <= 1.73925866))] = 1

        result[np.where(np.logical_and(result > 1.73925866, result <= 2.22506454))] = 2

        result[result > 2.22506454] = 3

        result = result.astype("int")

        print(quadratic_kappa(val_y.accuracy_group, result))

    elif loss_type=="category":

        result = model.predict(val_x)

        print(quadratic_kappa(val_y.accuracy_group, result.argmax(axis=1)))
if validation==False:

    if loss_type=="mse":

        result = model.predict(train_x)

        result[result <= 1.12232214] = 0

        result[np.where(np.logical_and(result > 1.12232214, result <= 1.73925866))] = 1

        result[np.where(np.logical_and(result > 1.73925866, result <= 2.22506454))] = 2

        result[result > 2.22506454] = 3

        result = result.astype("int")

        print(quadratic_kappa(train_y.accuracy_group, result))

    elif loss_type=="category":

        result = model.predict(train_x)

        print(quadratic_kappa(train_y.accuracy_group, result.argmax(axis=1)))
test = pd.read_csv("/kaggle/input/data-science-bowl-2019/test.csv", dtype=dtypes)

test["timestamp"] = pd.to_datetime(test.timestamp)



label = []

df_test = []

for idx, val in tqdm.tqdm_notebook(test.groupby(["installation_id"])):

    label.append(idx)

    df_test.append(val)
col = {}

for i in ["world_log", "world_time", "world_session", "event_id", "play_time", "gap_time", "session_count", "session_length", "session_time", "types", "title", "assessment", "time", "game"]:

    vars()[i].rename(columns={"ID":"installation_id"}, inplace=True)

    col[i] = list(vars()[i].columns)



# world_log

installation_id = []

world = []

size = []



for i in tqdm.tqdm_notebook(df_test):

    # world_log

    _ = i.groupby(["installation_id", "world"]).size().reset_index()

    installation_id.extend(_.installation_id)

    world.extend(_.world)

    size.extend(_[0])



world_log = pd.DataFrame(data={"installation_id":installation_id, "world":world, "size":size})

world_log = world_log.pivot_table(index="installation_id", columns="world", values="size")

world_log = world_log.fillna(0)

world_log.columns.name = None

world_log.reset_index(inplace=True)

world_log = world_log.loc[:, ["installation_id", "CRYSTALCAVES", "MAGMAPEAK", "TREETOPCITY", "NONE"]]

world_log = world_log.add_suffix("_l")

world_log.rename(columns={"installation_id_l":"installation_id"}, inplace=True)

world_log = world_log.loc[:, col["world_log"]]

world_log = world_log.fillna(0)



# world_time

installation_id = []

world = []

game_time = []



for i in tqdm.tqdm_notebook(df_test):

    # world_time

    _ = i.groupby(["installation_id", "world", "game_session"]).game_time.max().reset_index()

    installation_id.extend(_.installation_id)

    world.extend(_.world)

    game_time.extend(_.game_time)



world_time = pd.DataFrame(data={"installation_id":installation_id, "world":world, "game_time":game_time})

world_time = world_time.groupby(["installation_id", "world"]).sum().reset_index()

world_time = world_time.pivot_table(index="installation_id", columns="world", values="game_time")

world_time = world_time.fillna(-1)

world_time.columns.name = None

world_time["installation_id"] = world_time.index

world_time.reset_index(drop=True, inplace=True)

world_time = world_time.loc[:, ["installation_id", "CRYSTALCAVES", "MAGMAPEAK", "TREETOPCITY", "NONE"]]

world_time = world_time.add_suffix("_t")

world_time.rename(columns={"installation_id_t":"installation_id"}, inplace=True)

world_time = world_time.loc[:, col["world_time"]]

world_time = world_time.fillna(-1)



# world_session

installation_id = []

world = []

game_session = []



for i in tqdm.tqdm_notebook(df_test):

    # world_session

    _ = i.groupby(["installation_id", "world"]).game_session.nunique().reset_index()

    installation_id.extend(_.installation_id)

    world.extend(_.world)

    game_session.extend(_.game_session)



world_session = pd.DataFrame(data={"installation_id":installation_id, "world":world, "game_session":game_session})

world_session = world_session.pivot_table(index="installation_id", columns="world", values="game_session")

world_session = world_session.fillna(0)

world_session.columns.name = None

world_session["installation_id"] = world_session.index

world_session.reset_index(drop=True, inplace=True)

world_session = world_session.loc[:, ["installation_id", "CRYSTALCAVES", "MAGMAPEAK", "TREETOPCITY", "NONE"]]

world_session = world_session.add_suffix("_s")

world_session.rename(columns={"installation_id_s":"installation_id"}, inplace=True)

world_session = world_session.loc[:, col["world_session"]]

world_session = world_session.fillna(0)



# event_id

installation_id = []

event_id = []

size = []



for i in tqdm.tqdm_notebook(df_test):

    # event_id

    _ = i.groupby(["installation_id", "event_id"]).size().reset_index()

    installation_id.extend(_.installation_id)

    event_id.extend(_.event_id)

    size.extend(_[0])



event_id = pd.DataFrame(data={"installation_id":installation_id, "event_id":event_id, "size":size})

event_id = event_id.pivot_table(index="installation_id", columns="event_id", values="size")

event_id = event_id.fillna(0)

event_id.columns.name = None

event_id.index.name = None

event_id["installation_id"] = event_id.index

event_id.reset_index(drop=True, inplace=True)

event_id = event_id.loc[:, col["event_id"]]

event_id = event_id.fillna(0)



# play_time

installation_id = []

game_time = []



for i in tqdm.tqdm_notebook(df_test):

    # play_time

    _ = i.groupby(["installation_id", "game_session"]).game_time.max().reset_index()

    installation_id.extend(_.installation_id)

    game_time.extend(_.game_time)



play_time = pd.DataFrame(data={"installation_id":installation_id, "game_time":game_time})

play_time = play_time.groupby(["installation_id"]).sum().reset_index()

play_time.reset_index(drop=True, inplace=True)

play_time = play_time.fillna(0)

play_time = play_time.loc[:, col["play_time"]]

play_time = play_time.fillna(0)



# gap_time

gap_time = pd.DataFrame()

for i in tqdm.tqdm_notebook(df_test):

    # gap_time

    _ = i.groupby(["installation_id"]).timestamp.agg(["min", "max"])

    _.columns.name = None

    gap_time = pd.concat([gap_time, _], sort=True)



gap_time.reset_index(inplace=True)

gap_time["gap"] = gap_time["max"]-gap_time["min"]

gap_time["gap"] = gap_time["gap"].astype("int")

gap_time = gap_time.loc[:, col["gap_time"]]

gap_time = gap_time.fillna(0)



# session_count

session_count = pd.DataFrame()

for i in tqdm.tqdm_notebook(df_test):

    # session_count

    _ = i.groupby(["installation_id"]).game_session.nunique().reset_index()

    _.columns.name = None

    session_count = pd.concat([session_count, _], sort=False)



session_count.reset_index(drop=True, inplace=True)

session_count = session_count.loc[:, col["session_count"]]

session_count = session_count.fillna(0)



# session_length

session_length = pd.DataFrame()

for i in tqdm.tqdm_notebook(df_test):

    # session_length

#     _ = i.query("type!='Clip'").groupby(["installation_id", "game_session"]).size().groupby(["installation_id"]).mean().reset_index().rename(columns={0:"session_length"})

    _ = i.groupby(["installation_id", "game_session"]).size().groupby(["installation_id"]).mean().reset_index().rename(columns={0:"session_length"})

    _.columns.name = None

    session_length = pd.concat([session_length, _], sort=False)



session_length.reset_index(drop=True, inplace=True)

session_length = session_length.loc[:, col["session_length"]]

session_length = session_length.fillna(0)



# session_time

session_time = pd.DataFrame()

for i in tqdm.tqdm_notebook(df_test):

    # session_time

    _ = i.groupby(["installation_id", "game_session"]).game_time.max().groupby(["installation_id"]).mean().reset_index()

    _.columns.name = None

    session_time = pd.concat([session_time, _], sort=False)



session_time.reset_index(drop=True, inplace=True)

session_time = session_time.loc[:, col["session_time"]]

session_time = session_time.fillna(0)



# types

installation_id = []

types = []

size = []



for i in tqdm.tqdm_notebook(df_test):

    # types

    _ = i.groupby(["installation_id", "type"]).size().reset_index()

    installation_id.extend(_.installation_id)

    types.extend(_.type)

    size.extend(_[0])



types = pd.DataFrame(data={"installation_id":installation_id, "type":types, "size":size})

types = types.pivot_table(index="installation_id", columns="type", values="size")

types.columns.name = None

types.index.name = None

types = types.fillna(0)

types["installation_id"] = types.index

types = types.loc[:, ["installation_id", "Activity", "Assessment", "Clip", "Game"]]

types = types.loc[:, col["types"]]

types = types.fillna(0)



# title

installation_id = []

title = []

size = []



for i in tqdm.tqdm_notebook(df_test):

    # title

    _ = i.groupby(["installation_id", "title"]).size().reset_index()

    installation_id.extend(_.installation_id)

    title.extend(_.title)

    size.extend(_[0])



title = pd.DataFrame(data={"installation_id":installation_id, "title":title, "size":size})

title = title.pivot_table(index="installation_id", columns="title", values="size")

title.columns.name = None

title.index.name = None

title = title.fillna(0)

title["installation_id"] = title.index

title = title.loc[:, col["title"]]

title = title.fillna(0)



# assessment

assessment = pd.DataFrame(columns=["installation_id", "title"])

for i in tqdm.tqdm_notebook(df_test):

    # assessment

    _ = i.tail(1).loc[:, ["installation_id", "title"]].reset_index(drop=True)

    assessment = pd.concat([assessment, _], sort=False)



assessment['Assessment_1'] = 0

assessment['Assessment_2'] = 0

assessment['Assessment_3'] = 0

assessment['Assessment_4'] = 0

assessment['Assessment_5'] = 0



assessment.loc[assessment.title=='Mushroom Sorter (Assessment)', 'Assessment_1'] = 1

assessment.loc[assessment.title=='Cauldron Filler (Assessment)', 'Assessment_2'] = 1

assessment.loc[assessment.title=='Chest Sorter (Assessment)', 'Assessment_3'] = 1

assessment.loc[assessment.title=='Cart Balancer (Assessment)', 'Assessment_4'] = 1

assessment.loc[assessment.title=='Bird Measurer (Assessment)', 'Assessment_5'] = 1

del assessment["title"]

assessment = assessment.loc[:, col["assessment"]]

assessment = assessment.fillna(0)



# time

time = pd.DataFrame(columns=["installation_id", "timestamp"])

for i in tqdm.tqdm_notebook(df_test):

    # time

    _ = i.tail(1).loc[:, ["installation_id", "timestamp"]]

    time = pd.concat([time, _], sort=False)



time.reset_index(drop=True, inplace=True)

time["hour"] = time.timestamp.dt.hour

time["hour"] = time.hour.astype("object")

time = time.merge(pd.get_dummies(time.hour, prefix="hour"), how="left", 

                  left_index=True, right_index=True)

time.drop(columns=["timestamp", "hour"], inplace=True)

time = time.loc[:, col["time"]]

time = time.fillna(0)



# game

installation_id = []

game_title = []

game_round = []



for i in tqdm.tqdm_notebook(df_test):

    if "Game" in i.type.unique():

        _ = i.query("type=='Game'").loc[:, ["installation_id", "title", "event_data"]].set_index(["installation_id", "title"]).event_data.apply(lambda x: json.loads(x)["round"]).groupby(["installation_id", "title"]).max().reset_index()

        installation_id.extend(list(_.installation_id))

        game_title.extend(_.title)

        game_round.extend(_.event_data)

        

game = pd.DataFrame(data={"installation_id":installation_id, "game_title":game_title, "round":game_round})

game = game.pivot_table(index="installation_id", columns="game_title", values="round")

game.reset_index(inplace=True)

game.columns.name = None

game = game.fillna(-1)



installation_id = pd.DataFrame(data={"installation_id":label})

game = installation_id.merge(game, how="left")

game = game.fillna(-1)

game = game.add_suffix("_r")

game.rename(columns={"installation_id_r":"installation_id"}, inplace=True)

game = game.loc[:, col["game"]]

game = game.fillna(-1)
if dp_log==True:

    world_log.iloc[:, 1:] = np.log(world_log.iloc[:, 1:]+1)



# world_time.drop(columns=["NONE"], inplace=True)

if dp_log==True:

    world_time.iloc[:, 1:] = np.log(world_time.iloc[:, 1:]+2)



if dp_log==True:

    world_session.iloc[:, 1:] = np.log(world_session.iloc[:, 1:]+1)



if dp_log==True:

    event_id.iloc[:, :-1] = np.log(event_id.iloc[:, :-1]+1)

#     event_id.iloc[:, 1:] = np.log(event_id.iloc[:, 1:]+1)



if dp_log==True:

    play_time.iloc[:, 1:] = np.log(play_time.iloc[:, 1:]+1)



# gap_time.drop(columns=["max", "min"], inplace=True)

if dp_log==True:

    gap_time.iloc[:, 1:] = np.log(gap_time.iloc[:, 1:]+1)



if dp_log==True:

    session_count.iloc[:, 1:] = np.log(session_count.iloc[:, 1:])



if dp_log==True:

    session_length.iloc[:, 1:] = np.log(session_length.iloc[:, 1:])



if dp_log==True:

    session_time.iloc[:, 1:] = np.log(session_time.iloc[:, 1:]+1)



if dp_log==True:

    types.iloc[:, 1:] = np.log(types.iloc[:, 1:]+1)



if dp_log==True:

    title.iloc[:, :-1] = np.log(title.iloc[:, :-1]+1)
_ = world_log.merge(world_time, how="left", on=["installation_id"])

_ = _.merge(world_session, how="left", on=["installation_id"])

_ = _.merge(event_id, how="left", on=["installation_id"])

_ = _.merge(play_time, how="left", on=["installation_id"])

_ = _.merge(gap_time, how="left", on=["installation_id"])

_ = _.merge(session_count, how="left", on=["installation_id"])

_ = _.merge(session_length, how="left", on=["installation_id"])

_ = _.merge(session_time, how="left", on=["installation_id"])

_ = _.merge(types, how="left", on=["installation_id"])

_ = _.merge(title, how="left", on=["installation_id"])

_ = _.merge(assessment, how="left", on=["installation_id"])

_ = _.merge(time, how="left", on=["installation_id"])

_ = _.merge(game, how="left", on=["installation_id"])

train_x_col[0] = "installation_id"

_ = _.loc[:, train_x_col]

_ = _.fillna(-1)

_.to_csv("test.csv", index=False)



test_x = scaler.transform(_.loc[:, train_x_col[1:]])
result = model.predict(test_x)
# result[result <= 1.12232214] = 0

# result[np.where(np.logical_and(result > 1.12232214, result <= 1.73925866))] = 1

# result[np.where(np.logical_and(result > 1.73925866, result <= 2.22506454))] = 2

# result[result > 2.22506454] = 3

# result = result.astype("int")
if loss_type=="mse":

    submission = pd.DataFrame({"installation_id":_.installation_id, "accuracy_group":result.flatten()})

    submission.to_csv("submission.csv", index=False)

elif loss_type=="category":

    submission = pd.DataFrame({"installation_id":_.installation_id, "accuracy_group":result.argmax(axis=1)})

    submission.to_csv("submission.csv", index=False)
plt.figure(figsize=(20, 10))

plt.hist(submission.accuracy_group)

plt.show()
np.unique(submission.accuracy_group, return_counts=True)