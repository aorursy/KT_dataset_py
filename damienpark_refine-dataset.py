import pandas as pd

import numpy as np

import json

import tqdm

import gc



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

        _["accuracy_group_"] = _["num_incorrect_"].apply(lambda x : 3 if x==0 else (2 if x==1 else 1))*_["num_correct_"]



        return _.loc[:, ["installation_id", "game_session", "num_correct_", "num_incorrect_", "accuracy_", "accuracy_group_"]]



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

        _["accuracy_group_"] = _["num_incorrect_"].apply(lambda x : 3 if x==0 else (2 if x==1 else 1))*_["num_correct_"]



        return _.loc[:, ["installation_id", "game_session", "num_correct_", "num_incorrect_", "accuracy_", "accuracy_group_"]]

    

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

        _["accuracy_group_"] = _["num_incorrect_"].apply(lambda x : 3 if x==0 else (2 if x==1 else 1))*_["num_correct_"]



        return _.loc[:, ["installation_id", "game_session", "num_correct_", "num_incorrect_", "accuracy_", "accuracy_group_"]]



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

        _["accuracy_group_"] = _["num_incorrect_"].apply(lambda x : 3 if x==0 else (2 if x==1 else 1))*_["num_correct_"]



        return _.loc[:, ["installation_id", "game_session", "num_correct_", "num_incorrect_", "accuracy_", "accuracy_group_"]]

    

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

        _["accuracy_group_"] = _["num_incorrect_"].apply(lambda x : 3 if x==0 else (2 if x==1 else 1))*_["num_correct_"]



        return _.loc[:, ["installation_id", "game_session", "num_correct_", "num_incorrect_", "accuracy_", "accuracy_group_"]]

    

    def bird_assessment_2(self):

        _ = self.df.query("title=='Bird Measurer (Assessment)' and event_id=='f6947f54'")

        _ = _.loc[:, ["game_session", "installation_id", "event_data"]]

        _["misses"] = _.event_data.apply(lambda x:(json.loads(x)["misses"] if "misses" in json.loads(x).keys() else -999))

        _["num_correct_"] = 1

        _ = _.groupby("game_session").sum().reset_index()

        _["accuracy_"] = _["num_correct_"]/(_["num_correct_"]+_["misses"])



        return _.loc[:, ["game_session", "num_correct_", "misses", "accuracy_"]]
train = pd.read_csv("/kaggle/input/data-science-bowl-2019/train.csv", dtype=dtypes)

test = pd.read_csv("/kaggle/input/data-science-bowl-2019/test.csv", dtype=dtypes)

label_ = pd.read_csv("/kaggle/input/data-science-bowl-2019/train_labels.csv")

# sample = pd.read_csv("/kaggle/input/data-science-bowl-2019/sample_submission.csv")

# specs = pd.read_csv("/kaggle/input/data-science-bowl-2019/specs.csv")
test["timestamp"] = pd.to_datetime(test.timestamp)

test.sort_values(["timestamp", "event_count"], ascending=True, inplace=True)
_ = accuracy(test).cart_assessment()

_ = _.append(accuracy(test).chest_assessment(), ignore_index=True)

_ = _.append(accuracy(test).cauldron_assessment(), ignore_index=True)

_ = _.append(accuracy(test).mushroom_assessment(), ignore_index=True)

_ = _.append(accuracy(test).bird_assessment(), ignore_index=True)
test = test[test.installation_id.isin(pd.unique(_.installation_id))]

test = test.merge(_, how="left", on=["installation_id", "game_session"])

test = test.loc[:, ['event_id', 'game_session', 'timestamp', 'event_data',

                    'installation_id', 'event_count', 'event_code', 'game_time', 

                    'title', 'type', 'world', 'accuracy_group_']]

test.rename(columns={"accuracy_group_":"accuracy_group"}, inplace=True)
%%time

df_test = []

idx = 0

for _, val in tqdm.tqdm_notebook(test.groupby("installation_id", sort=False)):

    val.reset_index(drop=True, inplace=True)

    _ = val.query("type=='Assessment'")

    _ = _[~_.accuracy_group.isnull()]

    session = _.reset_index().groupby("game_session", sort=False).index.first().values

    for j in session:

        sample = val[:j+1]

        sample["ID"] = idx

        idx += 1

        df_test.append(sample)



del test

df_test = pd.concat(df_test, axis=0, ignore_index=True)



label = df_test.groupby(["ID"]).accuracy_group.last().reset_index()

gc.collect()
# df_test.num_correct_ = df_test.num_correct_.astype("float16")

# df_test.num_incorrect_ = df_test.num_incorrect_.astype("float16")

# df_test.accuracy_ = df_test.accuracy_.astype("float16")

df_test.accuracy_group = df_test.accuracy_group.astype("float16")

df_test.ID = df_test.ID.astype("int16")
# data merge and sort

df = train[train.installation_id.isin(pd.unique(label_.installation_id))]

del train

df = df.merge(label_, on=["installation_id", "game_session", "title"], how="left")

del label_

df["timestamp"] = pd.to_datetime(df.timestamp)

df.sort_values(["timestamp", "event_count"], ascending=True, inplace=True)

df.reset_index(drop=True, inplace=True)

df = df.loc[:, ['event_id', 'game_session', 'timestamp', 'event_data',

                'installation_id', 'event_count', 'event_code', 'game_time', 

                'title', 'type', 'world', 'accuracy_group']]

gc.collect()
%%time

df_train = []

idx = max(label.ID)+1

for _, val in tqdm.tqdm_notebook(df.groupby("installation_id", sort=False)):

    val.reset_index(drop=True, inplace=True)

    session = val.query("type=='Assessment'").reset_index().groupby("game_session", sort=False).index.first().values

    for j in session:

        if ~np.isnan(val.iat[j, -1]):

            sample = val[:j+1]

            sample["ID"] = idx

            idx += 1

            df_train.append(sample)



del df

df_train = pd.concat(df_train, axis=0, ignore_index=True)



_ = df_train.groupby(["ID"]).accuracy_group.last().reset_index()

label = pd.concat([label, _], axis=0, ignore_index=True)



label.ID = label.ID.astype("int16")

label.accuracy_group = label.accuracy_group.astype("int8")



df = pd.concat([df_test, df_train], axis=0, ignore_index=True)

del df_test, df_train
df = df.loc[:, ["ID", 'event_id', 'game_session', 'timestamp', 'event_data',

                'installation_id', 'event_count', 'event_code', 'game_time', 

                'title', 'type', 'world', 'accuracy_group']]

df.ID = df.ID.astype("int16")

df.accuracy_group = df.accuracy_group.astype("float16")



_ = pd.unique(df.game_session)

_ = pd.DataFrame({"game_session":_, "game_session_":np.arange(0, len(_))})

df = df.merge(_, how="left", on=["game_session"])

del _, df["game_session"]

gc.collect()



_ = pd.unique(df.installation_id)

_ = pd.DataFrame({"installation_id":_, "installation_id_":np.arange(0, len(_))})

df = df.merge(_, how="left", on=["installation_id"])

del _, df["installation_id"]

gc.collect()



df.game_session_ = df.game_session_.astype("int32")

df.installation_id_ = df.installation_id_.astype("int16")

df.rename(columns={"game_session_":"game_session", "installation_id_":"installation_id"}, inplace=True)



df = df.loc[:, ["ID", 'event_id', 'game_session', 'timestamp', 'event_data',

                'installation_id', 'event_count', 'event_code', 'game_time', 

                'title', 'type', 'world', 'accuracy_group']]

display(df.head()), display(label.head())
# df.to_csv("df_train.csv", index=False)

# label.to_csv("label.csv", index=False)