import pandas as pd

import numpy as np



import json



from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split



import keras



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

window = 70

batch_sizes = 20

validation = True
train = pd.read_csv("/kaggle/input/data-science-bowl-2019/train.csv", dtype=dtypes)

test = pd.read_csv("/kaggle/input/data-science-bowl-2019/test.csv", dtype=dtypes)

label_ = pd.read_csv("/kaggle/input/data-science-bowl-2019/train_labels.csv", dtype=label)

# sample = pd.read_csv("/kaggle/input/data-science-bowl-2019/sample_submission.csv")

# specs = pd.read_csv("/kaggle/input/data-science-bowl-2019/specs.csv")
train.head()
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
event_id_col = ['003cd2ee', '0086365d', '00c73085', '01ca3a3c', '022b4259',

                '02a42007', '0330ab6a', '0413e89d', '04df9b66', '05ad839b',

                '06372577', '070a5291', '08fd73f3', '08ff79ad', '0a08139c',

                '0ce40006', '0d18d96c', '0d1da71f', '0db6d71d', '119b5b02',

                '1325467d', '1340b8d7', '1375ccb7', '13f56524', '14de4c5d',

                '155f62a4', '1575e76c', '15a43e5b', '15ba1109', '15eb4a7d',

                '15f99afc', '160654fd', '16667cc5', '16dffff1', '17113b36',

                '17ca3959', '19967db1', '1996c610', '1af8be29', '1b54d27f',

                '1bb5fbdb', '1beb320a', '1c178d24', '1cc7cfca', '1cf54632',

                '1f19558b', '222660ff', '2230fab4', '250513af', '25fa8af4',

                '262136f4', '26a5a3dd', '26fd2d99', '27253bdc', '28520915',

                '28a4eb9a', '28ed704e', '28f975ea', '29a42aea', '29bdd9ba',

                '29f54413', '2a444e03', '2a512369', '2b058fe3', '2b9272f4',

                '2c4e6db0', '2dc29e21', '2dcad279', '2ec694de', '2fb91ec1',

                '30614231', '30df3273', '31973d56', '3323d7e9', '33505eae',

                '3393b68b', '363c86c9', '363d3849', '36fa3ebe', '37937459',

                '37c53127', '37db1c2f', '37ee8496', '38074c54', '392e14df',

                '3a4be871', '3afb49e6', '3afde5dd', '3b2048ee', '3babcb9b',

                '3bb91ced', '3bb91dda', '3bf1cf26', '3bfd1a65', '3ccd3f02',

                '3d0b9317', '3d63345e', '3d8c61b0', '3dcdda7f', '3ddc79c3',

                '3dfd4aa4', '3edf6747', '3ee399c3', '4074bac2', '44cb4907',

                '45d01abe', '461eace6', '46b50ba8', '46cd75b4', '47026d5f',

                '47efca07', '47f43a44', '48349b14', '4901243f', '499edb7c',

                '49ed92e9', '4a09ace1', '4a4c3d21', '4b5efe37', '4bb2f698',

                '4c2ec19f', '4d6737eb', '4d911100', '4e5fc6f5', '4ef8cdd3',

                '51102b85', '51311d7a', '5154fc30', '5290eab1', '532a2afb',

                '5348fd84', '53c6e11a', '55115cbd', '562cec5f', '565a3990',

                '56817e2b', '56bcd38d', '56cd3b43', '5859dfb6', '587b5989',

                '58a0de5c', '598f4598', '5a848010', '5b49460a', '5be391b5',

                '5c2f29ca', '5c3d2b2f', '5d042115', '5dc079d8', '5de79a6a',

                '5e109ec3', '5e3ea25a', '5e812b27', '5f0eb72c', '5f5b2617',

                '6043a2b4', '6077cc36', '6088b756', '611485c5', '63f13dd7',

                '65a38bf7', '65abac75', '67439901', '67aa2ada', '69fdac0a',

                '6aeafed4', '6bf9e3e1', '6c517a88', '6c930e6e', '6cf7d25c',

                '6d90d394', '6f445b57', '6f4adc4b', '6f4bd64e', '6f8106d9',

                '7040c096', '709b1251', '71e712d8', '71fe8f75', '731c0cbe',

                '736f9581', '7372e1a5', '73757a5e', '7423acbc', '74e5f8a7',

                '7525289a', '756e5507', '763fc34e', '76babcde', '77261ab5',

                '77c76bc5', '77ead60d', '792530f8', '795e4a37', '7961e599',

                '7ab78247', '7ad3efc6', '7cf1bc53', '7d093bf9', '7d5c30a2',

                '7da34a02', '7dfe6d8a', '7ec0c298', '7f0836bf', '7fd1ac25',

                '804ee27f', '828e68f9', '832735e1', '83c6c409', '84538528',

                '84b0e0c8', '857f21c0', '85d1b0de', '85de926c', '86ba578b',

                '86c924c4', '87d743c1', '884228c8', '88d4a5be', '895865f3',

                '89aace00', '8ac7cce4', '8af75982', '8b757ab8', '8d748b58',

                '8d7e386c', '8d84fa81', '8f094001', '8fee50e2', '907a054b',

                '90d848e0', '90ea0bac', '90efca10', '91561152', '923afab1',

                '92687c59', '93b353f2', '93edfe2e', '9554a50b', '9565bea6',

                '99abe2bb', '99ea62f3', '9b01374f', '9b23e8ee', '9b4001e4',

                '9c5ef70c', '9ce586dd', '9d29771f', '9d4e7b25', '9de5e594',

                '9e34ea74', '9e4c8c7b', '9e6b7fb5', '9ed8f6da', '9ee1c98c',

                'a0faea5d', 'a1192f43', 'a16a373e', 'a1bbe385', 'a1e4395d',

                'a29c5338', 'a2df0760', 'a44b10dc', 'a52b92d5', 'a592d54e',

                'a5be6304', 'a5e9da97', 'a6d66e51', 'a76029ee', 'a7640a16',

                'a8876db3', 'a8a78786', 'a8cc6fec', 'a8efe47b', 'ab3136ba',

                'ab4ec3a4', 'abc5811c', 'ac92046e', 'acf5c23f', 'ad148f58',

                'ad2fc29c', 'b012cd7f', 'b120f2ac', 'b1d5101d', 'b2dba42b',

                'b2e5b0f1', 'b5053438', 'b738d3d3', 'b74258a0', 'b7530680',

                'b7dc8128', 'b80e5e84', 'b88f38da', 'bb3e370b', 'bbfe0445',

                'bc8f2793', 'bcceccc6', 'bd612267', 'bd701df8', 'bdf49a58',

                'beb0a7b9', 'bfc77bd6', 'c0415e5c', 'c189aaf2', 'c1cac9a2',

                'c277e121', 'c2baf0bd', 'c51d8688', 'c54cf6c5', 'c58186bf',

                'c6971acf', 'c7128948', 'c74f40cd', 'c7f7f0e1', 'c7fe2a55',

                'c952eb01', 'ca11f653', 'cb1178ad', 'cb6010f8', 'cc5087a3',

                'cdd22e43', 'cf7638f3', 'cf82af56', 'cfbd47c8', 'd02b7a8e',

                'd06f75b5', 'd122731b', 'd185d3ea', 'd2278a3b', 'd2659ab4',

                'd2e9262e', 'd3268efa', 'd3640339', 'd38c2fd7', 'd3f1e122',

                'd45ed6a1', 'd51b1749', 'd88ca108', 'd88e8f25', 'd9c005dd',

                'daac11b0', 'db02c830', 'dcaede90', 'dcb1663e', 'dcb55a27',

                'de26c3a6', 'df4940d3', 'df4fe8b6', 'e04fb33d', 'e080a381',

                'e37a2b78', 'e3ff61fb', 'e4d32835', 'e4f1efe6', 'e5734469',

                'e57dd7af', 'e5c9df6f', 'e64e2cfd', 'e694a35b', 'e720d930',

                'e7561dd2', 'e79f3763', 'e7e44842', 'e9c52111', 'ea296733',

                'ea321fb1', 'eb2c19cd', 'ec138c1c', 'ecaab346', 'ecc36b7f',

                'ecc6157f', 'f28c589a', 'f32856e4', 'f3cd5473', 'f50fc6c1',

                'f54238ee', 'f56e0afc', 'f5b8c21a', 'f6947f54', 'f71c4741',

                'f7e47413', 'f806dc10', 'f93fc684', 'fbaf3456', 'fcfdffb6',

                'fd20ea40']



event_code_col = [2000, 2010, 2020, 2025, 2030, 2035, 2040, 

                  2050, 2060, 2070, 2075, 2080, 2081, 2083, 

                  3010, 3020, 3021, 3110, 3120, 3121, 4010, 

                  4020, 4021, 4022, 4025, 4030, 4031, 4035, 

                  4040, 4045, 4050, 4070, 4080, 4090, 4095, 

                  4100, 4110, 4220, 4230, 4235, 5000, 5010]



world_col = ["NONE", "CRYSTALCAVES", "MAGMAPEAK", "TREETOPCITY"]



type_col = ["Activity", "Assessment", "Clip", "Game"]
len(event_code_col), len(world_col), len(type_col)
# event_code

df = []

max_ = 0

for val in tqdm.tqdm_notebook(df_train):

# for val in tqdm.notebook.tqdm(df_train):

    # game_session

    game_session = val[-2:].game_session.values

    val = val.query("game_session in @game_session")

    # event_code

    event_code = pd.get_dummies(val[::-1].reset_index(drop=True)[:window].event_code).loc[:, event_code_col].fillna(0).astype("int8").values

    event_code = np.append(event_code, np.zeros((window-event_code.shape[0], 42)), axis=0)

    # world

    world = pd.get_dummies(val[::-1].reset_index(drop=True)[:window].world).loc[:, world_col].fillna(0).astype("int8").values

    world = np.append(world, np.zeros((window-world.shape[0], 4)), axis=0)

    # type

    types = pd.get_dummies(val[::-1].reset_index(drop=True)[:window].type).loc[:, type_col].fillna(0).astype("int8").values

    types = np.append(types, np.zeros((window-types.shape[0], 4)), axis=0)

    # game_time

    game_time = val.timestamp.diff()[::-1].reset_index(drop=True)[:window].values

    game_time[np.isnat(game_time)] = 0

    game_time = game_time.reshape(len(game_time), 1)

    game_time

#     game_time = MinMaxScaler().fit_transform(game_time)

    game_time.dtype = "int"

    if max_<game_time.max():

        max_ = game_time.max()

    game_time = np.append(game_time, np.zeros((window-game_time.shape[0], 1)), axis=0)

    

    _ = np.hstack((event_code, world, types, game_time))

    

    if val["title"].iloc[-1] == 'Mushroom Sorter (Assessment)':

        _ = np.hstack((_, np.array([1, 0, 0, 0, 0])*np.ones((window, 5))))

    elif val["title"].iloc[-1] == 'Cauldron Filler (Assessment)':

        _ = np.hstack((_, np.array([0, 1, 0, 0, 0])*np.ones((window, 5))))

    elif val["title"].iloc[-1] == 'Chest Sorter (Assessment)':

        _ = np.hstack((_, np.array([0, 0, 1, 0, 0])*np.ones((window, 5))))

    elif val["title"].iloc[-1] == 'Cart Balancer (Assessment)':

        _ = np.hstack((_, np.array([0, 0, 0, 1, 0])*np.ones((window, 5))))

    elif val["title"].iloc[-1] == 'Bird Measurer (Assessment)':

        _ = np.hstack((_, np.array([0, 0, 0, 0, 1])*np.ones((window, 5))))



    df.append(np.flip(_, axis=0))
df = np.array(df)

df[:, :, -6] = df[:, :, -6]/max_

df.shape
del df_train

gc.collect()
label.head()
# remove 8 data for fit

df = df[:19700]

label = label.loc[:19699, :]
_ = np.unique(label.accuracy_group, return_counts=True)

class_weight_ = {_[0][0]:_[1][0]/len(label), 

                 _[0][1]:_[1][1]/len(label), 

                 _[0][2]:_[1][2]/len(label), 

                 _[0][3]:_[1][3]/len(label)}

class_weight_
from sklearn.utils import class_weight

class_weights = class_weight.compute_class_weight('balanced', np.unique(label.accuracy_group),

                                                  label.accuracy_group)
np.unique(label.accuracy_group), class_weights
# label["class_weight"] = 0

# label.loc[label.accuracy_group==0, "class_weight"] = class_weight_[0]

# label.loc[label.accuracy_group==1, "class_weight"] = class_weight_[1]

# label.loc[label.accuracy_group==2, "class_weight"] = class_weight_[2]

# label.loc[label.accuracy_group==3, "class_weight"] = class_weight_[3]
label["class_weight"] = 0

label.loc[label.accuracy_group==0, "class_weight"] = class_weights[0]

label.loc[label.accuracy_group==1, "class_weight"] = class_weights[1]

label.loc[label.accuracy_group==2, "class_weight"] = class_weights[2]

label.loc[label.accuracy_group==3, "class_weight"] = class_weights[3]
if validation:

    train_x, val_x, train_y, val_y = train_test_split(df, label, test_size=4700, random_state=1228)

    plt.hist(train_y.accuracy_group, align="left", label="train", alpha=.7)

    plt.hist(val_y.accuracy_group, align="right", label="val", alpha=.7)

    plt.show()

    display(train_x.shape, train_y.shape)    
# import keras.backend as K



# def quadratic_kappa(y_true, y_pred, N=4):

#     w = np.zeros((4,4))

#     O = confusion_matrix(y_true, y_pred)

#     for i in range(len(w)): 

#         for j in range(len(w)):

#             w[i][j] = float(((i-j)**2)/(4-1)**2)

    

#     act_hist=np.zeros([4])

#     for item in actuals: 

#         act_hist[item]+=1

    

#     pred_hist=np.zeros([4])

#     for item in preds: 

#         pred_hist[item]+=1

                         

#     E = np.outer(act_hist, pred_hist);

#     E = E/E.sum();

#     O = O/O.sum();

    

#     num=0

#     den=0

#     for i in range(len(w)):

#         for j in range(len(w)):

#             num+=w[i][j]*O[i][j]

#             den+=w[i][j]*E[i][j]

#     return K.mean(1 - (num/den))



# import keras.backend as K

# def _cohen_kappa(y_true, y_pred, num_classes, weights=None, metrics_collections=None, updates_collections=None, name=None):

#     kappa, update_op = tf.contrib.metrics.cohen_kappa(y_true, y_pred, num_classes, weights, metrics_collections, updates_collections, name)

#     K.get_session().run(tf.local_variables_initializer())

#     with tf.control_dependencies([update_op]):

#         kappa = tf.identity(kappa)

#     return kappa



# def cohen_kappa_loss(num_classes, weights=None, metrics_collections=None, updates_collections=None, name=None):

#     def cohen_kappa(y_true, y_pred):

#         return -_cohen_kappa(y_true, y_pred, num_classes, weights, metrics_collections, updates_collections, name)

#     return cohen_kappa

# def mean_pred(y_true, y_pred):

#     return K.mean(y_pred)



# model.compile(optimizer='rmsprop',

#               loss='binary_crossentropy',

#               metrics=['accuracy', mean_pred])
RNN = keras.models.Sequential()
# RNN.add(keras.layers.LSTM(units=30, stateful=False, return_sequences=True, return_state=False, 

#                           recurrent_dropout=.2, batch_input_shape=(batch_sizes, df.shape[1], df.shape[2])))

# RNN.add(keras.layers.LSTM(units=20, stateful=False, return_sequences=True, return_state=False, 

#                           recurrent_dropout=.2))

# # RNN.add(keras.layers.LSTM(units=10, stateful=False, return_sequences=True, return_state=False, 

# #                           recurrent_dropout=.2))

# RNN.add(keras.layers.TimeDistributed(keras.layers.Dense(units=10, activation="relu", 

#                                                         kernel_initializer="he_normal")))

# RNN.add(keras.layers.LSTM(units=5, stateful=False, return_sequences=False, return_state=False, 

#                           recurrent_dropout=.2))

# RNN.add(keras.layers.Dense(units=10, activation="relu", kernel_initializer="he_normal"))

# RNN.add(keras.layers.Dropout(.2))

# RNN.add(keras.layers.Dense(units=10, activation="relu", kernel_initializer="he_normal"))



# if loss_type=="mse":

#     RNN.add(keras.layers.Dense(units=1, activation="relu", name="output"))

# elif loss_type=="category":

#     RNN.add(keras.layers.Dense(units=4, activation="softmax", name="output"))
# RNN.add(keras.layers.LSTM(units=50, stateful=False, return_sequences=True, return_state=False, 

#                           dropout=.2, recurrent_dropout=.2, 

#                           batch_input_shape=(batch_sizes, df.shape[1], df.shape[2])))

# RNN.add(keras.layers.LSTM(units=20, stateful=False, return_sequences=False, return_state=False, 

#                           dropout=.2, recurrent_dropout=.2))

# RNN.add(keras.layers.Dense(units=10, activation="relu", kernel_initializer="he_normal"))

# RNN.add(keras.layers.Dropout(.2))



# if loss_type=="mse":

#     RNN.add(keras.layers.Dense(units=1, activation="relu", name="output"))

# elif loss_type=="category":

#     RNN.add(keras.layers.Dense(units=4, activation="softmax", name="output"))
RNN.add(keras.layers.LSTM(units=30, stateful=False, return_sequences=True, return_state=False, 

                          dropout=.3, recurrent_dropout=.3, batch_input_shape=(batch_sizes, df.shape[1], df.shape[2])))

RNN.add(keras.layers.LSTM(units=20, stateful=False, return_sequences=True, return_state=False, 

                          dropout=.3, recurrent_dropout=.3))

RNN.add(keras.layers.TimeDistributed(keras.layers.Dense(units=10, activation="relu", 

                                                        kernel_initializer="he_normal")))

RNN.add(keras.layers.Flatten())

RNN.add(keras.layers.Dense(units=10, activation="relu", kernel_initializer="he_normal"))

RNN.add(keras.layers.Dropout(.3))

RNN.add(keras.layers.Dense(units=10, activation="relu", kernel_initializer="he_normal"))



if loss_type=="mse":

    RNN.add(keras.layers.Dense(units=1, activation="relu", name="output"))

elif loss_type=="category":

    RNN.add(keras.layers.Dense(units=4, activation="softmax", name="output"))
if loss_type=="mse":

    RNN.compile(loss="mse", optimizer="rmsprop")

elif loss_type=="category":

    RNN.compile(loss="categorical_crossentropy", optimizer="Adam", 

                metrics=['categorical_accuracy'])
# ["SGD", "RMSprop", "Adagrad", "Adadelta", "Adam", "Adamax", "Nadam"]

# keras.backend.reset_uids()
RNN.summary()
!rm -rf "/kaggle/working/model/"
if not os.path.exists("model"):

    os.mkdir("model")
if validation==False:

    if loss_type=="mse":

        RNN.fit(x=df[:19700], y=label.loc[:19699, ["accuracy_group"]], 

                epochs=10, batch_size=batch_sizes, shuffle=True, class_weight=class_weight)

    elif loss_type=="category":

        RNN.fit(x=df[:19700], y=label.loc[:19699, ["y_0", "y_1", "y_2", "y_3"]],

                epochs=1000, batch_size=batch_sizes, shuffle=True, 

                sample_weight=label.loc[:, ["class_weight"]].values.flatten(), 

                callbacks=[keras.callbacks.EarlyStopping(monitor="categorical_accuracy", 

                                                         patience=50, mode="auto"), 

                           keras.callbacks.ModelCheckpoint("model/weights.{epoch:02d}-{categorical_accuracy:.3f}.hdf5", 

                                                           monitor='categorical_accuracy', 

                                                           verbose=0, save_best_only=True, save_weights_only=False, 

                                                           mode="auto", period=1)])
if validation:

    if loss_type=="mse":

        RNN.fit(x=train_x, y=train_y.loc[:, ["accuracy_group"]], 

                validation_data=[val_x, val_y.loc[:, ["accuracy_group"]]], 

                epochs=50, batch_size=batch_sizes, shuffle=True, class_weight=class_weight)

    elif loss_type=="category":

        RNN.fit(x=train_x, y=train_y.loc[:, ["y_0", "y_1", "y_2", "y_3"]], 

                validation_data=[val_x, val_y.loc[:, ["y_0", "y_1", "y_2", "y_3"]]], 

                epochs=1000, batch_size=batch_sizes, shuffle=True, 

                sample_weight=train_y.loc[:, ["class_weight"]].values.flatten(), 

                callbacks=[keras.callbacks.EarlyStopping(monitor="val_categorical_accuracy", 

                                                         patience=50, mode="auto"), 

                           keras.callbacks.ModelCheckpoint("model/weights.{epoch:02d}-{val_categorical_accuracy:.3f}.hdf5", 

                                                           monitor='val_categorical_accuracy', 

                                                           verbose=0, save_best_only=True, save_weights_only=False, 

                                                           mode="auto", period=1)])
if validation:

    if loss_type=="mse":

        plt.figure(figsize=(20, 10))

        plt.plot(RNN.history.history["loss"], "o-", alpha=.4, label="loss")

        plt.plot(RNN.history.history["val_loss"], "o-", alpha=.4, label="val_loss")

        plt.axhline(1.2, linestyle="--", c="C2")

        plt.legend()

        plt.show()

    elif loss_type=="category":

        plt.figure(figsize=(20, 10))

        plt.subplot(2, 1, 1)

        plt.plot(RNN.history.history["loss"], "o-", alpha=.4, label="loss")

        plt.plot(RNN.history.history["val_loss"], "o-", alpha=.4, label="val_loss")

        plt.axhline(1.25, linestyle="--", c="C2")

        plt.legend()

        plt.subplot(2, 1, 2)

        plt.plot(RNN.history.history["categorical_accuracy"], "o-", alpha=.4, label="categorical_accuracy")

        plt.plot(RNN.history.history["val_categorical_accuracy"], "o-", alpha=.4, label="val_categorical_accuracy")

        plt.axhline(.6, linestyle="--", c="C2")

        plt.legend()

        plt.show()

        

if not validation:

    if loss_type=="mse":

        plt.figure(figsize=(20, 10))

        plt.plot(RNN.history.history["loss"], "o-", alpha=.4, label="loss")

        plt.axhline(1.2, linestyle="--", c="C2")

        plt.legend()

        plt.show()

    elif loss_type=="category":

        plt.figure(figsize=(20, 10))

        plt.subplot(2, 1, 1)

        plt.plot(RNN.history.history["loss"], "o-", alpha=.4, label="loss")

        plt.axhline(1.25, linestyle="--", c="C2")

        plt.legend()

        plt.subplot(2, 1, 2)

        plt.plot(RNN.history.history["categorical_accuracy"], "o-", alpha=.4, label="categorical_accuracy")

        plt.axhline(.6, linestyle="--", c="C2")

        plt.legend()

        plt.show()
os.listdir("model")
RNN = keras.models.load_model("model/"+os.listdir("model")[-2])

os.listdir("model")[-2]
result = RNN.predict(df, batch_size=batch_sizes)

plt.figure(figsize=(20, 10))

if loss_type=="mse":

    plt.hist(result, alpha=.7, label="predict")

    plt.hist(label.accuracy_group, alpha=.7, label="real")

    plt.legend()

elif loss_type=="category":

    plt.hist(result.argmax(axis=1), alpha=.4, align="left", label="predict")

    plt.hist(label.accuracy_group, alpha=.4, align="right", label="real")

    plt.legend()

plt.show()
if validation:

    result = RNN.predict(val_x, batch_size=batch_sizes)

    plt.figure(figsize=(20, 10))

    if loss_type=="mse":

        plt.hist(result, alpha=.7, label="predict")

        plt.hist(val_y.accuracy_group, alpha=.7, label="real")

        plt.legend()

    elif loss_type=="category":

        plt.hist(result.argmax(axis=1), alpha=.4, align="left", label="predict")

        plt.hist(val_y.accuracy_group, alpha=.4, align="right", label="real")

        plt.legend()

    plt.show()
if validation and loss_type=="category":

    print(np.unique(result.argmax(axis=1), return_counts=True))

    print(quadratic_kappa(val_y.accuracy_group.reset_index(drop=True), result.argmax(axis=1)))

    print(confusion_matrix(val_y.accuracy_group.reset_index(drop=True), result.argmax(axis=1)))

    print(confusion_matrix(result.argmax(axis=1), val_y.accuracy_group.reset_index(drop=True)))

    

if not validation and loss_type=="category":

    print(np.unique(result.argmax(axis=1), return_counts=True))

    print(quadratic_kappa(label.accuracy_group.reset_index(drop=True), result.argmax(axis=1)))

    print(confusion_matrix(label.accuracy_group.reset_index(drop=True), result.argmax(axis=1)))

    print(confusion_matrix(result.argmax(axis=1), label.accuracy_group.reset_index(drop=True)))
label
del train_x, val_x, train_y, val_y, result

gc.collect()



test = pd.read_csv("/kaggle/input/data-science-bowl-2019/test.csv", dtype=dtypes)

test["timestamp"] = pd.to_datetime(test.timestamp)



# event_code

df_test = []

df_id = []

for idx, val in tqdm.tqdm_notebook(test.groupby("installation_id", sort=False)):

# for idx, val in tqdm.notebook.tqdm(test.groupby("installation_id", sort=False)):

    df_id.append(idx)

    # game_session

    game_session = val[-2:].game_session.values

    val = val.query("game_session in @game_session")

    # event_code

    event_code = pd.get_dummies(val[::-1].reset_index(drop=True)[:window].event_code).loc[:, event_code_col].fillna(0).astype("int8").values

    event_code = np.append(event_code, np.zeros((window-event_code.shape[0], 42)), axis=0)

    # world

    world = pd.get_dummies(val[::-1].reset_index(drop=True)[:window].world).loc[:, world_col].fillna(0).astype("int8").values

    world = np.append(world, np.zeros((window-world.shape[0], 4)), axis=0)

    # type

    types = pd.get_dummies(val[::-1].reset_index(drop=True)[:window].type).loc[:, type_col].fillna(0).astype("int8").values

    types = np.append(types, np.zeros((window-types.shape[0], 4)), axis=0)

    # game_time

    game_time = val.timestamp.diff()[::-1].reset_index(drop=True)[:window].values

    game_time[np.isnat(game_time)] = 0

    game_time = game_time.reshape(len(game_time), 1)

#     game_time = MinMaxScaler().fit_transform(game_time)

    game_time.dtype = "int"

    game_time = np.append(game_time, np.zeros((window-game_time.shape[0], 1)), axis=0)

    

    _ = np.hstack((event_code, world, types, game_time))

    

    if val["title"].iloc[-1]=='Mushroom Sorter (Assessment)':

        _ = np.hstack((_, np.array([1, 0, 0, 0, 0])*np.ones((window, 5))))

    elif val["title"].iloc[-1]=='Cauldron Filler (Assessment)':

        _ = np.hstack((_, np.array([0, 1, 0, 0, 0])*np.ones((window, 5))))

    elif val["title"].iloc[-1]=='Chest Sorter (Assessment)':

        _ = np.hstack((_, np.array([0, 0, 1, 0, 0])*np.ones((window, 5))))

    elif val["title"].iloc[-1]=='Cart Balancer (Assessment)':

        _ = np.hstack((_, np.array([0, 0, 0, 1, 0])*np.ones((window, 5))))

    elif val["title"].iloc[-1]=='Bird Measurer (Assessment)':

        _ = np.hstack((_, np.array([0, 0, 0, 0, 1])*np.ones((window, 5))))



    df_test.append(np.flip(_, axis=0))



df_test = np.array(df_test)

df_test[:, :, -6] = df_test[:, :, -6]/max_
df_test.shape
result = RNN.predict(df_test, batch_size=batch_sizes)

if loss_type=="mse":

    plt.figure(figsize=(20, 10))

    sns.distplot(result)

    plt.show()

elif loss_type=="category":

    plt.figure(figsize=(20, 10))

    plt.hist(result.argmax(axis=1))

    plt.show()
if loss_type=="mse":

    _ = pd.qcut(result.flatten(), 14)

    result[result<=_.categories[2].right] = 0

    result[np.where(np.logical_and(result>_.categories[2].right, 

                                   result<=_.categories[4].right))] = 1

    result[np.where(np.logical_and(result>_.categories[4].right, 

                                   result<=_.categories[6].right))] = 2

    result[result>_.categories[6].right] = 3

    result = result.astype("int")

    

#     result[result<= 1.12232214] = 0

#     result[np.where(np.logical_and(result>1.12232214, 

#                                    result<=1.73925866))] = 1

#     result[np.where(np.logical_and(result>1.73925866, 

#                                    result<=2.22506454))] = 2

#     result[result> 2.22506454] = 3

#     result = result.astype("int")



    submission = pd.DataFrame({"installation_id":df_id, "accuracy_group":result.flatten()})

    submission.to_csv("submission.csv", index=False)

elif loss_type=="category":

    submission = pd.DataFrame({"installation_id":df_id, "accuracy_group":result.argmax(axis=1)})

    submission.to_csv("submission.csv", index=False)
plt.figure(figsize=(20, 10))

plt.hist(submission.accuracy_group)

plt.show()
np.unique(submission.accuracy_group, return_counts=True)
os.listdir()