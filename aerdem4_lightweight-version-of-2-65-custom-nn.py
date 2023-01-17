import numpy as np

import pandas as pd



dev_df = pd.read_csv("../input/champs-scalar-coupling/train.csv")

test_df = pd.read_csv("../input/champs-scalar-coupling/test.csv")



dev_df.shape, test_df.shape
dev_df.head()
dev_df["median_error"] = np.abs(dev_df["scalar_coupling_constant"] - dev_df.groupby("type")["scalar_coupling_constant"].transform("median"))



stats = dev_df.groupby("type").agg({"id": "count", "median_error": "mean"}).reset_index()

stats["effect"] = stats["id"]*stats["median_error"]

stats["effect"] /= stats["effect"].min()

stats["sw"] = 1/stats["effect"]



dev_df = dev_df.merge(stats[["type", "sw"]], on="type", how="left")



del stats
from sklearn.preprocessing import LabelEncoder



le_type = LabelEncoder()

dev_df["le_type"] = le_type.fit_transform(dev_df["type"].values)

test_df["le_type"] = le_type.transform(test_df["type"].values)
structures = pd.read_csv("../input/champs-scalar-coupling/structures.csv")

structures.head()
def get_loc(df):

    df = df.merge(structures.rename(columns={"x": "x0", "y": "y0", "z":"z0", "atom": "atom0", "atom_index": "atom_index_0"}), 

                          on=["molecule_name", "atom_index_0"], how="left")

    df = df.merge(structures.rename(columns={"x": "x1", "y": "y1", "z":"z1", "atom": "atom1", "atom_index": "atom_index_1"}), 

                          on=["molecule_name", "atom_index_1"], how="left")

    return df



dev_df = get_loc(dev_df)

test_df = get_loc(test_df)

dev_df.head()
mol_df = structures.groupby("molecule_name")["atom"].count().reset_index().rename(columns={"atom": "atom_count"})



def get_features(df):

    df = df.merge(mol_df, on="molecule_name", how="left")

    df["dist"] = 1.0/np.linalg.norm(df[["x0", "y0", "z0"]].values - df[["x1", "y1", "z1"]].values, axis=1)

    return df



dev_df = get_features(dev_df)

test_df = get_features(test_df)

dev_df.head()
structures["mol_features"] = structures[["x", "y", "z"]].values.tolist()

mol_df = structures.groupby("molecule_name").agg({"mol_features": list, "atom": list}).reset_index()



dev_df = dev_df.merge(mol_df, on="molecule_name", how="left")

test_df = test_df.merge(mol_df, on="molecule_name", how="left")



dev_df.head()
for col in dev_df.columns:

    if dev_df[col].dtype == np.int64:

        dev_df[col] = dev_df[col].astype(np.int32)

        

    if dev_df[col].dtype == np.float64:

        dev_df[col] = dev_df[col].astype(np.float32)

        

for col in test_df.columns:

    if test_df[col].dtype == np.int64:

        test_df[col] = test_df[col].astype(np.int32)

        

    if test_df[col].dtype == np.float64:

        test_df[col] = test_df[col].astype(np.float32)
from tqdm import tqdm, tqdm_notebook

from sklearn.metrics.pairwise import euclidean_distances



def get_mol(df):

    M = np.zeros((df.shape[0], 30, 5), dtype=np.float32)



    index = 0



    for properties, cx, mol_features, atom in tqdm_notebook(list(zip(df[["atom_index_0", "atom_index_1"]].values, 

                                                          df[["x0", "y0", "z0", "x1", "y1", "z1"]].values, 

                                                          df["mol_features"].values, df["atom"].values))):

        arr = np.array(mol_features)

        arr = np.delete(arr, np.array([properties[0], properties[1]]), 0)



        atoms = np.delete(np.array(atom), np.array([properties[0], properties[1]]), 0)



        v0 = arr - np.array([cx[0], cx[1], cx[2]])

        v1 = arr - np.array([cx[3], cx[4], cx[5]])



        dist0 = np.sqrt(np.power(v0, 2).sum(axis=1))

        dist1 = np.sqrt(np.power(v1, 2).sum(axis=1))



        M[index, :arr.shape[0], 0] = 1.0/dist0

        M[index, :arr.shape[0], 1] = 1.0/dist1

        M[index, :arr.shape[0], 2] = (atoms == "H")*1.0

        M[index, :arr.shape[0], 3] = (atoms != "H")*1.0

        M[index, :arr.shape[0], 4] = np.sum(v0*v1, axis=1)/(dist0*dist1)

        index += 1

        

    return M



M_train = get_mol(dev_df)

M_test = get_mol(test_df)
from keras.layers import *

from keras.models import Model

from keras.optimizers import Nadam



features = ["dist", "atom_count"]



def nn_block(input_layer, size, dropout_rate, activation):

    out_layer = Dense(size, activation=None)(input_layer)

    out_layer = BatchNormalization()(out_layer)

    out_layer = Activation(activation)(out_layer)

    out_layer = Dropout(dropout_rate)(out_layer)

    return out_layer



def cnn_block(input_layer, size, dropout_rate, activation):

    out_layer = Conv1D(size, 1, activation=None)(input_layer)

    out_layer = BatchNormalization()(out_layer)

    out_layer = Activation(activation)(out_layer)

    out_layer = Dropout(dropout_rate)(out_layer)

    return out_layer





def get_model():

    dense_input = Input(shape=(len(features) + 2,))



    type_input = Input(shape=(1,))

    type_emb = Flatten()(Embedding(dev_df["le_type"].max() + 1, 5)(type_input))

    

    mol_input = Input(shape=(M_train.shape[1], M_train.shape[2]))

    mol_layer = cnn_block(mol_input, 200, 0.0, "relu")

    mol_layer = cnn_block(mol_layer, 100, 0.0, "relu")

    

    merged_input = BatchNormalization()(concatenate([dense_input, type_emb, 

                                                     GlobalMaxPooling1D()(mol_layer), GlobalAvgPool1D()(mol_layer)]))

    

    x1 = Dense(200, activation="relu")(merged_input)

    x2 = nn_block(x1, 20, 0.0, "sigmoid")

    

    mol_layer = concatenate([RepeatVector(30)(x2), mol_layer])

    mol_layer = cnn_block(mol_layer, 200, 0.0, "relu")

    mol_layer = cnn_block(mol_layer, 100, 0.0, "relu")

    merged_input = BatchNormalization()(concatenate([dense_input, type_emb, 

                                                     GlobalMaxPooling1D()(mol_layer), GlobalAvgPool1D()(mol_layer)]))

    hidden_layer = concatenate([Dense(600, activation="relu")(merged_input), x1])

    

    hidden_layer = nn_block(hidden_layer, 400, 0.0, "relu")

    hidden_layer = nn_block(hidden_layer, 200, 0.0, "relu")

    hidden_layer = nn_block(hidden_layer, 100, 0.0, "relu")

    

    out = Dense(1, activation="linear")(hidden_layer)

    

    model = Model(inputs=[dense_input, type_input, mol_input], outputs=out)

    return model
import gc



TARGET = "scalar_coupling_constant"

keep_list = [TARGET, "id", "molecule_name", "le_type", "sw", "type"] + features

dev_df.drop([col for col in dev_df.columns if col not in keep_list], 

            axis=1, inplace=True)

test_df.drop([col for col in test_df.columns if col not in keep_list], 

             axis=1, inplace=True)



del mol_df, structures



gc.collect()
from sklearn.model_selection import GroupKFold

from keras.callbacks import EarlyStopping, ModelCheckpoint



NUM_FOLDS = 4

BATCH_SIZE = 2**10

best_model_path = "best_model.h5"



def to_emb(s):

    return s.values.reshape(-1, 1)



def get_features(df):

    return np.concatenate([df[features].values, np.log(df[features].values)], axis=1)



kfold = GroupKFold(NUM_FOLDS)

oof_preds = np.zeros(dev_df.shape[0])

test_preds = np.zeros(test_df.shape[0])



for train_index, val_index in kfold.split(dev_df[TARGET].values, dev_df[TARGET].values, dev_df["molecule_name"].values):

    

    train_df = dev_df[features + ["sw", "le_type", TARGET]].iloc[train_index]

    val_df = dev_df[features + ["sw", "le_type", TARGET]].iloc[val_index]

    

    train_df["sw"] *= train_df.shape[0]/train_df["sw"].sum()

    val_df["sw"] *= val_df.shape[0]/val_df["sw"].sum()

    

    model = get_model()

    #early_stopping = EarlyStopping(monitor="val_loss", patience=10, verbose=1)

    model_checkpoint = ModelCheckpoint(best_model_path, save_best_only=True, save_weights_only=True, verbose=1)

    

    for epochs, lr in zip([3, 3, 3, 3, 6], [0.004, 0.004, 0.001, 0.001, 0.0004]):

        model.compile(loss="mean_absolute_error", optimizer=Nadam(lr=lr))

        model.fit([get_features(train_df), to_emb(train_df["le_type"]), M_train[train_index]], train_df[TARGET], 

                  sample_weight=train_df["sw"].values,

                  validation_data=([get_features(val_df), to_emb(val_df["le_type"]), M_train[val_index]], val_df[TARGET], val_df["sw"].values),

                  verbose=1, epochs=epochs, batch_size=BATCH_SIZE, callbacks=[model_checkpoint])

    

    model.load_weights(best_model_path)

    

    oof_preds[val_index] = model.predict([get_features(val_df), to_emb(val_df["le_type"]), M_train[val_index]], 

                                         verbose=1, batch_size=BATCH_SIZE).ravel()

    

    test_preds += model.predict([get_features(test_df), to_emb(test_df["le_type"]), M_test], 

                                         verbose=1, batch_size=BATCH_SIZE).ravel()/NUM_FOLDS
del M_train, M_test



gc.collect()
dev_df["oof_pred"] = oof_preds



def competition_metric(df, pred_col):

    df["error"] = np.abs(df[TARGET] - df[pred_col])

    return np.log(df.groupby("type")["error"].mean()).mean()



competition_metric(dev_df, "oof_pred")
test_df[TARGET] = test_preds

test_df.to_csv("submission.csv", columns=["id", TARGET], index=False)
test_df.head()