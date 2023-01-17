import os

import random



import numpy as np

import pandas as pd

import torch

import torch.nn as nn

import torch.optim as optim

import torch.optim.lr_scheduler as lr

from sklearn.model_selection import KFold, GroupKFold

from torch.utils.data import DataLoader, Dataset

from tqdm.notebook import tqdm

import itertools

from sklearn.cluster import KMeans
def seed_everything(seed):

    random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False
token2int = {x: i for i, x in enumerate("().AGUCSMIBHEX ")}
train = pd.read_json("../input/stanford-covid-vaccine/train.json", lines=True)

test = pd.read_json("../input/stanford-covid-vaccine/test.json", lines=True)
temp = np.array(train["sequence"].apply(lambda seq: [token2int[x] for x in seq]).values.tolist())

kmeans = KMeans(n_clusters=200, random_state=42).fit(temp)

train["cluster"] = kmeans.labels_
def read_bpps_sum(df):

    bpps_arr = []

    for mol_id in df.id.to_list():

        bpps_arr.append(np.load(f"../input/stanford-covid-vaccine/bpps/{mol_id}.npy").max(axis=1))

    return bpps_arr





def read_bpps_max(df):

    bpps_arr = []

    for mol_id in df.id.to_list():

        bpps_arr.append(np.load(f"../input/stanford-covid-vaccine/bpps/{mol_id}.npy").sum(axis=1))

    return bpps_arr





def read_bpps_nb(df):

    bpps_nb_mean = 0.077522

    bpps_nb_std = 0.08914

    bpps_arr = []

    for mol_id in df.id.to_list():

        bpps = np.load(f"../input/stanford-covid-vaccine/bpps/{mol_id}.npy")

        bpps_nb = (bpps > 0).sum(axis=0) / bpps.shape[0]

        bpps_nb = (bpps_nb - bpps_nb_mean) / bpps_nb_std

        bpps_arr.append(bpps_nb)

    return bpps_arr





train["bpps_sum"] = read_bpps_sum(train)

train["bpps_max"] = read_bpps_max(train)

train["bpps_nb"] = read_bpps_nb(train)



test["bpps_sum"] = read_bpps_sum(test)

test["bpps_max"] = read_bpps_max(test)

test["bpps_nb"] = read_bpps_nb(test)
targets = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
train["seq_masked_1"] = train.apply(lambda x: "".join([y if x.structure[i] == "." else " " for i, y in enumerate(x.sequence)]), axis=1)

train["seq_masked_2"] = train.apply(lambda x: "".join([y if x.structure[i] == "(" else " " for i, y in enumerate(x.sequence)]), axis=1)

train["seq_masked_3"] = train.apply(lambda x: "".join([y if x.structure[i] == ")" else " " for i, y in enumerate(x.sequence)]), axis=1)



test["seq_masked_1"] = test.apply(lambda x: "".join([y if x.structure[i] == "." else " " for i, y in enumerate(x.sequence)]), axis=1)

test["seq_masked_2"] = test.apply(lambda x: "".join([y if x.structure[i] == "(" else " " for i, y in enumerate(x.sequence)]), axis=1)

test["seq_masked_3"] = test.apply(lambda x: "".join([y if x.structure[i] == ")" else " " for i, y in enumerate(x.sequence)]), axis=1)
class MyDataset(Dataset):

    def __init__(self, df, mode="train"):

        self.df = df.reset_index(drop=True)

        self.mode = mode



    def __len__(self):

        return len(self.df)



    def __getitem__(self, idx):

        x1 = np.array(self.df.loc[idx, ["sequence", "structure", "predicted_loop_type", "seq_masked_1", "seq_masked_2", "seq_masked_3"]].apply(lambda seq: [token2int[x] for x in seq]).values.tolist()).transpose(1, 0)

        x2 = np.array(self.df.loc[idx, ["bpps_sum", "bpps_max", "bpps_nb"]].values.tolist()).transpose(1, 0)



        if self.mode == "train":

            return x1, x2, np.array(self.df.loc[idx, targets].values.tolist()).transpose(1, 0)

        elif self.mode == "test":

            return x1, x2

        else:

            return None
class MyModel(nn.Module):

    def __init__(self, args):

        super().__init__()



        self.embedding = nn.Embedding(num_embeddings=args["num_embeddings"], embedding_dim=args["embedding_dim"])

        self.lstm = nn.LSTM(input_size=args["in_features"] * args["embedding_dim"] + 3, hidden_size=args["lstm_hidden"], num_layers=args["lstm_layers"], bias=True, batch_first=True, dropout=args["lstm_dropout"], bidirectional=True)

        self.fc = nn.Linear(2 * args["lstm_hidden"], args["out_features"])



    def forward(self, x1, x2):

        x1 = self.embedding(x1.long()).view(x1.shape[0], x1.shape[1], -1)

        x = torch.cat((x1, x2), dim=2)

        x = self.lstm(x)[0][:, :x1.shape[1]]

        x = self.fc(x)



        return x



    def count_parameters(self):

        return sum(p.numel() for p in self.parameters() if p.requires_grad)
def get_lr(optimizer):

    for p in optimizer.param_groups:

        return p["lr"]
def MCRMSE(y_pred, y_true):

    colwise_mse = torch.mean((y_pred - y_true)**2, axis=1)

    return torch.mean(torch.sqrt(colwise_mse))
TRAIN_MODE = False



args = {

    "batch_size": 228,

    "device": torch.device("cuda"),

    "dtype": torch.float,

    "embedding_dim": 41,

    "epochs": 100,

    "factor": 0.40876664932540246,

    "folds": 5,

    "in_features": 6,

    "lr": 0.000514670300651814,

    "lstm_dropout": 0.3,

    "lstm_hidden": 765,

    "lstm_layers": 2,

    "min_lr": 1e-8,

    "num_embeddings": len(token2int),

    "out_features": 5,

    "patience": 10,

    "seed": 42,

    "weight_decay": 0.23884714200145296,

}
if TRAIN_MODE:

    seed_everything(args["seed"])



    kfold = GroupKFold(n_splits=args["folds"])



    for fold, (train_idx, dev_idx) in enumerate(kfold.split(train, groups=train["cluster"])):

        train_loader = DataLoader(MyDataset(train.iloc[train_idx]), batch_size=args["batch_size"], shuffle=True, num_workers=3, pin_memory=True)

        dev_loader = DataLoader(MyDataset(train.iloc[dev_idx]), batch_size=args["batch_size"], shuffle=False, num_workers=3, pin_memory=True)



        model = MyModel(args).to(device=args["device"], dtype=args["dtype"])



        optimizer = optim.AdamW(model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"])

        scheduler = lr.ReduceLROnPlateau(optimizer, mode="min", factor=args["factor"], patience=args["patience"])



        pbar = tqdm(range(args["epochs"]))



        best_loss = np.inf



        for epoch in pbar:

            model.train()



            for x1, x2, y in train_loader:

                x1 = x1.to(device=args["device"], dtype=torch.long)

                x2 = x2.to(device=args["device"], dtype=args["dtype"])

                y = y.to(device=args["device"], dtype=args["dtype"])



                y_pred = model(x1, x2)[:, :y.shape[1]]

                loss = MCRMSE(y_pred, y)



                optimizer.zero_grad()

                loss.backward()

                optimizer.step()



            model.eval()



            train_loss = 0

            dev_loss = 0



            with torch.no_grad():

                for batch, (x1, x2, y) in enumerate(train_loader):

                    x1 = x1.to(device=args["device"], dtype=torch.long)

                    x2 = x2.to(device=args["device"], dtype=args["dtype"])

                    y = y.to(device=args["device"], dtype=args["dtype"])



                    y_pred = model(x1, x2)[:, :y.shape[1]]

                    loss = MCRMSE(y_pred, y)



                    train_loss = (batch * train_loss + loss.item()) / (batch + 1)



                for batch, (x1, x2, y) in enumerate(dev_loader):

                    x1 = x1.to(device=args["device"], dtype=torch.long)

                    x2 = x2.to(device=args["device"], dtype=args["dtype"])

                    y = y.to(device=args["device"], dtype=args["dtype"])



                    y_pred = model(x1, x2)[:, :y.shape[1]]

                    loss = MCRMSE(y_pred, y)



                    dev_loss = (batch * dev_loss + loss.item()) / (batch + 1)



            scheduler.step(dev_loss)



            pbar.set_description("train: {:.5f} dev: {:.5f}".format(train_loss, dev_loss))



            if dev_loss < best_loss:

                best_loss = dev_loss

                torch.save(model.state_dict(), "model_fold_{}.pth".format(fold))



            if get_lr(optimizer) < args["min_lr"]:

                break



        print(best_loss)
public_test = test[test["seq_length"] == 107].copy()

private_test = test[test["seq_length"] == 130].copy()
public_preds = np.zeros((len(public_test), 107, len(targets)))

private_preds = np.zeros((len(private_test), 130, len(targets)))



for fold in range(args["folds"]):

    model = MyModel(args).to(device=args["device"], dtype=args["dtype"])

    if TRAIN_MODE:

        model.load_state_dict(torch.load("model_fold_{}.pth".format(fold)))

    else:

        model.load_state_dict(torch.load("../input/openvaccine-trained/model_fold_{}.pth".format(fold)))

    model = model.eval()



    public_test_loader = DataLoader(MyDataset(public_test, mode="test"), batch_size=args["batch_size"], shuffle=False, num_workers=3, pin_memory=True)

    private_test_loader = DataLoader(MyDataset(private_test, mode="test"), batch_size=args["batch_size"], shuffle=False, num_workers=3, pin_memory=True)



    with torch.no_grad():

        public_fold_preds = []

        

        for batch, (x1, x2) in enumerate(public_test_loader):

            x1 = x1.to(device=args["device"], dtype=torch.long)

            x2 = x2.to(device=args["device"], dtype=torch.long)

            y_pred = model(x1, x2).cpu().detach().numpy()

            public_fold_preds.extend(y_pred)



        public_preds += np.array(public_fold_preds) / args["folds"]



        private_fold_preds = []



        for batch, (x1, x2) in enumerate(private_test_loader):

            x1 = x1.to(device=args["device"], dtype=torch.long)

            x2 = x2.to(device=args["device"], dtype=torch.long)

            y_pred = model(x1, x2).cpu().detach().numpy()

            private_fold_preds.extend(y_pred)



        private_preds += np.array(private_fold_preds) / args["folds"]
preds = []



for df, p in [(public_test, public_preds), (private_test, private_preds)]:

    for i, uid in enumerate(df.id):

        single_pred = p[i]



        single_df = pd.DataFrame(single_pred, columns=targets)

        single_df["id_seqpos"] = [f"{uid}_{x}" for x in range(single_df.shape[0])]



        preds.append(single_df)



preds = pd.concat(preds)
preds[["id_seqpos"] + targets].to_csv("submission.csv", index=False)