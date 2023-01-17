import pandas as pd

import numpy as np

import os

import random

import torch

import torch.nn as nn

import torch.utils.data

import torch.optim as optim

from tqdm.notebook import tqdm

import sklearn.model_selection

import torch.optim.lr_scheduler as lr

import sys

sys.path.append("../input/osic-vae")

from preprocess import *
#img = get_img_3d("ID00419637202311204720264", file_path="../input/osic-pulmonary-fibrosis-progression/test/")
#features = get_img_features(img, latent_features=100, model_path="../input/osic-vae/full_chest_model.pth")
train = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")

train["FVC"] = train.groupby(["Patient", "Weeks"])["FVC"].transform("mean")

train["Percent"] = train.groupby(["Patient", "Weeks"])["Percent"].transform("mean")

train.drop_duplicates(inplace=True, ignore_index=True)



chunk = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")



test = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/sample_submission.csv")

test["Patient"] = test["Patient_Week"].apply(lambda x: x.split("_")[0])

test["Weeks"] = test["Patient_Week"].apply(lambda x: int(x.split("_")[1]))

test = test[["Patient", "Weeks", "Confidence", "Patient_Week"]]

test = test.merge(chunk.drop("Weeks", axis=1), on="Patient")



train["WHERE"] = "train"

chunk["WHERE"] = "val"

test["WHERE"] = "test"

df = train.append([chunk, test])



df["Male"] = df["Sex"].apply(lambda x: int(x == "Male"))

df["Female"] = df["Sex"].apply(lambda x: int(x == "Female"))



df["ExSmoker"] = df["SmokingStatus"].apply(lambda x: int(x == "Ex-smoker"))

df["NeverSmoked"] = df["SmokingStatus"].apply(lambda x: int(x == "Never smoked"))

df["CurrentlySmokes"] = df["SmokingStatus"].apply(lambda x: int(x == "Currently smokes"))





df["MinWeeks"] = df["Weeks"]

df.loc[df.WHERE == "test", "MinWeeks"] = np.nan

df["MinWeeks"] = df.groupby("Patient")["MinWeeks"].transform("min")





df = df.merge(

    df.loc[df["Weeks"] == df["MinWeeks"], ["Patient", "FVC", "Percent", "WHERE"]].rename({"FVC": "StartFVC", "Percent": "StartPercent"}, axis=1),

    on=["Patient", "WHERE"]

)



df["Weeks"] = df["Weeks"] - df["MinWeeks"]



df["Weeks"] = (df["Weeks"] - df["Weeks"].mean()) / df["Weeks"].std()

df["Age"] = (df["Age"] - df["Age"].mean()) / df["Age"].std()

df["StartFVC"] = (df["StartFVC"] - df["StartFVC"].mean()) / df["StartFVC"].std()

df["StartPercent"] = (df["StartPercent"] - df["StartPercent"].mean()) / df["StartPercent"].std()



train = df.loc[df.WHERE == "train"]

chunk = df.loc[df.WHERE == "val"]

test = df.loc[df.WHERE == "test"]



del df
features = [x for x in test.columns if x not in ["Patient_Week", "Patient", "FVC", "Percent", "Confidence", "Sex", "SmokingStatus", "WHERE", "MinWeeks"]]

target = "FVC"
train[features]
test[features]
def seed_everything(seed):

    random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False
class MyDataset(torch.utils.data.Dataset):

    def __init__(self, df, mode="train"):

        self.df = df

        self.mode = mode

        self.cache = {}



    def __len__(self):

        return len(self.df)



    def __getitem__(self, idx):

        patient_id = self.df.loc[idx, "Patient"]



        if self.mode == "train":

            tab_features = self.df.loc[idx, features].values

            img_features = np.load("../input/osic-train-image-features/{}.npy".format(patient_id))



            return np.concatenate((tab_features, img_features)).astype(np.float32), self.df.loc[idx, target].astype(np.float32).reshape(1)

        elif self.mode == "test":

            tab_features = self.df.loc[idx, features].values



            if patient_id in self.cache:

                img_features = self.cache[patient_id]

            else:

                img = get_img_3d(patient_id, file_path="../input/osic-pulmonary-fibrosis-progression/test/")

                img_features = get_img_features(img, latent_features=100, model_path="../input/osic-vae/full_chest_model.pth")

                

                self.cache[patient_id] = img_features



            return np.concatenate((tab_features, img_features)).astype(np.float32)

        else:

            return None
class Swish(nn.Module):

    def forward(self, x):

        return x * torch.sigmoid(x)
class MyModel(nn.Module):

    def __init__(self, in_features, out_features, args):

        super().__init__()

        

        def get_activation(args):

            if args["activation"] == "ReLU":

                return nn.ReLU(inplace=True)

            elif args["activation"] == "LeakyReLU":

                return nn.LeakyReLU(inplace=True)

            elif args["activation"] == "Tanh":

                return nn.Tanh()

            elif args["activation"] == "Swish":

                return Swish()

            else:

                return None



        self.my_model = nn.Sequential(

            nn.Linear(in_features, args["mid_features"]),

            get_activation(args),

            nn.Dropout(p=args["dropout"]),

            

            nn.Linear(args["mid_features"], args["mid_features"]),

            get_activation(args),

            nn.Dropout(p=args["dropout"]),

            

            nn.Linear(args["mid_features"], out_features),

            get_activation(args)

        )



    def forward(self, x):

        return self.my_model(x)
def quantile_loss(preds, targets, quantiles):

    errors = targets - preds

    losses = [torch.max((q - 1) * errors[:, i], q * errors[:, i]).unsqueeze(1) for i, q in enumerate(quantiles)]



    return torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
def quantile_metric(preds, targets):

    sigma = preds[:, 2:3] - preds[:, 0:1]

    sigma[sigma < 70] = 70



    delta = (targets - preds[:, 1:2]).abs()

    delta[delta > 1000] = 1000



    return (-np.sqrt(2) * delta / sigma - torch.log(np.sqrt(2) * sigma)).mean()
def get_lr(optimizer):

    for p in optimizer.param_groups:

        return p["lr"]
def main(args, fold, train_df, dev_df):

    seed_everything(args["seed"])



    data_loaders = {

        "train": torch.utils.data.DataLoader(MyDataset(train_df), batch_size=args["batch_size"], shuffle=True, num_workers=3, pin_memory=True),

        "dev": torch.utils.data.DataLoader(MyDataset(dev_df), batch_size=args["batch_size"], shuffle=True, num_workers=3, pin_memory=True)

    }



    model = MyModel(len(features) + 200, 3, args).to(device=args["device"], dtype=args["dtype"])

    optimizer = optim.AdamW(model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"])

    scheduler = lr.StepLR(optimizer, step_size=args["step_size"], gamma=args["gamma"])



    best_metric = -10**10



    pbar = tqdm(range(args["epochs"]), disable=(not args["progress_bar"]))



    for epoch in pbar:

        stats = {

            "train": {

                "loss": 0,

                "metric": 0

            },

            "dev": {

                "loss": 0,

                "metric": 0

            }

        }



        model = model.train()



        for X, y in data_loaders["train"]:

            X = X.to(device=args["device"], dtype=args["dtype"])

            y = y.to(device=args["device"], dtype=args["dtype"])



            y_pred = model(X)

            loss = quantile_loss(y_pred, y, args["quantiles"])



            optimizer.zero_grad()

            loss.backward()

            optimizer.step()



        model = model.eval()



        with torch.no_grad():

            for mode in ["train", "dev"]:

                for X, y in data_loaders[mode]:

                    X = X.to(device=args["device"], dtype=args["dtype"])

                    y = y.to(device=args["device"], dtype=args["dtype"])



                    y_pred = model(X)

                    loss = quantile_loss(y_pred, y, args["quantiles"])

                    metric = quantile_metric(y_pred, y)



                    stats[mode]["loss"] += loss.item() / len(data_loaders[mode])

                    stats[mode]["metric"] += metric.item() / len(data_loaders[mode])



        if stats["dev"]["metric"] > best_metric:

            best_metric = stats["dev"]["metric"]



            if args["save"]:

                torch.save(model.state_dict(), "model_fold_{}.pth".format(fold))



        scheduler.step()



        pbar.set_description("best: {:.4f} current: {:.4f} lr: {}".format(best_metric, stats["dev"]["metric"], get_lr(optimizer)))



        if get_lr(optimizer) < args["min_lr"]:

            break



    return -best_metric
def main_folds(args):

    kfold = sklearn.model_selection.GroupKFold(n_splits=args["n_splits"])



    avg_metric = 0



    for fold, (train_idx, dev_idx) in enumerate(kfold.split(train, groups=train["Patient"])):

        train_df = train.loc[train_idx].reset_index(drop=True)

        dev_df = train.loc[dev_idx].reset_index(drop=True)



        avg_metric += main(args, fold, train_df, dev_df) / args["n_splits"]

        #break



    print(avg_metric, args)



    return avg_metric
from hyperopt import hp, fmin, tpe



space = {

    "seed": 42,

    "n_splits": 5,

    "batch_size": hp.choice("batch_size", [32, 64, 128, 256, 512, 1024, 2048]),

    "epochs": 25,

    "lr": hp.loguniform("lr", -10, 0),

    "weight_decay": hp.loguniform("weight_decay", -5, 0),

    "quantiles": [0.2, 0.5, 0.8],

    "mid_features": hp.choice("mid_features", [128, 256, 512, 1024, 2048]),

    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),

    "dtype": torch.float,

    "step_size": hp.randint("step_size", 2, 20),

    "gamma": hp.loguniform("gamma", -5, 0),

    "min_lr": 1e-8,

    "dropout": 0.5,

    "activation": hp.choice("activation", ["ReLU", "LeakyReLU", "Tanh", "Swish"]),

    "progress_bar": False,

    "save": False

}



#best = fmin(main_folds, space, algo=tpe.suggest, max_evals=100)
ARGS = {

    "seed": 42,

    "n_splits": 5,

    "batch_size": 32,

    "epochs": 100,

    "lr": 0.0027214256541696524,

    "weight_decay": 0.5435493827830923,

    "quantiles": [0.2, 0.5, 0.8],

    "mid_features": 1024,

    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),

    "dtype": torch.float,

    "step_size": 9,

    "gamma": 0.6642775979489709,

    "min_lr": 1e-8,

    "dropout": 0.5,

    "activation": "LeakyReLU",

    "progress_bar": True,

    "save": True

}
#main_folds(ARGS)
def inference(args):

    data_loaders = {

        "test": torch.utils.data.DataLoader(MyDataset(test.reset_index(drop=True), mode="test"), batch_size=args["batch_size"], shuffle=False),

    }



    preds = {

        "FVC": np.zeros((len(test), 1)),

        "Confidence": np.zeros((len(test), 1))

    }



    for fold in range(args["n_splits"]):

        fold_preds = {

            "FVC": [],

            "Confidence": []

        }



        model = MyModel(len(features) + 200, 3, args).to(device=args["device"], dtype=args["dtype"])

        #model.load_state_dict(torch.load("model_fold_{}.pth".format(fold)))

        model.load_state_dict(torch.load("../input/osic-trained/model_fold_{}.pth".format(fold)))

        model = model.eval()



        with torch.no_grad():

            for i, X in enumerate(tqdm(data_loaders["test"], disable=(not args["progress_bar"]))):

                X = X.to(device=args["device"], dtype=args["dtype"])



                y_pred = model(X).cpu().detach().numpy()



                fold_preds["FVC"].extend(y_pred[:, 1:2])

                fold_preds["Confidence"].extend(y_pred[:, 2:3] - y_pred[:, 0:1])



        preds["FVC"] += np.array(fold_preds["FVC"]) / args["n_splits"]

        preds["Confidence"] += np.array(fold_preds["Confidence"]) / args["n_splits"]



    return preds
preds = inference(ARGS)
test["FVC"] = preds["FVC"]

test["Confidence"] = preds["Confidence"]
test[["Patient_Week", "FVC", "Confidence"]].to_csv("submission.csv", index=False)