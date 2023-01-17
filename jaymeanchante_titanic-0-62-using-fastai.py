import os

import pandas as pd

from fastai.tabular import *
from pathlib import Path
path = Path("../input/")
print(os.listdir("../input"))
train = pd.read_csv("../input/train.csv")

train.head()
test = pd.read_csv("../input/test.csv")

test.head()
sub = pd.read_csv("../input/gender_submission.csv")

sub.head()
del train["PassengerId"]

del test["PassengerId"]
procs = [FillMissing, Categorify, Normalize]
valid_thresh = round(train.shape[0] * 0.1)
valid_idx = range(len(train) - valid_thresh, len(train))
dep_var = "Survived"

cat_names = ["Pclass", "Name", "Sex", "Ticket", "Cabin", "Embarked"]
data = TabularDataBunch.from_df(path, train, dep_var, valid_idx=valid_idx, procs=procs, cat_names=cat_names)

print(data.train_ds.cont_names)
(cat_x,cont_x),y = next(iter(data.train_dl))

for o in (cat_x, cont_x, y): print(to_np(o[:5]))
learn = tabular_learner(

    data,

    layers=[200,100],

    emb_szs={"Pclass": 3, "Name": 50, "Sex": 2, "Ticket": 25, "Cabin": 5, "Embarked": 3},

    metrics=Precision())
learn.fit_one_cycle(10, 1e-3)
preds = []

for i in range(len(test)):

    try:

        pred = learn.predict(test.iloc[i])

        preds.append(str(pred[0]))

    except:

        preds.append("0")
sub.loc[:, "Survived"] = preds
sub.loc[:, "Survived"] = sub["Survived"].astype(int)
sub.to_csv("sub.csv", index=False)