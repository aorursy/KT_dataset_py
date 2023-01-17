import numpy as np 

import pandas as pd 

from sklearn.ensemble import GradientBoostingClassifier
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")

ntrain = train_df.shape[0]

ntest = test_df.shape[0]
trues = train_df.loc[train_df['target'] == 1]

falses = train_df.loc[train_df['target'] != 1].sample(frac=1)[:len(trues)]

data = pd.concat([trues, falses], ignore_index=True).sample(frac=1)



data.shape
cols = [i for i in data.columns.values if i not in ["ID_code","target"]]

target = data["target"].values
def augment(x,y,t=2):

    xs,xn = [],[]

    for i in range(t):

        mask = y>0

        x1 = x[mask].copy()

        ids = np.arange(x1.shape[0])

        for c in range(x1.shape[1]):

            np.random.shuffle(ids)

            x1[:,c] = x1[ids][:,c]

        xs.append(x1)



    for i in range(t//2):

        mask = y==0

        x1 = x[mask].copy()

        ids = np.arange(x1.shape[0])

        for c in range(x1.shape[1]):

            np.random.shuffle(ids)

            x1[:,c] = x1[ids][:,c]

        xn.append(x1)



    xs = np.vstack(xs)

    xn = np.vstack(xn)

    ys = np.ones(xs.shape[0])

    yn = np.zeros(xn.shape[0])

    x = np.vstack([x,xs,xn])

    y = np.concatenate([y,ys,yn])

    return x,y
X_t, y_t = augment(data[cols].values, target)

X_t = pd.DataFrame(X_t)

X_t = X_t.add_prefix('var_')
GBC = GradientBoostingClassifier(learning_rate=0.05,min_samples_split=800,max_depth = 9,n_estimators=400,min_samples_leaf=80,

                                                        random_state=1,max_features = 11 ,verbose=50,subsample=0.75)
%%time

GBC.fit(X_t,y_t)
GBC.train_score_
pred = np.zeros(ntest)

sub = pd.DataFrame({"ID_code":test_df["ID_code"].values})

pred = GBC.predict_proba(test_df[cols])[:,1]

sub["target"] = pred

sub.to_csv("submission5.csv",index=False)