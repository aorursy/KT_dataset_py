import pandas as pd

from sklearn import model_selection

TRAIN_CSV = '../input/shopee-product-detection-student/train.csv'
df = pd.read_csv(TRAIN_CSV)

df["kfold"] = -1

df = df.sample(frac=1,random_state=33).reset_index(drop=True)

kf = model_selection.StratifiedKFold(n_splits=5)

for fold, (trn_, val_) in enumerate(kf.split(X=df, y=df.category.values)):

    print(len(trn_), len(val_))

    df.loc[val_, 'kfold'] = fold



df.head()
df.to_csv("train_folds.csv", index=False)
fold = 1

df_train = df[df.kfold != fold].reset_index(drop=True)

df_valid = df[df.kfold == fold].reset_index(drop=True)