import pandas as pd

from sklearn import model_selection

TRAIN_CSV = "../input/student-shopee-code-league-sentiment-analysis/train.csv"
df = pd.read_csv(TRAIN_CSV)

df.drop_duplicates(subset='review',inplace=True)
df["kfold"] = -1

df = df.sample(frac=1,random_state=33).reset_index(drop=True)

kf = model_selection.StratifiedKFold(n_splits=5)

for fold, (trn_, val_) in enumerate(kf.split(X=df, y=df.rating.values)):

    print(len(trn_), len(val_))

    df.loc[val_, 'kfold'] = fold



df.head()
df.to_csv('folds.csv',index=False)
fold = 1

df_train = df[df.kfold != fold].reset_index(drop=True)

df_valid = df[df.kfold == fold].reset_index(drop=True)