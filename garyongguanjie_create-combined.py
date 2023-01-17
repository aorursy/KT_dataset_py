import pandas as pd

from sklearn import model_selection
df_train = pd.read_csv("../input/student-shopee-code-league-sentiment-analysis/train.csv")
df_train2 = pd.read_csv("../input/test-labelled/test_labelled.csv")
df_train3 = pd.read_csv("../input/shopee-reviews/shopee_reviews.csv")
df_train3 = df_train3.rename({'label':'rating','text':'review'},axis=1)

df_train3 = df_train3.drop(df_train3[df_train3.rating == 'label'].index)

df_train3.astype({'rating': int}).dtypes
df_train = df_train.drop(['review_id'],axis=1)

df_train2 = df_train2.drop(['review_id'],axis=1)
df_combined = pd.concat([df_train,df_train2,df_train3],ignore_index=True)
df_combined = df_combined.sample(frac=1)
df_combined.drop_duplicates(subset='review',inplace=True)
df = df_combined.dropna()
df["kfold"] = -1

df = df.sample(frac=1,random_state=33).reset_index(drop=True)

kf = model_selection.StratifiedKFold(n_splits=5)

for fold, (trn_, val_) in enumerate(kf.split(X=df, y=df.rating.values.astype(int))):

    print(len(trn_), len(val_))

    df.loc[val_, 'kfold'] = fold



df.head()
df.to_csv('folds4.csv',index=False)