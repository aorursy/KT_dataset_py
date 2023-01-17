import pandas as pd
from sklearn.linear_model import LinearRegression
df_train = pd.read_csv("../input/train_set.csv")
df_test = pd.read_csv("../input/test_set.csv")
df_train.columns
pd.concat([df_train, df_test], ignore_index=True, sort=True)["GBA"].mean()
cols = ['BATHRM', 'ROOMS', 'GBA']
df_x = df_train[cols].copy()
df_y = df_train['PRICE'].copy()
df_x['GBA'] = df_x['GBA'].fillna(1725.786303972366)
cols = ['BATHRM', 'ROOMS', 'GBA']
df_test_x = df_test[cols].copy()
df_test_id = df_test['Id'].copy()
df_test_x['GBA'] = df_test_x['GBA'].fillna(1725.786303972366)
def preprocess(train_flg):
    # train set の場合の処理
    if train_flg:
        df=pd.read_csv('../input/train_set.csv')
        df_y = df['PRICE']
    else:
        df=pd.read_csv('../input/test_set.csv')
    # 使う変数を定義
    cols = ['BATHRM', 'ROOMS', 'GBA']
    df_x = df.loc[:,cols]
    #GBAの面積を平均値で埋める
    df_x['GBA'] = df_x['GBA'].fillna(1725.786303972366)
    # train set の場合のoutput
    if train_flg:
        return df_x,df_y
    # test set の場合のoutput、kaggleへ提出するファイルの規定により、idも一緒に出力する
    else:
        return df_x,df['Id']
clf = LinearRegression()
df_x,df_y = preprocess(train_flg=True)
clf.fit(df_x, df_y)
df_test_x,df_test_id = preprocess(train_flg=False)

y_pred = clf.predict(df_test_x)

pred_df = pd.DataFrame(y_pred, index=df_test_id, columns=["PRICE"])
pred_df.to_csv('./output.csv', header=True, index_label='Id')