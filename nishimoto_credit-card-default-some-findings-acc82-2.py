import pandas as pd

import matplotlib.pylab as plt

from tqdm import tqdm_notebook as tqdm
import pandas as pd

df_train = pd.read_csv("../input/UCI_Credit_Card.csv", index_col=0)

nohist = "BILL_AMT1 == 0 and BILL_AMT2 == 0 and BILL_AMT3 == 0 and BILL_AMT4 == 0 and BILL_AMT5 == 0 and BILL_AMT6 == 0 and PAY_AMT1 == 0 and PAY_AMT2 == 0 and PAY_AMT3 == 0 and PAY_AMT4 == 0 and PAY_AMT5 == 0 and PAY_AMT6 == 0"

nouse = "PAY_0 == -2 and PAY_2 == -2 and PAY_3 == -2 and PAY_4 == -2 and PAY_5 == -2 and PAY_6 == -2"



df_train_nohist = df_train.query(nohist)

df_train_nouse = df_train_nohist.query(nouse)



print("repay rate:", df_train_nouse["default.payment.next.month"].sum() * 100 / df_train_nouse.shape[0], "%")



df_train__2 = df_train.query("PAY_0 == -2")

print("repay rate:", df_train__2["default.payment.next.month"].sum() * 100 / df_train__2.shape[0], "%")
df = pd.read_csv("../input/UCI_Credit_Card.csv", index_col=0)

print(df.loc[[7133, 29828], :])



plt.subplots(figsize=(16, 3))

plt.scatter(df.index, df["AGE"])

plt.xlabel("ID")

plt.ylabel("AGE")
import pandas as pd

from tqdm import tqdm_notebook as tqdm



df = pd.read_csv("../input/UCI_Credit_Card.csv", index_col=0)

# df_train.drop("y", axis=1, inplace=True)



query_str1 = "LIMIT_BAL == @LIMIT_BAL and SEX == @SEX and EDUCATION== @EDUCATION and MARRIAGE == @MARRIAGE and AGE == @AGE"

query_str2 = "PAY_0 == @PAY_2 and PAY_2 == @PAY_3 and PAY_3 == @PAY_4 and PAY_4 == @PAY_5 and PAY_5 == @PAY_6"

query_str3 = "BILL_AMT1 == @BILL_AMT2 and BILL_AMT2 == @BILL_AMT3 and BILL_AMT3 == @BILL_AMT4 and BILL_AMT4 == @BILL_AMT5 and  BILL_AMT5 == @BILL_AMT6"

query_str4 = "PAY_AMT1 == @PAY_AMT2 and PAY_AMT2 == @PAY_AMT3 and PAY_AMT3 == @PAY_AMT4 and PAY_AMT4 == @PAY_AMT5 and  PAY_AMT5 == @PAY_AMT6"



leak_id = []

leak_val = []

for ind in tqdm(df.index):

    LIMIT_BAL = df.loc[ind, "LIMIT_BAL"]

    SEX = df.loc[ind, "SEX"]

    EDUCATION = df.loc[ind, "EDUCATION"]

    MARRIAGE = df.loc[ind, "MARRIAGE"]

    AGE = df.loc[ind, "AGE"]

    PAY_0 = df.loc[ind, "PAY_0"]

    PAY_2 = df.loc[ind, "PAY_2"]

    PAY_3 = df.loc[ind, "PAY_3"]

    PAY_4 = df.loc[ind, "PAY_4"]

    PAY_5 = df.loc[ind, "PAY_5"]

    PAY_6 = df.loc[ind, "PAY_6"]

    BILL_AMT1 = df.loc[ind, "BILL_AMT1"]

    BILL_AMT2 = df.loc[ind, "BILL_AMT2"]

    BILL_AMT3 = df.loc[ind, "BILL_AMT3"]

    BILL_AMT4 = df.loc[ind, "BILL_AMT4"]

    BILL_AMT5 = df.loc[ind, "BILL_AMT5"]

    BILL_AMT6 = df.loc[ind, "BILL_AMT6"]

    PAY_AMT1 = df.loc[ind, "PAY_AMT1"]

    PAY_AMT2 = df.loc[ind, "PAY_AMT2"]

    PAY_AMT3 = df.loc[ind, "PAY_AMT3"]

    PAY_AMT4 = df.loc[ind, "PAY_AMT4"]

    PAY_AMT5 = df.loc[ind, "PAY_AMT5"]

    PAY_AMT6 = df.loc[ind, "PAY_AMT6"]



    # skip new customer

    if PAY_0+PAY_2+PAY_3+PAY_4+PAY_5+PAY_6 == -12 and BILL_AMT1+BILL_AMT2+BILL_AMT3+BILL_AMT4+BILL_AMT5+BILL_AMT6+PAY_AMT1+PAY_AMT2+PAY_AMT3+PAY_AMT4+PAY_AMT5+PAY_AMT6 == 0:

        continue

    if PAY_0+PAY_2+PAY_3+PAY_4+PAY_5+PAY_6 == -9 and BILL_AMT1+BILL_AMT2+BILL_AMT3+BILL_AMT4+BILL_AMT5+BILL_AMT6+PAY_AMT1+PAY_AMT2+PAY_AMT3+PAY_AMT4+PAY_AMT5+PAY_AMT6 == 0:

        continue



    queried1 = df.query(query_str1)

    if len(queried1) == 0:

        continue



    queried2 = queried1.query(query_str2)

    if len(queried2) == 0:

        continue



    queried3 = queried2.query(query_str3)

    if len(queried3) == 0:

        continue



    queried4 = queried3.query(query_str4)

    if len(queried4) != 1:

        continue



    if ind != queried4.index[0]:

        leak_id.append(queried4.index[0])

        leak_id.append(ind)



df.loc[leak_id, :].tail(10)
import warnings

import numpy as np

import pandas as pd

import xgboost as xgb



from tqdm import tqdm

from collections import defaultdict



from sklearn.metrics import accuracy_score

from sklearn.model_selection import StratifiedKFold

from sklearn.neighbors import KNeighborsClassifier



warnings.filterwarnings('ignore')

params = {"seed": 0}





def preprocess_addtrend(df):

    a1s = []

    a2s = []

    a3s = []

    b1s = []

    b2s = []

    b3s = []

    for ind in tqdm(df.index):

        x = [1, 2, 3, 4, 5, 6]

        y1 = df.loc[ind, ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]]

        y2 = df.loc[ind, ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]]

        y3 = df.loc[ind, ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]]



        a1, b1 = np.polyfit(x, y1, 1)

        a2, b2 = np.polyfit(x, y2, 1)

        a3, b3 = np.polyfit(x, y3, 1)



        a1s.append(a1); a2s.append(a2); a3s.append(a3)

        b1s.append(b1); b2s.append(b2); b3s.append(b3)



    df["a1"] = a1s

    df["a2"] = a2s

    df["a3"] = a3s

    df["b1"] = b1s

    df["b2"] = b2s

    df["b3"] = b3s

    return df





def preprocess_df(df):

    df.drop(["a1", "b1", "b2", "a3", "b3"], axis=1, inplace=True)

    df.drop(["AGE"], axis=1, inplace=True)

    df["LIMIT_BAL/PAY_0"] = df["LIMIT_BAL"] / df["PAY_0"]

    df["PAY_0/PAY_2"] = df["PAY_0"] / df["PAY_2"]

    df["BILL_AMT1/BILL_AMT2"] = df["BILL_AMT1"] / df["BILL_AMT2"]

    df["LIMIT_BAL/BILL_AMT1"] = df["LIMIT_BAL"] / df["BILL_AMT1"]

    df["PAY_0/BILL_AMT1"] = df["PAY_0"] / df["BILL_AMT1"]



    # replace value(maybe means -2 & -1 same)

    for col in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]:

        df[col] = [-1 if val == -2 else val for val in df[col]]



    df["PAY_0/PAY_2"] = df["PAY_0/PAY_2"].fillna(0)

    df["BILL_AMT1/BILL_AMT2"] = df["BILL_AMT1/BILL_AMT2"].fillna(0)

    df["PAY_0/BILL_AMT1"] = df["PAY_0/BILL_AMT1"].fillna(0)

    return df





def preprocess_knn(df):

    """historyがほぼない人用の前処理関数"""

    # preprocess

    cols = ["LIMIT_BAL", "MARRIAGE", "AGE"]

    df = df[cols]



    # z_score

    # for col in cols:

    #     df[col] = (df[col] - df[col].mean()) / df[col].std()



    # min-max scaling

    for col in cols:

        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())



    return df





def main():

    # main

    df_train = pd.read_csv("../input/UCI_Credit_Card.csv", index_col=0)



    # preprocess 1st

    df_train = preprocess_addtrend(df_train)



    # preprocess 2nd

    train_y = df_train["default.payment.next.month"]

    train_x = df_train.drop("default.payment.next.month", axis=1)

    train_x = preprocess_df(train_x)



    # get nohist data

    nohist = "BILL_AMT1 == 0 and BILL_AMT2 == 0 and BILL_AMT3 == 0 and BILL_AMT4 == 0 and BILL_AMT5 == 0 and BILL_AMT6 == 0 and PAY_AMT1 == 0 and PAY_AMT2 == 0 and PAY_AMT3 == 0 and PAY_AMT4 == 0 and PAY_AMT5 == 0 and PAY_AMT6 == 0"



    # preprocess to knn

    train_x_selected = preprocess_knn(pd.read_csv("../input/UCI_Credit_Card.csv", index_col=0))



    # main_predict

    accuracies = []

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)



    i = 0

    for train_idx, test_idx in cv.split(train_x, train_y):    

        trn_x = train_x.iloc[train_idx, :]

        val_x = train_x.iloc[test_idx, :]



        trn_y = train_y.iloc[train_idx]

        val_y = train_y.iloc[test_idx]



        clf_xgb = xgb.XGBClassifier(seed=0)

        clf_xgb.fit(trn_x, trn_y)

        pred_y_xgb = clf_xgb.predict(val_x)

        result_df = pd.DataFrame({

            "pred": pred_y_xgb,

        }, index=val_y.keys())



        # predict nohist by knn

        trn_nohist_index = trn_x.query(nohist).index

        val_nohist_index = val_x.query(nohist).index



        trn_x_nohist = train_x_selected.loc[trn_nohist_index, :]

        val_x_nohist = train_x_selected.loc[val_nohist_index, :]



        trn_y_nohist = trn_y.loc[trn_nohist_index]

        val_y_nohist = val_y.loc[val_nohist_index]



        clf_knn = KNeighborsClassifier(n_neighbors=14) # parameter tuning by gridsearch

        clf_knn.fit(trn_x_nohist, trn_y_nohist)

        pred_y_knn = clf_knn.predict(val_x_nohist)

        result_df.loc[val_nohist_index, "pred"] = pred_y_knn



        accuracies.append(accuracy_score(val_y, result_df["pred"]))

    print(accuracies, np.mean(accuracies))



if __name__ == "__main__":

    main()