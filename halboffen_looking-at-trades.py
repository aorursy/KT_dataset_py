import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



with pd.HDFStore("../input/train.h5", "r") as hfdata:

    data = hfdata.get("train")

        

%matplotlib inline
ids = data[["id", "timestamp"]]

id_timestamp_ct = pd.crosstab(index=ids.id, columns=ids.timestamp)

id_timestamp_ct.insert(1813, 1813, 0)

id_timestamp_ct.insert(0, -1, 0)

id_timestamp_ct_diff = id_timestamp_ct.diff(axis=1).abs().fillna(0)

transaction_indexes = np.where(id_timestamp_ct_diff)



tmp = [(id_timestamp_ct_diff.index[x], x, y) for x,y in zip(transaction_indexes[0], transaction_indexes[1])]

trades = [[id, entry-1, exit-1] for ((id, _, entry), (_, _, exit)) in list(zip(tmp[0::2], tmp[1::2]))]

trades_df = pd.DataFrame(data=trades, columns=["id","entry","exit"])

trades_df = trades_df.sort_values(by=["entry", "exit"]).reset_index(drop=True)

trades_df.head(20)
print("Number of assets: {}".format(len(trades_df.id.unique())))

print("Number of trades: {}".format(len(trades_df.id)))
_ = trades_df[["entry","exit"]].plot(linestyle="none", marker=".")
trades_df[["id", "exit"]].groupby("exit").count().sort_values(by="id", ascending=False).head().rename(columns = {"id": "count"})
cnt_trades = trades_df[["id","entry"]].groupby("id").count()

cnt_trades[cnt_trades.entry > 1].sort_values(by="entry", ascending=False)