import pandas as pd 

import datetime as dt

import matplotlib as mpl
def astype_cat(dd, cols):

    for col in cols:

        if isinstance(col, tuple):

            col, idx1, idx2 = col

            for idx in range(idx1, idx2+1):

                full_col=col+str(idx)

                dd[full_col]=dd[full_col].astype("category")

        else:

            dd[col]=dd[col].astype("category")

            

dd=pd.read_csv("../input/train_transaction.csv")

astype_cat(dd, ["ProductCD", ("card", 1, 6), "addr1", "addr2", "P_emaildomain", "R_emaildomain", ("M", 1, 9)])



ddid=pd.read_csv("../input/train_identity.csv")

astype_cat(ddid, ["DeviceType", "DeviceInfo", ("id_", 12, 38)])



dd=dd.merge(ddid, "left", "TransactionID")



dd["datetime"]=(dd["TransactionDT"].apply(lambda x:dt.timedelta(seconds=x)+pd.Timestamp("2017-11-30")))



del ddid



dd.head()
show_cols=["datetime", "TransactionAmt", "card1", "card2", "card3", "id_31", "DeviceType", "DeviceInfo"]

sort_cols=["TransactionAmt"]



fraud=dd.query("isFraud==1")



cat_cols=dd.dtypes.loc[lambda x:x=="category"].index



colors=mpl.cm.tab20.colors

n_colors=len(colors)

color_spec={i:'background: rgb({})'.format(",".join(str(int(255*colval)) for colval in color)) for i, color in enumerate(colors)}



def color_hash(value):

    color_idx = hash(value) % n_colors

    return color_spec[color_idx]



all_show_cols=set(sort_cols)|set(cat_cols)



fraud.sort_values(sort_cols + ["TransactionDT"])[show_cols + sorted(all_show_cols-set(show_cols))].head(100).style.applymap(color_hash)