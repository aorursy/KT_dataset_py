import pandas as pd
import IPython





def display(*dfs, head=True):

    for df in dfs:

        IPython.display.display(df.head() if head else df)
INPUT_DIR = "/kaggle/input/m5-forecasting-accuracy"



sales = pd.read_csv(f"{INPUT_DIR}/sales_train_validation.csv")

sales.head()
cal = pd.read_csv(f"{INPUT_DIR}/calendar.csv")

cal["date"] = pd.to_datetime(cal["date"])

cal["day"] = cal["date"].dt.day

cal.head()
submission = pd.read_csv(f"{INPUT_DIR}/sample_submission.csv")

submission.head()
from datetime import datetime

from functools import reduce





val_ranges = [(datetime(y, 4, 25), datetime(y, 5, 22)) for y in range(2011, 2016)]

eval_ranges = [(datetime(y, 5, 23), datetime(y, 6, 19)) for y in range(2011, 2016)]



val_mask = reduce(lambda x, y: x | y, [cal["date"].between(*r) for r in val_ranges])

eval_mask = reduce(lambda x, y: x | y, [cal["date"].between(*r) for r in eval_ranges])



val_d_cols = cal[val_mask]["d"].unique().tolist()

eval_d_cols = cal[eval_mask]["d"].unique().tolist()



id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

sales_val = sales[id_cols + val_d_cols]

sales_eval = sales[id_cols + eval_d_cols]



display(sales_val, sales_eval)
def predict(sales, cal):

    return (

        sales

        .melt(id_vars=id_cols, var_name="d", value_name="demand")

        .merge(cal[["d","date", "month", "day"]], how="left", on="d")

        .groupby(["id", "month", "day"], sort=False)

        .agg({"demand": "mean", "date": "first"}).reset_index()

        .pivot(index="id", columns="date", values="demand").reset_index()

        .pipe(lambda df: submission[["id"]].merge(df, how="inner", on="id"))

    )
val_pred = predict(sales_val, cal)

eval_pred = predict(sales_eval, cal)



eval_pred["id"] = eval_pred["id"].str.replace("_validation", "_evaluation")



display(val_pred, eval_pred)
cols = ["id"] + ["F" + str(d + 1) for d in range(28)]

val_pred.columns = cols

eval_pred.columns = cols



pred = pd.concat([val_pred, eval_pred]).reset_index(drop=True)



assert pred.drop("id", axis=1).isnull().sum().sum() == 0

assert pred["id"].equals(submission["id"])



pred
pred.to_csv("submission.csv", index=False)