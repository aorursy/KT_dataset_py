# Note: This code is designed to run on a machine, and not natively in Kaggle. For visual purposes only.

import pandas
import datetime
from pathlib import Path

# import master data
uni = pandas.read_csv("master_data.csv")

# import update, append to uni dataframe
if Path("update.csv").exists():
    up = pandas.read_csv("update.csv", header=11)
    up.columns = uni.columns
    uni = uni.append(up, ignore_index=True)
uni.drop_duplicates("ordernumber", inplace=True)

# narrow down uni to report fields
uni2 = uni[["supervisor", "type", "scheduled", "clientname", "ordernumber"]]
uni2 = uni2[uni2["type"].isin(["New Work", "Redo", "Redo/Warranty"])]

# narrow to last scheduled date
uni2["scheduled"] = pandas.to_datetime(uni2.scheduled)
df1 = uni2[["clientname", "scheduled"]].groupby(["clientname"]).max()
df2 = pandas.merge(uni2, df1, left_on=["clientname", "scheduled"], right_on=["clientname", "scheduled"])
df3 = df2[["clientname", "ordernumber"]].groupby(["clientname"]).max()
df4 = pandas.merge(df3, df2, left_on=["clientname", "ordernumber"], right_on=["clientname", "ordernumber"])

# sort results
df4 = df4.sort_values(by=["supervisor", "type", "scheduled"], ascending =[True, True, False])

# create / calculcate "last scheduled" field
df4["daysago"] = df4[["scheduled"]]
today = pandas.to_datetime(datetime.date.today())
df4.loc[df4["scheduled"]<= today, "daysago"]= today - df4["daysago"]
df4["daysago"] = df4.daysago.dt.days

# export csv, remove index
df4.to_csv("Report.csv", index=False)
uni.to_csv("master_data.csv", index=False)

