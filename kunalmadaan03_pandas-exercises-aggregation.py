import pandas as pd

import numpy as np
pd.read_csv("../input/aggregationdata/occupation.csv",sep="|",index_col="user_id")
user = pd.read_csv("../input/aggregationdata/occupation.csv",sep="|",index_col="user_id")
user.groupby(by="occupation")[["age"]].mean().add_prefix("MeanOf_").reset_index()
out1 = user.groupby(by=["occupation","gender"])[["gender"]].count().add_prefix("CountOf_").reset_index()

out2 = out1.pivot(index="occupation",columns="gender",values="CountOf_gender")

out2 = out2.fillna(0)
res1 = out2.F + out2.M

res2 = out2.M/res1

final_res = res2.sort_values(ascending=False)

final_res
temp1 = user.groupby(by="occupation")[["age"]].min().add_prefix("Min_")

temp2 = user.groupby(by="occupation")[["age"]].max().add_prefix("Max_")

out5a = pd.concat([temp1,temp2],axis=1)

out5a
out5 = user.groupby(by=["occupation","gender"])[["age"]].mean().add_prefix("MeanOf_").reset_index()

out6 = out5.pivot(index="occupation",columns="gender",values="MeanOf_age")

out6
res3 = round(res2*100,2)

res4 = 100-res3

user_tab = pd.concat([res3,res4],axis=1).reset_index()
user_tab