import os
import numpy as np
import pandas as pd
import json
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
#read train.csv
df = pd.read_csv("../input/train.csv", index_col=3)
#specify json columns
json_col = ["device", "geoNetwork", "totals", "trafficSource"]
#JSON to columns 
for col in json_col:
    tmp_array = np.array(df[col])
    
    tmp_jsonstr = []
    for i in range(len(tmp_array)):
        tmp_jsonstr.append(json.loads(tmp_array[i]))
    
    tmp_df = pd.DataFrame(tmp_jsonstr, index=df.index)
    df = pd.concat([df, tmp_df], axis=1)
        
#     #each col to each dataframe 
#     exec("df_json_{} = pd.DataFrame(tmp_jsonstr, index={})".format(col, index))
#resolve nested
df_adwordsClickInfo = pd.DataFrame(list(df["adwordsClickInfo"]), index=df.index)
df = pd.concat([df, df_adwordsClickInfo], axis=1)
df.head()