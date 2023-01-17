
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#Data Load
train =pd.read_csv("../input/train.csv",index_col = "item_id", parse_dates = ["activation_date"])

test =pd.read_csv("../input/test.csv",index_col = "item_id", parse_dates = ["activation_date"])

del(train["deal_probability"])
df = pd.concat([train,test],axis = 0)
print(len(df["price"]))
print(df["price"].isnull().sum())
def fillna(lis,df,target):
    count = 0
    while 0 < len(lis):
        count += 1
        colname = str(count)
        print("groupby_"+",".join(lis)+"_mean")
        tmp = df.groupby(lis)[[target]].mean()
        tmp.reset_index(inplace = True)
        tmp.columns = lis + [colname]
        df = pd.merge(df, tmp, how='left', on=lis)
        df.loc[df[target].isnull(),target] = df.loc[df[target].isnull(),colname]
        
        del(df[colname])
        lis.pop()
    
    return df
    print("price_nan_sum : "+str(df["price"].isnull().sum()))
    
df = fillna(["category_name","parent_category_name","image_top_1","city"],df,"price")
