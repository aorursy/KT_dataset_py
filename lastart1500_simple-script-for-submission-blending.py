import pandas as pd 

# read submission file to blend
df1 = pd.read_csv("../add_csvfile/blending_1.csv")
df2 = pd.read_csv("../add_csvfile/blending_2.csv")

# hold target label
label = df1["MachineIdentifier"]


del df1["MachineIdentifier"]
del df2["MachineIdentifier"]

# average of two submissionfiles 
df_blend = (df1 + df2)/2

# output blending file
df_final = pd.concat([label,df_blend],axis = 1)
df_final.to_csv('blending).csv', index=False)
