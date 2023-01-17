import numpy as np

import pandas as pd
#read csv

df = pd.read_csv("../input/iris/Iris.csv")                                    

df = df.rename(columns={"Species": "Label"})                    



df
df = df.reindex(np.random.permutation(df.index))                                                          

print("Randomized data by randomizing the index : ")



df
df = df.reset_index(drop=True)                                                                                   

print("Randomized data with reseted index: ")



df
#folds                                                              

fold1 = df.loc[0:29]                                            

fold2 = df.loc[30:59]

fold3 = df.loc[60:89]

fold4 = df.loc[90:119]

fold5 = df.loc[120:149]
train_val1 = pd.concat([fold1, fold2, fold3, fold4])

test_val1 = fold5



train_val2 = pd.concat([fold1, fold2, fold3, fold5])

test_val2 = fold4



train_val3 = pd.concat([fold1, fold2, fold4, fold5])

test_val3 = fold3



train_val4 = pd.concat([fold1, fold3, fold4, fold5])

test_val4 = fold2



train_val5 = pd.concat([fold2, fold3, fold4, fold5])

test_val5 = fold1
train_val4
test_val4