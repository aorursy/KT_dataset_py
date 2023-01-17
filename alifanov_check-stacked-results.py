import pandas as pd



sub = pd.read_csv("../input/stack-mean/sample_sub.csv")

sub.to_csv('sample_sub.csv',index=False)