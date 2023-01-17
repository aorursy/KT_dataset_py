from pandas import pandas

pn = pandas.read_csv("../input/tweeterclean/submission (2).csv")
print(pn)
pn.to_csv("/kaggle/working/submission.csv",index=False)