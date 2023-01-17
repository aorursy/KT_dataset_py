import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# df = pd.read_csv("../input/train-balanced-sarc.csv.gz",sep="\t",names=["label","comment","author","subreddit","score","ups","downs","date","created_utc","parent_comment"]) # Applies to original data , which lacked headers!

df = pd.read_csv("../input/train-balanced-sarcasm.csv")
print(df.shape)
df.head()
# Parse UNIX epoch timestamp as datetime: 
# df.created_utc = pd.to_datetime(df.created_utc,unit="s") # Applies to original data , which had UNIX Epoch timestamp! 
df.created_utc = pd.to_datetime(df.created_utc,infer_datetime_format=True) # Applies to original data , which had UNIX Epoch timestamp! 
df.describe()
## Nothing interesting over time (Likely due to the data being sampled then downsampled by class)
df.set_index("created_utc")["label"].plot()
df.drop(["date"],axis=1).to_csv("train-balanced-sarcasm.csv.gzip",index=False, compression="gzip")