import pandas as pd
df = pd.read_csv("/kaggle/input/ptt-gossiping-hot-posts-20190608-to-20200607/data.csv", sep="\t")
df.info()
df.head()