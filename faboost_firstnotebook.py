import pandas as pd
%ls
%ls ../input
df = pd.read_csv("../input/gifts.csv")
df.head()
df = pd.read_csv("../input/sample_submission.csv")
df.head()
