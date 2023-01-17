import pandas as pd
df = pd.read_csv("/kaggle/input/submission/sub.csv")
dv = pd.read_csv("/kaggle/input/Kannada-MNIST/sample_submission.csv")
for i in range(5000):

    dv["label"][i] = df["label"][i]
dv.to_csv("submission.csv", index=False);
dv.iloc[0:5]