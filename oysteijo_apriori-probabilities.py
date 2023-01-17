import numpy as np

import pandas as pd



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
labels = pd.read_csv("../input/train_labels.csv")

sub = pd.read_csv("../input/sample_submission.csv", index_col=0)

sub.invasive = labels.invasive.mean()

sub.to_csv("apriori.csv")