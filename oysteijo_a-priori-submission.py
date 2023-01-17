import numpy as np

import pandas as pd
df = pd.read_csv("../input/training_variants", index_col=0)
df.head()
prior = df.groupby("Class").count()["Gene"] / df.shape[0]
prior
sub = pd.read_csv("../input/submissionFile", index_col=0)

sub.drop(["class{}".format(s+1) for s in range(9)], axis=1, inplace=True)

pr = prior.as_matrix()

for c in range(9):

    col = "class{}".format(c+1)

    sub[col] = pr[c]
sub.head()
sub.to_csv("apriori.csv")