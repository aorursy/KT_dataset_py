import numpy as np

import pandas as pd



np.random.seed(1)

sub = pd.read_csv("/kaggle/input/alaska2-image-steganalysis/sample_submission.csv")

sub.Label = np.random.random(len(sub))

sub.to_csv("submission.csv", index=False)