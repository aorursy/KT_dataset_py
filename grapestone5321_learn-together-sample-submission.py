import numpy as np

import pandas as pd

import os

print(os.listdir("../input/learn-together"))
df = pd.read_csv("../input/learn-together/sample_submission.csv")

df.to_csv("submission.csv", index=False)