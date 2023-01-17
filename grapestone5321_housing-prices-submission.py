import numpy as np 

import pandas as pd 

import os

print(os.listdir("../input"))
df = pd.read_csv("../input/sample_submission.csv")

df.to_csv("submission.csv", index = False)