import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
print("hello world")
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )

print(train)