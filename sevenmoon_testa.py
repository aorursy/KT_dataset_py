import numpy as np
import pandas as pd
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
data = pd.read_csv("../input/train_info.csv")
data.info()