import numpy as np

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
test = pd.read_csv("../input/test.csv")
test["Survived"] = np.where(test["Sex"] == "female", 1, 0)

test[["PassengerId", "Survived"]].to_csv("submission.csv", index=False)