import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Output all random to see a baseline

data = pd.read_csv("../input/sample_submission.csv")

for i in range(len(data)):

    data.loc[i,1:] = np.random.uniform(0.0, 1.0)

data.to_csv("submission.csv", index=False)
data.head()