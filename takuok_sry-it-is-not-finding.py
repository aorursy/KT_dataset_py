import pandas as pd

import matplotlib.pyplot as plt
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")
feature = [c for c in train_df.columns if c not in ['ID_code', 'target']]
train_df.describe()
test_df.describe()
med = train_df[feature].median()

med = med.reset_index()

med.columns = ["feature", "value"]

med.sort_values(by="value", inplace=True)

f = list(med.feature)
for i in range(10):

    train_df.loc[i, f].plot()

    plt.show()