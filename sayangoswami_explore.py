# imports

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from collections import Counter

%matplotlib inline
%config InlineBackend.figure_format = 'svg'

print(os.listdir("../input"))
train_df = pd.read_csv('../input/drugsComTrain_raw.csv')
test_df = pd.read_csv('../input/drugsComTest_raw.csv')

train_df.info()
test_df.info()
conditions = train_df['condition'].value_counts().sort_values(ascending=False)
topk = 25

conditions[0:topk].plot(kind="bar")
plt.xlabel("Conditions")
plt.ylabel("Counts")
plt.title(f"Top {topk} conditions")

plt.show()
conditions = train_df['condition'].value_counts().sort_values(ascending=True)
bottomk = 25

conditions[0:bottomk].plot(kind="bar")
plt.xlabel("Conditions")
plt.ylabel("Counts")
plt.title(f"Bottom {topk} conditions")

plt.show()
plt.title("Distribution of usefulCount")
ax = sns.distplot(train_df["usefulCount"])

plt.show()
