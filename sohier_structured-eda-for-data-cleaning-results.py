import pandas as pd

import random
df = pd.read_csv("../input/AER_credit_card_data.csv")
df.info()
random.seed(42)

df[random.randint(0, len(df)):].head()
df.describe()
%matplotlib inline

import seaborn as sns
sns.pairplot(df);
df.dependents.value_counts()