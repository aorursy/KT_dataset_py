import pandas as pd





train_df = pd.read_csv('/kaggle/input/train.csv')

train_df.describe()
import pandas as pd



test_df = pd.read_csv('/kaggle/input/test.csv')

test_df.describe()
df = train_df.append(test_df)

df.describe()
import matplotlib.pyplot as plt





plt.figure(figsize=(16, 8))

for ticker in df.columns.values:

    plt.plot(list(range(len(df[ticker]))), df[ticker])

plt.legend(df.columns.values)

plt.show()