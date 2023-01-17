import pandas as pd

# Load train data
df_train = pd.read_csv('../input/train.csv')
# Drop ID and target, they are not needed for the analys
df_train = df_train.drop(['ID', 'target'], axis=1)

# Similar for test
df_test = pd.read_csv('../input/test.csv')
df_test = df_test.drop(['ID'], axis=1)

# Concat both datasets
all_data = df_train.append(df_test, ignore_index=True)

all_data.info()