import pandas as pd

# get airbnb & test csv files as a DataFrame
airbnb_df  = pd.read_csv('../input/train_users.csv')
test_df    = pd.read_csv('../input/test_users.csv')

# preview the data
#airbnb_df.head()

test_df.head()
