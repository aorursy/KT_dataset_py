import pandas as pd
import numpy as np
df_train_users = pd.read_csv('../input/train_users_2.csv')
df_test_users = pd.read_csv('../input/test_users.csv')
# let's look at the destinations accounted for each occurence in the train set
df_train_users.groupby("country_destination").count()["id"]
# the 5 most frequent "destinations" are ["NDF","US","other","FR","IT"]
# baseline: predict ["NDF","US","other","FR","IT"] for each user in the test set

res = [[x, destination] for x in df_test_users["id"] for destination in ["NDF","US","other","FR","IT"]]
sub_baseline = pd.DataFrame(np.array(res), columns=['id', 'country'])
sub_baseline.to_csv('sub_baseline.csv', index=False)