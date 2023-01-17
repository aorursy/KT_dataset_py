import numpy as np

import pandas as pd



users_df = pd.read_csv('../input/meta-kaggle/Users.csv')

followers_df = pd.read_csv('../input/meta-kaggle/UserFollowers.csv')



user_id_dict = dict(zip(users_df.Id, users_df.UserName))

followers = list(followers_df["FollowingUserId"].value_counts())

followed = list(followers_df["FollowingUserId"].value_counts().index)



most_followed = enumerate(followed[:100])



for i, user_id in most_followed:

    if user_id in user_id_dict.keys():

        print(str(i+1) + ". " + user_id_dict[user_id] + " (" + str(followers[i]) + ")")