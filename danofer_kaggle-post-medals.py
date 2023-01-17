import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))

forums = pd.read_csv('../input/Forums.csv').set_index("Id").rename(columns={"Title":"Forum_Title"})

forums.head()
## basic users info (name..). Note that this is leaky(PerformanceTier)! it's not time stamped. It can also be joined with more data, e.g. teams, organization

users = pd.read_csv('../input/Users.csv').set_index("Id").drop("PerformanceTier",axis=1).drop_duplicates()

print(users.shape)

users.head()
### there are multiple ForumId s . Not unique! 

## Many of these column are effectively leaky variables (not time stamped, and they could indicate a "topic" containing a popular post)



forumsTopics = pd.read_csv('../input/ForumTopics.csv').drop(['Score'],axis=1).drop_duplicates().set_index("Id")

print(forumsTopics.shape)

print(forumsTopics.nunique())

forumsTopics.head()
forumsTopics.columns
messages = pd.read_csv('../input/ForumMessages.csv').drop("MedalAwardDate",axis=1)

messages = messages[messages.Message.notna()].drop_duplicates(subset="Message")

messages.Medal.fillna(-1,inplace=True)

messages['PostDate'] = pd.to_datetime(messages['PostDate'], infer_datetime_format=True)

# messages = messages.sort_values('PostDate')

messages.tail()
messages["Medal"].hist()
print(messages["Medal"].isna().sum())

messages["Medal"].describe()
print(messages.shape)

# messages= messages.join(forumsTopics,on="ForumTopicId",how="left")

messages = messages.join(forumsTopics,on="ForumTopicId",how="inner")

print(messages.shape)

# print(messages.join(forumsTopics,on="ForumTopicId",how="inner").shape)

# messages.tail()
messages.tail()
forums.head()
messages = messages.join(forums,on="ForumId")

## add parent forum title, if any

messages = messages.join(forums,on="ParentForumId",rsuffix="_parent")

print(messages.shape)
messages = messages.join(users,on="PostUserId")
messages.head()
messages["message_word_length"] = messages["Message"].str.split().str.len()

messages["message_char_length"] = messages["Message"].str.len()
print(messages.loc[messages.Medal > -1]["message_word_length"].quantile(0.995))

messages.loc[messages.Medal > -1]["message_word_length"].describe()
print(messages["message_word_length"].quantile(0.995))

messages["message_word_length"].describe()
messages["message_char_length"].describe()
messages = messages.loc[messages["message_word_length"]<900]

print(messages.shape[0])
messages["message_word_length"].describe()
messages.columns
messages["Medal"] = (messages["Medal"]>-1).astype(int)
# messages["First_or_last_message"] = ((messages['Id']==messages['LastForumMessageId']) | messages['Id']==messages['FirstForumMessageId'])



# messages["First_or_last_message"].describe()  ### only 20 such cases - maybe wrong key? Ignore for now
messages.tail()
messages.drop(['Id', 'ForumId','LastForumMessageId',

       'FirstForumMessageId', 'ForumTopicId','ParentForumId', 'ParentForumId_parent',

               'TotalReplies'],axis=1).drop_duplicates().to_csv("kaggle_medals_train_v1.csv.gz",index=False,compression="gzip")