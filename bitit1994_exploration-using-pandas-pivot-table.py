import pandas as pd
train = pd.read_csv("../input/train.csv")

train.head()
sysTab_df = pd.pivot_table(train, values='song_id', index=['source_system_tab'], columns=['target'], aggfunc=len)
sysTab_df['RecurringRatio'] = sysTab_df[1]/sysTab_df[0]
sysTab_df.sort_values(by='RecurringRatio', ascending=False)
screenName = pd.pivot_table(train, values='song_id', index=['source_screen_name'], columns=['target'], aggfunc=len)

screenName['RecurringRatio'] = screenName[1]/screenName[0]

screenName.sort_values(by='RecurringRatio', ascending=False)
srType = pd.pivot_table(train, values='song_id', index=['source_type'], columns=['target'], aggfunc=len)

srType['RecurringRatio'] = srType[1]/srType[0]

srType.sort_values(by='RecurringRatio', ascending=False)