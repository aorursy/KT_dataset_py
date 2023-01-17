import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



df_train = pd.read_csv(r"../input/fashion_data_info_train_competition.csv")

df_test = pd.read_csv(r"../input/fashion_data_info_val_competition.csv")

df_train = df_train[['itemid', 'title']].set_index('itemid').rename(columns={'title': 'title_train'})

df_test = df_test[['itemid', 'title']].set_index('itemid').rename(columns={'title': 'title_test'})

df_combined = pd.merge(df_train, df_test, on='itemid')

df_combined.loc[(df_combined['title_train'] == df_combined['title_test'])]