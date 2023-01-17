import pandas as pd
FILE_PATH = '/kaggle/input/groceries-dataset/Groceries_dataset.csv'

df = pd.read_csv(FILE_PATH)

df.head()
df.shape
df.itemDescription.hist()
df.itemDescription.nunique(), df.Member_number.nunique()
df.Member_number.hist()
max_item = 5

n_association = 10

dict_cross_sell_result = {}



for item in df.itemDescription.unique()[:max_item]: # (1)

    list_member_with_item = df[df['itemDescription'] == item].Member_number.tolist() # (2)

    df_transaction_of_member = df[df['Member_number'].isin(list_member_with_item)] # (3)

    cross_sell_item = df_transaction_of_member.itemDescription.value_counts().index.tolist()[1:n_association] # (4)

    dict_cross_sell_result[item] = cross_sell_item # (5)

    
dict_cross_sell_result