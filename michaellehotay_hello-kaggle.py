import pandas as pd
df = pd.read_csv('/kaggle/input/hyperion/train.csv')

df2 = pd.read_csv('/kaggle/input/hyperion/test.csv')
df2['price'] = df['price'].mean()

df2[['id', 'price']].to_csv('ML-2019-12-10-1.csv', index=False)