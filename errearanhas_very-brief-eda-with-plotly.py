import pandas as pd

import plotly.express as px
train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
print(train_df.shape)

print(test_df.shape)
train_df.head()
data = train_df

fig = px.pie(data, values='id', names='target')

fig.show()
data = train_df['text'].apply(lambda x: len(x))

fig = px.histogram(data, x="text")

fig.show()
train_df['len'] = train_df['text'].apply(lambda x: len(x))



fig = px.histogram(train_df, x="len", y="target", color='target', barmode='group', height=700)

fig.show()