import pandas as pd
train_df = pd.read_csv("../input/train.csv")
kaggle=train_df.loc[train_df['question_text'].str.contains("(?i)kaggle")].copy().reset_index()
list(kaggle['question_text'])
