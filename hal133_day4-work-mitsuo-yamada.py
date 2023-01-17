import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from IPython.display import Image
df = pd.read_csv("../input/survey.csv")
display(df.head())
display(df.tail())
display(pd.DataFrame([df.shape], columns=["row", "col"]))
print(df.isnull().sum())
df['comments'] = (df['comments'].notnull()).astype(int)
print(df['comments'].value_counts().sort_index(), "\n")
# country != "United States"でstateに値があるものはNoneにする
print("対象件数", df[(df['Country'] != "United States") & (df['state'].notnull())].shape[0], "\n")
df['state'] = df['state'].where(df['Country'] == "United States", None)

print("対象件数", df[(df['Country'] != "United States") & (df['state'].notnull())].shape[0], "\n")
print(df['state'].isnull().sum(), "\n")
df['year'] = pd.to_datetime(df['Timestamp']).apply(lambda x: x.year).astype(str)
df['month'] = pd.to_datetime(df['Timestamp']).apply(lambda x: x.month).astype(str)

[print(df[col].value_counts().sort_index(), "\n") for col in ['year', 'month']]

# Timestamp列を削除
df = df.drop(['Timestamp'], axis=1)
