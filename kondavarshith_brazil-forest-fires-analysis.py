import pandas as pd

import chardet

import matplotlib.pyplot as plt

import seaborn as sns 


with open('../input/forest-fires-in-brazil/amazon.csv', 'rb') as f:

    result = chardet.detect(f.read())  # or readline if the file is large





data = pd.read_csv('../input/forest-fires-in-brazil/amazon.csv', encoding=result['encoding'])
data.head(200)

months = data['month']

fig=plt.figure(figsize=(80,6))
data.shape

data.count()
data.number[:200].plot()
data.state.unique()
state_data=data.pivot_table(index="state")

print(state_data)
plt.figure(figsize=(25,7))

sns.barplot(x=data.state,y=data.number)

plt.title("fires WRT state")
data.number[:200].plot()
type(data.index)
data.index=pd.to_datetime(data.index)
data.number[:100].plot()