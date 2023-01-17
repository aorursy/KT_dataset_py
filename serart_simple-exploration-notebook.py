import numpy as np

import pandas as pd 

import matplotlib.pyplot as plot

import seaborn as sns



color = sns.color_palette()



%matplotlib inline



pd.options.mode.chained_assignment = None

pd.options.display.max_columns = 999



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")
print('Train shape: ', train_df.shape)

print('Test shape: ', test_df.shape)



train_df.head()



y = train_df['y'].values



print('Y mean: ', np.mean(y))

print('Standart: ', np.std(y))
plot.figure(figsize=(12,8))

plot.scatter(range(train_df.shape[0]), np.sort(train_df.y.values))

plot.xlabel('index', fontsize=12)

plot.ylabel('y', fontsize=12)

plot.show()
plot.figure(figsize=(12,8))

sns.distplot(y, bins=50, kde=False)

plot.xlabel('y value', fontsize=12)

plot.show()