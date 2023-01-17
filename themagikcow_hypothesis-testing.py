

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # graphs

import seaborn as sns # graphs





df = pd.read_csv('../input/train.csv')

df.head()
sample = df.sample(n=100, random_state=100)

sample.corr().iloc[6,-1]
ax = sns.regplot(x="YearBuilt", y="SalePrice", data=df)

regression_line = np.polyfit(df['YearBuilt'], df['SalePrice'], 1)

print('The regression line is sale price = ', int(abs(regression_line[0])), '* yearBuilt + ', int(abs(regression_line[1])))