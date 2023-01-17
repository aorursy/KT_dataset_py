import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error
df=pd.read_csv('../input/heart-disease/heart.csv')

df
df.info()
df.describe()
df.isnull().sum()
plt.figure(figsize=(15,7))

sns.heatmap(df.corr(),annot=True,cmap='YlGnBu')
# almost all the columns have an effect on the survival rates
feature=df.columns

y=df.target

X=df[feature]



train_x,test_x,train_y,test_y=train_test_split(X,y)
model=RandomForestRegressor(random_state=1)

model.fit(train_x,train_y)

pred=model.predict(test_x)
print("Mean abolute error",mean_absolute_error(test_y,pred))
my_submission = pd.DataFrame({'Id': test_x.index, 'target': pred})

# you could use any filename. We choose submission here

my_submission.to_csv('submission.csv', index=False)