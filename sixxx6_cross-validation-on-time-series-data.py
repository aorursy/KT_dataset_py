import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit  

df=pd.read_csv('../input/ntt-data-global-ai-challenge-06-2020/COVID-19_and_Price_dataset.csv')  

fig, axes = plt.subplots(5, 1, figsize=(15, 20))  
folds = TimeSeriesSplit(n_splits=5)  
for i, (train_index, test_index) in enumerate(folds.split(df)):  
    sns.lineplot(data=df, x='Date', y='Price', ax=axes[i], label='no_use',color="0.8")  
    sns.lineplot(data=df.iloc[train_index], x='Date', y='Price', ax=axes[i], label='train',color="b")  
    sns.lineplot(data=df.iloc[test_index], x='Date', y='Price', ax=axes[i], label='Validation',color="r")  

plt.legend()  
plt.show()  
def rolling_time_series_split(df,splits):
    n_samples = len(df)
    folds = n_samples // splits
    indices = np.arange(n_samples)

    margin = 0
    for i in range(splits): 
        start = i * folds  
        stop = start + folds  
        temp = int(0.8 * (stop - start)) + start #If you want to change the data ratio of train/Validation, change the 0.8 part.
        yield indices[start: temp], indices[temp + margin: stop]  

fig, axes = plt.subplots(5, 1, figsize=(15, 20))  
for i, (train_index, test_index) in enumerate(rolling_time_series_split(df,5)):  
    sns.lineplot(data=df, x='Date', y='Price', ax=axes[i], label='no_use',color="0.8")  
    sns.lineplot(data=df.iloc[train_index], x='Date', y='Price', ax=axes[i], label='train',color="b")  
    sns.lineplot(data=df.iloc[test_index], x='Date', y='Price', ax=axes[i], label='Validation',color="r")  

plt.legend()  
plt.show()  