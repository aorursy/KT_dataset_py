import numpy as np
import pandas as pd
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
train_df = pd.read_csv("../input/titanic_train.csv", 
                       index_col='PassengerId') 
train_df.head(2)
train_df.describe(include='all')
train_df.info()
train_df = train_df.drop('Cabin', axis=1).dropna()
train_df.shape
# You code here
# You code here
# You code here
# You code here
# You code here
# You code here