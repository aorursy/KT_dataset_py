import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
tips = pd.read_csv('../input/tips.csv')
tips.head()
sns.countplot(x='sex',data=tips)
sns.set_style('ticks')
sns.countplot(x='sex',data=tips,palette='deep')
sns.despine()
sns.countplot(x='sex',data=tips)
sns.despine(left=True)
plt.figure(figsize=(12,3))
sns.countplot(x='sex',data=tips)
sns.set_context('poster',font_scale=2)
sns.countplot(x='sex',data=tips,palette='coolwarm')