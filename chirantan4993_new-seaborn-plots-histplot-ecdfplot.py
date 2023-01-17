!pip install seaborn==0.11.0
print(sns.__version__)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
wine=pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
wine.head(10)
wine.info()
wine.columns
plt.figure(figsize=(12,12))

sns.histplot(data=wine,x='alcohol',kde=True,hue='quality',element='step')
plt.figure(figsize=(12,12))
sns.histplot(data=wine,x='alcohol',kde=True)
plt.figure(figsize=(12,12))
sns.histplot(data=wine,y='fixed acidity')
plt.figure(figsize=(12,12))
sns.histplot(data=wine,x='pH',y='quality',hue='quality')

plt.figure(figsize=(12,12))
sns.ecdfplot(data=wine, x="fixed acidity")
plt.figure(figsize=(12,12))
sns.ecdfplot(data=wine, x="pH", hue="quality", stat="proportion")