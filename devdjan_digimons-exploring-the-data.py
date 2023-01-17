# Importing libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

df_train = pd.read_csv('../input/DigiDB_digimonlist.csv')
# pd.read_csv('../input/DigiDB_movelist.csv')
# pd.read_csv('../input/DigiDB_supportlist.csv')

# WE will dislpay the number of rows and colum
df_train.shape
df_train.describe()
df_train.head()
# Displaying the columns in our dataset
df_train.columns
df_train.info()
# This command gives basic information about each column in dataset
f, ax = plt.subplots(figsize=(15,15))
sns.heatmap(df_train.corr(),annot=True, linewidths=.1, fmt= '.1f',ax=ax, cmap="YlGnBu")
# let's look for the maximum HP
df_train['Lv 50 HP'].max()
df_train.boxplot(column='Lv 50 HP', by="Lv50 Atk")
data = df_train.loc[:,['Lv 50 HP','Lv50 Atk', 'Lv50 Def']]
data.plot()
# subplots
data.plot(subplots=True)
# scatter plot
data.plot(kind = 'scatter', x='Lv50 Atk', y='Lv50 Def')
# hist plot
data.plot(kind='hist', y='Lv50 Def', bins = 50, range=(0,300), normed=True)
df_train['Type'].unique()
# Now i want to know, what type of digimons have the biggest attack ?
pd.crosstab(df_train['Type'], df_train['Lv50 Atk'])
df_train[df_train['Type'] == 'Data']['Lv50 Atk'].max()
df_train[df_train['Type'] == 'Free']['Lv50 Atk'].max()
df_train[df_train['Type'] == 'Vaccine']['Lv50 Atk'].max()
df_train[df_train['Type'] == 'Virus']['Lv50 Atk'].max()
digimons_movelist = pd.read_csv("../input/DigiDB_digimonlist.csv")
digimons_movelist['Type'].value_counts().plot(kind='bar')
plt.title('Digimonsters type')
plt.ylabel('Count')
plt.show()