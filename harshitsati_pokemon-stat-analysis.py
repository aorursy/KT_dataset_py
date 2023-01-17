import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('../input/poketmonster/Pokemon.csv',index_col = 0, encoding = 'unicode_escape')
df.head()
sns.lmplot(x = 'Attack', y = 'Defense', data = df)
plt.show()
sns.lmplot(x = 'Attack', y = 'Defense', data = df, fit_reg = False, hue = 'Stage') #hue will colour the pokemon according to the stage of it
plt.show()
sns.boxplot(data = df)
df_copy = df.drop(['Total','Stage','Legendary'], axis = 1) # axis - column
sns.boxplot(data = df_copy)

sns.violinplot(data = df_copy)
plt.show()
plt.figure(figsize = (10,6))
sns.violinplot(x = 'Type 1',y = 'Attack', data = df)
plt.show()
corr = df_copy.corr()
sns.heatmap(corr)
sns.distplot(df.Attack, color = 'blue')
sns.countplot(x = 'Type 1', data = df)
sns.countplot(x = 'Type 1', data = df)
plt.xticks(rotation = -45)
sns.jointplot(df.Attack,df.Defense,kind = 'kde', color = 'lightblue' ) # kind of the plot is kde which implies a density plot