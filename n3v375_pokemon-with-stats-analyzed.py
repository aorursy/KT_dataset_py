import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

df = pd.read_csv('../input/pokemon/Pokemon.csv')
df.head()
df.describe()
df.info()
plt.figure(figsize=(15,5))

sns.heatmap(df.corr(), annot=True, cmap='plasma')

plt.title('Correlation Heatmap')
sns.clustermap(df.corr(), annot=True, cmap='plasma')

plt.title('Correlation Clustermap')
plt.figure(figsize=(15,5))

sns.countplot(df['Generation'])

plt.title('Count of Pokemon by Generation')
plt.figure(figsize=(15,5))

sns.countplot(df['Generation'], hue=df['Type 1'], palette='rainbow')

plt.legend(bbox_to_anchor=(1,1), loc="best")

plt.title('Count of Type 1 Pokemon by Generation')
plt.figure(figsize=(15,5))

sns.countplot(df['Legendary'], hue=df['Generation'])

plt.title('Count of Non-Legendary vs Legendary')
plt.figure(figsize=(15,5))

sns.countplot(df[df['Legendary']==True]['Type 1'], hue=df['Generation'])

plt.title('Count of Legendary Type 1 by Generation')
plt.figure(figsize=(15,5))

sns.countplot(df[df['Legendary']!=True]['Type 1'], hue=df['Generation'])

plt.title('Count of Non-Legendary Type 1 by Generation')
plt.figure(figsize=(15,5))

sns.countplot(df[df['Legendary']!=True]['Type 2'], hue=df['Generation'])

plt.title('Count of Non-Legendary Type 2 by Generation')
plt.figure(figsize=(15,5))

sns.countplot(df['Type 1'], hue=df['Legendary'])

plt.title('Count of Legendary vs Non-Legendary by Type 1')
plt.figure(figsize=(15,5))

sns.countplot(df['Type 2'], hue=df['Legendary'])

plt.title('Count of Legendary vs Non-Legendary by Type 2')
plt.figure(figsize=(15,5))

sns.countplot(df['Type 1'])

plt.title('Count of Type 1 Pokemon')
plt.figure(figsize=(15,5))

sns.countplot(df['Type 2'])

plt.title('Count of Type 2 Pokemon')
plt.figure(figsize=(15,5))

sns.countplot(df[df['Legendary']==True]['Type 1'])

plt.title('Count of Legendary Type 1')
plt.figure(figsize=(15,5))

sns.countplot(df[df['Legendary']==True]['Type 2'])

plt.title('Count of Legendary Type 2')
plt.figure(figsize=(15,5))

sns.countplot(df[df['Legendary']!=True]['Type 1'])

plt.title('Count of Non-Legendary Type 1')
plt.figure(figsize=(15,5))

sns.countplot(df[df['Legendary']!=True]['Type 2'])

plt.title('Count of Non-Legendary Type 2')
gen1 = df[df['Generation']==1]

plt.figure(figsize=(15,5))

sns.countplot(gen1['Type 1'])

plt.title('Count of Pokemon by Generation 1')
gen2 = df[df['Generation']==2]

plt.figure(figsize=(15,5))

sns.countplot(gen2['Type 1'])

plt.title('Count of Pokemon by Generation 2')
gen3 = df[df['Generation']==3]

plt.figure(figsize=(15,5))

sns.countplot(gen3['Type 1'])

plt.title('Count of Pokemon by Generation 3')
gen4 = df[df['Generation']==4]

plt.figure(figsize=(15,5))

sns.countplot(gen4['Type 1'])

plt.title('Count of Pokemon by Generation 4')
gen5 = df[df['Generation']==5]

plt.figure(figsize=(15,5))

sns.countplot(gen5['Type 1'])

plt.title('Count of Pokemon by Generation 5')
gen6 = df[df['Generation']==6]

plt.figure(figsize=(15,5))

sns.countplot(gen6['Type 1'])

plt.title('Count of Pokemon by Generation 6')
print(f"""

Gen1 has {gen1['Type 1'].nunique()} types

Gen2 has {gen2['Type 1'].nunique()} types

Gen3 has {gen3['Type 1'].nunique()} types

Gen4 has {gen4['Type 1'].nunique()} types

Gen5 has {gen5['Type 1'].nunique()} types

Gen6 has {gen6['Type 1'].nunique()} types

      """)
df.nlargest(10, 'Total')
df.nlargest(10, 'HP')
df.nlargest(10, 'Attack')
df.nlargest(10, 'Defense')
df.nlargest(10, 'Sp. Atk')
df.nlargest(10, 'Sp. Def')
df.nlargest(10, 'Speed')