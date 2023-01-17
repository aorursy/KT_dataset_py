import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import re



df = pd.read_csv('../input/menu.csv')

print(list(df.columns))

print('done with imports.')
corr = df.corr()

plt.figure(figsize=(10,10))

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True)

plt.title("Mc D Correlation Heatmap")



plt.show()
df['Category'] = df['Category'].astype('category')



plt.figure(figsize=(10,10))

sns.violinplot(x="Category", y="Calories", data=df, inner=None)

sns.swarmplot(x="Category", y="Calories", data=df, color="white", edgecolor="gray")

plt.xticks(rotation = 75,fontsize=15)

plt.yticks(fontsize=15)

plt.xlabel('Category',fontsize=15)

plt.ylabel('Calories',fontsize=15)

plt.title('Calories by Category', fontsize=15)

plt.tight_layout()

plt.show()



n_bins = 5

x = np.arange(min(df['Calories']),max(df['Calories']),(max(df['Calories'])-min(df['Calories']))/n_bins)

plt.figure(figsize=(10,10))

plt.hist(df['Calories'], bins=n_bins, density=True)

plt.xticks(x,fontsize=15)

plt.yticks(fontsize=15)

plt.xlabel('Calories',fontsize=15)

plt.ylabel('Probability',fontsize=15)

plt.title('Calories Distribution', fontsize=15)

plt.tight_layout()

plt.show()
print(df[df['Calories'] > 1750]['Item'])
oz_list = []

for i in df['Serving Size'].tolist():

    oz = re.match(r'\d+\.?\d*',i).group(0)

    oz_list.append(float(oz))

df['oz'] = oz_list

df['cal/unit'] = df['Calories']/df['oz']



plt.figure(figsize=(10,10))

sns.violinplot(x="Category", y="cal/unit", data=df, inner=None)

sns.swarmplot(x="Category", y="cal/unit", data=df, color="white", edgecolor="gray")

plt.xticks(rotation = 75,fontsize=15)

plt.yticks(fontsize=15)

plt.xlabel('Category',fontsize=15)

plt.ylabel('Calories/Unit (oz/fl oz)',fontsize=15)

plt.title('Calories/Unit (oz/fl oz) by Category', fontsize=15)

plt.tight_layout()

plt.show()



n_bins = 5

x = np.arange(min(df['cal/unit']),max(df['cal/unit']),(max(df['cal/unit'])-min(df['cal/unit']))/n_bins)

plt.figure(figsize=(10,10))

plt.hist(df['cal/unit'], bins=n_bins, density=True)

plt.xticks(x,fontsize=15)

plt.yticks(fontsize=15)

plt.xlabel('Calories/Unit (oz/fl oz)',fontsize=15)

plt.ylabel('Probability',fontsize=15)

plt.title('Calories/Unit (oz/fl oz) Distribution', fontsize=15)

plt.tight_layout()

plt.show()
print(df[df['cal/unit'] > 150]['Item'])


cols = ['Saturated Fat (% Daily Value)', 'Cholesterol (% Daily Value)', 'Sodium (% Daily Value)']



for i in cols:

    plt.figure(figsize=(10,10))

    sns.swarmplot(x="Category", y=i, data=df)

    plt.xticks(rotation = 75,fontsize=15)

    plt.yticks(fontsize=15)

    plt.xlabel('Category',fontsize=15)

    plt.ylabel(i,fontsize=15)

    plt.title(i + ' by Category', fontsize=15)

    plt.tight_layout()

    plt.show()

    
df['fat/unit'] = df['Saturated Fat']/df['oz']

df['chol/unit'] = df['Cholesterol']/df['oz']

df['sod/unit'] = df['Sodium']/df['oz']



new_cols = list(df.iloc[:,-3:].columns)



for i in new_cols:

    plt.figure(figsize=(10,10))

    sns.swarmplot(x="Category", y=i, data=df)

    plt.xticks(rotation = 75,fontsize=15)

    plt.yticks(fontsize=15)

    plt.xlabel('Category',fontsize=15)

    plt.ylabel(i,fontsize=15)

    plt.title(i + ' by Category', fontsize=15)

    plt.tight_layout()

    plt.show()