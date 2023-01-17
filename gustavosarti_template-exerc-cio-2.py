import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

df = pd.read_csv('../input/dataviz-facens-20182-ex3/BlackFriday.csv') 
print(df.info())
print("")
print(df.describe())

df.head(3)
idd = df.sort_values(by='Age')

dims = (16, 9)
fig, ax = plt.subplots(figsize=dims)
p=sns.violinplot( y=idd["Age"], x=idd["Purchase"], palette="GnBu_d")
p.set_title('Valor gasto por faixa de idade')
pmc = df[df['Product_Category_1'] > 8]
pmc = pmc['Product_Category_1'].astype(str)
pmc.value_counts()

pd = pmc.value_counts()

#figure(num=None, figsize=(15, 10), dpi=80, edgecolor='green')
plt.bar(pd.keys(), pd, color='teal')
plt.title('Produtos mais comprados com N > 8')
plt.show()
asd = pd.DataFrame
for indice, val in df['Occupation'].value_counts().head(5).iteritems():    
    if asd.empty:        
        asd = df[df['Occupation'] == indice]
    else:
        asd = asd.append(df[df['Occupation'] == indice])

asd = asd.sort_values(by='Age')
        
plt.figure(figsize=(16, 9))
plt.title('Gastos por Faixa Etária nas 5 ocupações mais frequentes')
sns.boxplot(x = asd['Occupation'],
              y = asd['Purchase'], 
              hue = asd['Age'],
              linewidth = 3,
              palette="GnBu_d")
purchase = df[df['Purchase'] > 9000]
sns.catplot(x='Marital_Status', 
            y='Purchase',
            hue='Marital_Status',
            margin_titles=True,
            kind='violin',
            col='Occupation',
            data=purchase,
            aspect=.4,
            col_wrap=7,
            palette="GnBu_d")