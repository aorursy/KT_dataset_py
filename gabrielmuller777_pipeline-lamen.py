#Importando o pandas, matplotlib e a nossa base de dados em csv fornecida pelo próprio Kaggle

import matplotlib.pyplot as plt

import pandas as pd

df = pd.read_csv("../input/lalamen/ramen-ratings.csv")

#Vamos retirar algumas colunas como a top ten, já que utilizaremos nossa própria metodologia para classificação

df.drop(columns=['Top Ten','Style'], inplace=True)
df[['Stars']].dtypes
df['Stars'] = pd.to_numeric(df['Stars'], errors='coerce')

df.dtypes

good_lamen = df[(df['Stars'] > 4)]

good_lamen
good_lamen_japan = good_lamen[(good_lamen['Country'] == 'Japan')]

good_lamen_japan
good_lamen = good_lamen_japan[good_lamen_japan['Variety'].str.contains("Kimchi")==True]

good_lamen
#Vamos agrupar por país para sabermos quais países tem mais variedades de marcas

df2 = df.groupby('Country')['Brand'].nunique()

df2.sort_values(ascending=False, inplace=True)

df3= df2.head(10)

df3.plot(kind='bar')

plt.ylabel('Number of brands')

plt.xlabel('Countries')

plt.show()
df2 = df.groupby('Country')['Stars'].mean()

df2.sort_values(ascending=False, inplace=True)

df3= df2.head(10)

df3.plot(kind='bar')

plt.ylabel('mean value of stars')

plt.xlabel('Countries')

plt.show()
df.loc[df['Country'] == 'Brazil']