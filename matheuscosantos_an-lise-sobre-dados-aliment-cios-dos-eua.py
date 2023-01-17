import matplotlib.pyplot as plt 

import pandas as pd

import os

diretorios = os.listdir("../input")



df = pd.read_csv('../input/en.openfoodfacts.org.products.tsv',delimiter='\t',encoding='utf-8')

df = df.drop(['creator','created_t','last_modified_datetime','last_modified_t','url','created_datetime','no_nutriments', 'additives_n','ingredients_from_palm_oil_n','ingredients_from_palm_oil','ingredients_that_may_be_from_palm_oil_n','ingredients_that_may_be_from_palm_oil','nutrition_grade_uk'], axis=1)
colunas = []



for coluna in df.columns:

    colunas.append(coluna.replace('-','_'))

    

df.columns = colunas
df_countrys = df.groupby(df.countries_en).count()

df_countrys = df_countrys.sort_values(by='code',ascending=False)

df_countrys = df_countrys.code.head(5)

df_countrys = df_countrys.to_frame()

df_countrys.columns = ['total_of_products']

print(df_countrys)
df_countrys.plot.pie(y=('total_of_products'), figsize=(8, 8),autopct='%.2f', title="Porcentagem de produtos cadastrados por país")

plt.legend(loc='center left', bbox_to_anchor=(1.3, 0.5), title="Países")
df_only_us = df[df['countries_en']=='United States']

df_brands = df_only_us.groupby(df_only_us.brands)

df_brands = df_brands.count()

df_brands = df_brands.sort_values(by='code',ascending=False)

df_brands = df_brands.head(15)

df_brands = df_brands['code'].to_frame()

print(df_brands.head())
df_brands.plot.pie(y=('code'), figsize=(8, 8),autopct='%.2f')

plt.legend(loc='center left', bbox_to_anchor=(1.3, 0.5))
df_us_saturated_fat = df_only_us.reset_index()

df_us_saturated_fat = df_us_saturated_fat[['brands','saturated_fat_100g']]

df_us_saturated_fat = df_us_saturated_fat.groupby(['brands'])['saturated_fat_100g'].sum()

df_us_saturated_fat = df_us_saturated_fat.to_frame()

df_us_saturated_fat = df_us_saturated_fat.sort_values(by='saturated_fat_100g',ascending=False)

df_us_saturated_fat.head(10)
df_us_fat = df_only_us.reset_index()

df_us_fat = df_us_fat[['brands','fat_100g']]

df_us_fat = df_us_fat.groupby(['brands'])['fat_100g'].sum()

df_us_fat = df_us_fat.to_frame()

df_us_fat = df_us_fat.sort_values(by='fat_100g',ascending=False)

df_us_fat.head(10)
df_us_fat.head(15).plot.barh(figsize=(8, 8),grid=True, title="Quantidade de gordura em cada 100g, agrupadas por marcas de produtos").set_axisbelow(True)

plt.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))
df_us_saturated_fat.head(15).plot.barh(figsize=(8, 8),grid=True, title="Quantidade de gordura saturada em cada 100g, agrupadas por marcas de produtos").set_axisbelow(True)

plt.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))
df_only_us['sum'] = df_only_us[['fat_100g','saturated_fat_100g']].astype(float).sum(axis=1)

df_us_sum_fat = df_only_us.groupby(['brands'])['sum'].sum()

df_us_sum_fat = df_us_sum_fat.to_frame()

df_us_sum_fat = df_us_sum_fat.sort_values(by='sum',ascending=False)

df_us_sum_fat.head(10)
df_us_sum_fat.head(15).plot.pie(y=('sum'), figsize=(8, 8), title="Porcentagem das quinze marcas de produtos com maior concentração de gordura normal e saturada em 100g de alimento",autopct='%.2f')

plt.legend(loc='center left', bbox_to_anchor=(1.3, 0.5))
df_us_sum_fat.head(15).plot.barh(figsize=(8, 8),grid=True, title="Quantidade de gordura normal e saturada em cada 100g, agrupadas por marcas de produtos").set_axisbelow(True)

plt.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))