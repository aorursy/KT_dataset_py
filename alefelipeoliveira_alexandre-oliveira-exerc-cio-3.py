import pandas as pd

from matplotlib import pyplot as plt



df = pd.read_csv("../input/BlackFriday.csv")



age_band = list(df["Age"].unique())



#Reordenando variável nominal ordinal

age_band[1], age_band[3], age_band[4], age_band[5], age_band[6] = age_band[6], age_band[5], age_band[3], age_band[4], age_band[1]



import seaborn as sns

sns.set(rc={'figure.figsize':(13,8.27)})

sns.violinplot( x=df["Age"], y=df["Purchase"], linewidth=0.3, order=age_band)

plt.title('Distribuição dos valores gastos por faixa de idade')

#Data ink Ratio

plt.ylabel('')

plt.xlabel('')
sns.set(rc={'figure.figsize':(13,8.27)})

sns.boxplot( x=df["Age"], y=df["Purchase"], linewidth=0.3, order=age_band)

plt.title('Distribuição dos valores gastos por faixa de idade')

plt.ylabel('')

plt.xlabel('')
#Retirando "nan"

df = df.fillna(0)



cat_list1 = ['Product_Category_1', 'Product_Category_2', 'Product_Category_3']



#Dataframe para N>8

product_df = df.drop(df[(df.Product_Category_1 + df.Product_Category_2 + df.Product_Category_3) <= 8].index)



sns.barplot(x="Age" , y="Product_Category_1", data=product_df, order = age_band)

plt.title('Quantidade Média de compra de produtos da categoria 1 por faixa de idade')

plt.ylabel('')

plt.xlabel('')
sns.barplot(x="Age" , y="Product_Category_2", data=product_df, order = age_band)

plt.title('Quantidade Média de compra de produtos da categoria 2 por faixa de idade')

plt.ylabel('')

plt.xlabel('')
sns.barplot(x="Age" , y="Product_Category_3", data=product_df, order = age_band)

plt.title('Quantidade Média de compra de produtos da categoria 3 por faixa de idade')

plt.ylabel('')

plt.xlabel('')
occ_list = list(df.Occupation.value_counts().index[0:5])

print (occ_list)
sns.set(rc={'figure.figsize':(13,8.27)})

#Separa no dataframe apenas as ocupações top 5.

occ_df_17 = df[(df.Occupation.isin(occ_list)) & (df.Age == '0-17')]

sns.violinplot( x=occ_df_17["Occupation"], y=occ_df_17["Purchase"], linewidth=0.3)

plt.title('Consumo das ocupações top 5 na faixa de 0-17 anos')

plt.ylabel('')

plt.xlabel('')
sns.set(rc={'figure.figsize':(13,8.27)})

#Separa no dataframe apenas as ocupações top 5.

occ_df_18 = df[(df.Occupation.isin(occ_list)) & (df.Age == '18-25')]

sns.violinplot( x=occ_df_18["Occupation"], y=occ_df_18["Purchase"], linewidth=0.3)

plt.title('Consumo das ocupações top 5 na faixa de 18-25 anos')

plt.ylabel('')

plt.xlabel('')
sns.set(rc={'figure.figsize':(13,8.27)})

#Separa no dataframe apenas as ocupações top 5.

occ_df_26 = df[(df.Occupation.isin(occ_list)) & (df.Age == '26-35')]

sns.violinplot( x=occ_df_26["Occupation"], y=occ_df_26["Purchase"], linewidth=0.3)

plt.title('Consumo das ocupações top 5 na faixa de 26-35 anos')

plt.ylabel('')

plt.xlabel('')
sns.set(rc={'figure.figsize':(13,8.27)})

#Separa no dataframe apenas as ocupações top 5.

occ_df_36 = df[(df.Occupation.isin(occ_list)) & (df.Age == '36-45')]

sns.violinplot( x=occ_df_36["Occupation"], y=occ_df_36["Purchase"], linewidth=0.3)

plt.title('Consumo das ocupações top 5 na faixa de 36-45 anos')

plt.ylabel('')

plt.xlabel('')
sns.set(rc={'figure.figsize':(13,8.27)})

#Separa no dataframe apenas as ocupações top 5.

occ_df_46 = df[(df.Occupation.isin(occ_list)) & (df.Age == '46-50')]

sns.violinplot( x=occ_df_46["Occupation"], y=occ_df_46["Purchase"], linewidth=0.3)

plt.title('Consumo das ocupações top 5 na faixa de 46-50 anos')
sns.set(rc={'figure.figsize':(13,8.27)})

#Separa no dataframe apenas as ocupações top 5.

occ_df_51 = df[(df.Occupation.isin(occ_list)) & (df.Age == '51-55')]

sns.violinplot( x=occ_df_51["Occupation"], y=occ_df_51["Purchase"], linewidth=0.3)

plt.title('Consumo das ocupações top 5 na faixa de 51-55 anos')
sns.set(rc={'figure.figsize':(13,8.27)})

#Separa no dataframe apenas as ocupações top 5.

occ_df_18 = df[(df.Occupation.isin(occ_list)) & (df.Age == '55+')]

sns.violinplot( x=occ_df_18["Occupation"], y=occ_df_18["Purchase"], linewidth=0.3)

plt.title('Consumo das ocupações top 5 na faixa de +55 anos')
bigger_df = df[(df.Purchase > 9000)]



plt.scatter(bigger_df.Occupation, bigger_df.Marital_Status, s=bigger_df.Occupation*100, alpha=0.5)

plt.yticks([0,1])

plt.xticks(list(bigger_df.Occupation.unique()))

plt.title("Consumidores com compras acima de $9000 por estado civil e ocupação")

plt.ylabel("Estado Civil")

plt.xlabel("Ocupação")

plt.show()