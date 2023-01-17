import pandas as pd
df_eleitorado = pd.read_csv('../input/BR_eleitorado_2016_municipio.csv', delimiter=',')
df_eleitorado.head(10)
x = df_eleitorado[df_eleitorado['uf'] == 'BA'].nlargest(15,'total_eleitores')
#fem = x['gen_feminino']
#mas = x['gen_masculino']
#print(df_eleitorado[df_eleitorado['uf'] == 'BA'].groupby(by=['nome_municipio','gen_feminino'])['gen_feminino'].sum().head(15))
#print(df_eleitorado[df_eleitorado['uf'] == 'BA'].groupby(by=['nome_municipio','gen_masculino'])['gen_masculino'].sum().head(15))
#dr = (df_eleitorado.groupby(by='Regiao').count()['cod_municipio_tse']).reset_index().rename(columns={'index':'Regiao', 'cod_municipio_tse' : 'Frequencia_absoluta'})
#df_feminino = (df_eleitorado.groupby(by='Regiao').count()['cod_municipio_tse']).reset_index().rename(columns={'index':'Regiao', 'cod_municipio_tse' : 'Frequencia_absoluta'})
x
all_data = [x['gen_masculino'], x['gen_feminino']]
plt.violinplot(all_data,showmeans=True, showmedians=False) #default
plt.show()

y = df_eleitorado[df_eleitorado['uf'] == 'MG'].nlargest(15,'total_eleitores')
all_data2 = [y['f_18_20'], y['f_21_24'], y['f_70_79'], y['f_sup_79']]
plt.violinplot(all_data2,showmeans=True, showmedians=False) #default
plt.show()

