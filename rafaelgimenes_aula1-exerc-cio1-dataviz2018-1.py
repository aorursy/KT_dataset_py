import pandas as pd
df = pd.read_csv('../input/anv.csv', delimiter=',')

listaColunas=(list(df))
listaTipoVar = ['Qualitativa Nominal','Qualitativa Ordinal','Quantitativa Discreta','Quantitativa Continua']

resposta = [[listaColunas[3], listaTipoVar[0]], #tipo veiculo
            [listaColunas[4], listaTipoVar[0]], #fabricante
            [listaColunas[10], listaTipoVar[1]], #categoria pmd
            [listaColunas[13], listaTipoVar[0]], #pais fabricante
            [listaColunas[22],listaTipoVar[1]], #nivel dano
            [listaColunas[23],listaTipoVar[2]], #total fatalidades
            [listaColunas[24],listaTipoVar[1]] # dia extração
            ] 
#variáveis relacionadas a tempo são contínuas, mas podem ser discretas pois não há perdas -- (discretização)
resposta = pd.DataFrame(resposta, columns=["Variavel", "Classificação"])
resposta

tab_aeronave_tipo_veiculo = df["aeronave_tipo_veiculo"].value_counts()
tab_aeronave_tipo_veiculo = tab_aeronave_tipo_veiculo.to_frame().reset_index()
tab_aeronave_tipo_veiculo.columns = ['Tipo','Qtde']
tab_aeronave_tipo_veiculo
tab_aeronave_fabricante = df["aeronave_fabricante"].value_counts()
tab_aeronave_fabricante = tab_aeronave_fabricante.to_frame().reset_index()
tab_aeronave_fabricante.columns = ['Fabricante','Qtde']
tab_aeronave_fabricante.head()
tab_aeronave_pmd_categoria = df["aeronave_pmd_categoria"].value_counts()
tab_aeronave_pmd_categoria = tab_aeronave_pmd_categoria.to_frame().reset_index()
tab_aeronave_pmd_categoria.columns = ['Categoria','Qtde']
tab_aeronave_pmd_categoria.head()
tab_aeronave_pais_fabricante = df["aeronave_pais_fabricante"].value_counts()
tab_aeronave_pais_fabricante = tab_aeronave_pais_fabricante.to_frame().reset_index()
tab_aeronave_pais_fabricante.columns = ['Pais Fabriante','Qtde']
tab_aeronave_pais_fabricante.head()
tab_aeronave_nivel_dano = df["aeronave_nivel_dano"].value_counts()
tab_aeronave_nivel_dano = tab_aeronave_nivel_dano.to_frame().reset_index()
tab_aeronave_nivel_dano.columns = ['Dano','Qtde']
tab_aeronave_nivel_dano
tab_aeronave_dia_extracao  = df["aeronave_dia_extracao"].value_counts()
tab_aeronave_dia_extracao = tab_aeronave_dia_extracao.to_frame().reset_index()
tab_aeronave_dia_extracao.columns = ['Dia Extração','Qtde']
tab_aeronave_dia_extracao.head()
import matplotlib.pyplot as plt
tab_aeronave_nivel_dano.plot(kind='bar',x='Dano',y='Qtde',legend=False)
plt.style.use('fast')
plt.title('Quantidade X Dano')
plt.ylabel("Quantidade")
plt.show()

df.groupby('aeronave_tipo_veiculo')['total_fatalidades'].nunique().nlargest(3).plot(kind='pie',legend=False)
plt.title('Top 3 tipos de veiculos que mais mataram')
plt.show()
plt.bar(tab_aeronave_pmd_categoria['Categoria'], tab_aeronave_pmd_categoria['Qtde'])
plt.title('Quantidade X Categoria')
plt.xticks(rotation='vertical')
plt.xlabel("Categoria")
plt.ylabel("Quantidade")
plt.show()
grp1 = df[df['aeronave_nivel_dano']=='DESTRUÍDA']
grp1 = grp1.groupby(['aeronave_pais_fabricante']).size().to_frame().reset_index()
grp1.columns = ['Pais','Qtde']
grp1 = grp1.sort_values(by=['Qtde'],ascending=False)
plt.title('Quantidade Destruidos X Pais Fabricante')
plt.bar(grp1['Pais'],grp1['Qtde'])
plt.xticks(rotation='vertical')
plt.xlabel("Pais Fabricante")
plt.style.use('fast')
plt.show()

grp3 = df[ ((df['aeronave_ano_fabricacao']>=1998) & (df['aeronave_fabricante']=='EMBRAER'))]
grp3 = grp3.groupby(['aeronave_ano_fabricacao']).size().nlargest(5).to_frame().reset_index()
grp3.columns = ['Ano','Qtde']
grp3 = grp3.sort_values('Ano')


plt.xticks(rotation='vertical')
plt.style.use('fast')
plt.xlabel("Anos")
plt.ylabel("Quantidade Fabricadas")
plt.plot(grp3['Ano'],grp3['Qtde'],'--')
plt.show()

df.head()