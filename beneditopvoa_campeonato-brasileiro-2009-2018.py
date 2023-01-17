#Importando a biblioteca pandas e criando uma variável para receber o dataset em csv





import pandas as pd

v_dataset = pd.read_csv("../input/campeonato-braileiro-20092018/Tabela_Clubes.csv")
#Analisando o conjunto de dados 



v_dataset.head()
#excluindo as últimas colunas 



v_dataset.drop(columns=["Unnamed: 13","Unnamed: 14","Unnamed: 15","Unnamed: 16"], inplace=True)





#p/estudo -> a função inplace=True valida a alteração p/o conjunto 
# renomeando algumas colunas



v_dataset.rename(columns={"Pos.":"Posição","Qtd_Jogadores":"Elenco","Valor_total":"Valor_Elenco","Media_Valor":"Media_Valor_Elenco"}, inplace=True)
# dividindo a coluna Gols F/S e acrescentando cada um em coluna diferente



divisao = v_dataset["GolsF/S"].str.split(":") #dividindo pelo delimitador ':'

#divisao.head()

gols_feitos = divisao.str.get(0) #pegando a pos 0 do que foi dividido e colocando em uma variavel

gols_sofridos = divisao.str.get(1) #pegando a pos 1 do que foi dividido e colocando em uma variavel



v_dataset["Gols_Feitos"] = gols_feitos # passando o resultado da variável para uma coluna

v_dataset["Gols_Sofridos"] = gols_sofridos 



v_dataset.drop(columns=["GolsF/S"], inplace=True) # excluindo a coluna Gols F/S pq ela foi dividida



v_dataset.head()
#verificando o tipo dos campos

v_dataset.info()
#alterando os tipos de alguns campos



v_dataset["Valor_Elenco"] = v_dataset["Valor_Elenco"].astype("float32")

v_dataset["Media_Valor_Elenco"] = v_dataset["Valor_Elenco"].astype("float32")

v_dataset["Idade_Media"] = v_dataset["Valor_Elenco"].astype("float32")

v_dataset["Gols_Feitos"] = v_dataset["Gols_Feitos"].astype("int64")

v_dataset["Gols_Sofridos"] = v_dataset["Gols_Sofridos"].astype("int64")



v_dataset.info()
# verificando dados estatisticos do conj



v_dataset.describe()

#a data está errada, por isso irei somar +1 ano no campo Ano



v_dataset["Ano"] = v_dataset["Ano"]+1
#checando o conj novamente

v_dataset.describe()
vitorias = v_dataset[{"Clubes","Vitorias"}] #separando duas colunas do dataset e armazenando na variavel vitorias

vitorias = vitorias.groupby("Clubes").sum() #somando esses campos e agrupando por clubes

vitorias = vitorias.sort_values("Vitorias", ascending = False) #ordenando o que foi feito anteriormente por vitorias



vitorias.head(5) #mostrando os 5 primeiros
med = v_dataset[{"Ano","Clubes","Gols_Feitos"}]

med = med.groupby("Ano").mean().sort_values("Gols_Feitos", ascending = False).rename(columns={"Gols_Feitos": 

                                                                                             "Media_GP"})



med
#Times menos vitoriosos (em gráfico)



graf_vit = v_dataset[{"Clubes","Vitorias"}] #tbm separei duas colunas do dataset em uma variavel 



#plotando um gráfico dos times menos vitoriosos no CB entre os anos

graf_vit = graf_vit.groupby("Clubes").sum().sort_values("Vitorias", ascending = True).head(12).plot(kind='barh', color="orange")



money = v_dataset[{"Clubes", "Ano", "Valor_Elenco"}] # separei algumas colunas para utilizar nesta análise

money = money[{"Clubes","Valor_Elenco","Ano"}].sort_values("Valor_Elenco", ascending = False) # fiz a análise ordenando de forma crescente



money.head()
pont = v_dataset[{"Ano","Clubes","Gols_Feitos"}]

pont = pont.sort_values("Gols_Feitos", ascending = False)



pont.head()
defvas = v_dataset[{"Ano","Clubes","Gols_Sofridos"}]

defvas = defvas.sort_values("Gols_Sofridos", ascending = False)



defvas.head()
win = v_dataset[{"Clubes","Posição"}] 

#win = win[{"Clubes","Posição"}].where(win["Posição"]==1)

#win.rename(columns={"Posição":"Qtd de Títulos"})

#win = win.rename(columns={"Posição":"Qtd de Titulos"}, inplace=True)



win = win[win.Posição ==1].groupby("Clubes").count().sort_values("Posição",ascending = False).rename(

    columns={"Posição":"Títulos"})





win


