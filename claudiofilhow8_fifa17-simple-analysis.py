%matplotlib inline 

import pandas as pd 

import matplotlib.pyplot as plt

import numpy as np 

import seaborn as sns

%config InlineBackend.rc={'figure.figsize': (20,15)}
fifa = pd.read_csv('../input/FullData.csv') #Lendo o dataframe
fifa.head() # 5 Primeiras linhas



#Descrição do dataframe, para coletar maior e menor rating do fifa

fifa['Rating'].describe()

#Jogador que possui maior precisão em cobrança de falta 

falta_acuracia=fifa['Freekick_Accuracy'].describe() 

falta_acuracia['max'] # Coleta o valor do max encontrado no describe acima

fifa.loc[fifa['Freekick_Accuracy'] == falta_acuracia['max'] ] # Utiliza o loc para encontrar este valor e 

                                                              #trazer o registro que o contem
#Verificando se o melhor goleiro é realmente o que possui a maior soma das habilidades destinadas a posição.



somahabilidades=(fifa['GK_Positioning'] + fifa['GK_Diving'] + fifa['GK_Kicking']

                    + fifa['GK_Handling'] + fifa['GK_Reflexes'])



melhorgk=somahabilidades.argmax() #Todos jogadores tiveram estas habilidades somadas, 

                                  # então o argmax() traz o que tem a maior soma 







fifa.loc[melhorgk] # Localizando pela maior soma 







# Maior, media e menor rating por posição, podendo verificar qual possui mais craques. 



rating=(fifa.groupby('Club_Position') #Agrupa por posição

        .aggregate({'Rating':[min,np.mean,max]}) #Agrega o min media e max do Rating

        .plot(kind='bar',fontsize=20) ) #Plota os dados 



plt.ylabel('Rating',fontsize=30) #Titulo eixo y 

plt.xlabel('Position',fontsize=30) #Titulo eixo x 

plt.title('Min/Mean/Max by Position',fontsize=40) # Titulo do grafico 



plt.legend(loc=1, prop={'size': 20}) #Tamanho e localização da legenda 



plt.show() #Mostra o grafico.





#plt.savefig('grafico1.png') #Para salvar a figura no diretório
#Tabela que foi gerada pelo groupby + aggregate da celula acima. 



fifa.groupby('Club_Position').aggregate({'Rating':[min,np.mean,max]})

fifa.groupby('Name').aggregate({'Rating':[max]}) #Teste de groupby, coleta o rating de todos jogadores. 



attributes = ['Ball_Control','Dribbling','Marking','Aggression','Reactions', 'Attacking_Position',

       'Interceptions', 'Vision', 'Composure', 'Crossing', 'Short_Pass',

       'Long_Pass', 'Acceleration', 'Speed', 'Stamina', 'Strength', 'Balance',

       'Agility', 'Jumping', 'Heading', 'Shot_Power', 'Finishing',

       'Long_Shots', 'Curve', 'Freekick_Accuracy', 'Penalties', 'Volleys',

       'GK_Positioning', 'GK_Diving', 'GK_Kicking', 'GK_Handling',

       'GK_Reflexes'] #São todos atributos dos jogadores. 





#Funções abaixo plotam estes atributos de cada jogador definido pela posição que estão no dataset



fig, ax = plt.subplots()

messi=fifa.loc[1][attributes].plot(kind='bar',color='r',label='Messi',fontsize=20)

cr7 = fifa.loc[0][attributes].plot(kind='bar',label='CR7')

zlatan = fifa.loc[8][attributes].plot(kind='bar',color='y',label="Ibra")

ax.set_ylabel('Rate', fontsize=30)

plt.legend(loc=1, prop={'size': 20})

plt.xlabel('Skills',fontsize=20)

plt.title('Skills Properties',fontsize=20)





 

plt.show() #Mostra o grafico.



#Mesma representação do grafico acima porem em pontos, para melhor visualização. 



fig, ax = plt.subplots()



#Encontra o Messi pelo indice [1] que é a posição dele no dataframe 

messi=fifa.loc[1][attributes].plot(color='r', #Define a cor de seus pontos

                                   label='Messi', #Texto da legenda atribuida 

                                   ls='none',#ls=none deixa os pontos sem serem ligados por alguma linha

                                   marker='s', #define o marcador s=Square , quadrado 

                                   ms=8,#Tamanho do simbolo 

                                   fontsize=20) #Tamanho da fonte escrita na legenda. 



cr7 = fifa.loc[0][attributes].plot(color='b',label='CR7',ls='none',marker='o',ms=8) #Idem ao anterior

zlatan = fifa.loc[8][attributes].plot(color='y',label="Ibra",ls='none',marker='^',ms=8) # Idem ao anterior

x_axis = range(len(attributes))#tamanho do eixo X , range dos valores 

ax.set_ylabel('Rate', fontsize=50) #Nome e Tamanho do titulo eixo y

ax.set_xlabel('Skills', fontsize=50) #Nome e Tamanho do titulo eixo x 

ax.set_xticks(x_axis)#Define os valores do eixo X, fazendo todos aparecerem neste eixo.

ax.set_xticklabels(attributes,rotation='vertical',fontsize=30) #atribui a legenda os nomes dos vertices 

plt.legend(loc=1, prop={'size': 20}) #Tamanho da legenda

plt.title('Skills Properties - Dots',fontsize=30) # Titulo e tamanho do titulo





plt.show() #Mostra o grafico.



#Qual pais tem o maior numero de jogadores no FIFA? 

#O grafico abaixo mostra as 5 nacionalidades mais presentes no FIFA17.



Nationality_count=fifa.groupby(['Nationality']).count() #Agrupando por nacionalidade e contando

Top5Nat=Nationality_count.sort_values(['Name']).tail() #Alinha pela coluna 'names' que é apenas uma contagem,

                                                       #e pega os 5 ultimos valores

Top5Nat.drop(Top5Nat.columns[1:52],axis=1,inplace=True)

Top5Nat.columns = ["Players Count"]





Colors=['y','b','c','g','r']



Top5Nat.plot(kind='bar',fontsize=20,color=Colors,legend=None)

plt.title('Players by Country',fontsize=40)

plt.xlabel('Country',fontsize=30)







plt.show() #Mostra o grafico.



    

#Jogadores pela nacionalidade 



playerbycountry=(fifa.groupby(['Nationality', 'Name'])

                 .aggregate({'Rating':[max]}))#Agrupa por nacionalidade e nome e traz o rating max. 

                    

playerbycountry
#Top 5 por país, traz a lista de 5 jogadores com mais rating de cada pais. 

nationality=fifa.Nationality.unique() #Transforma dataframe em uma unica lista contendo as nacionalidades



dataframe=[] #cria um dataframe



for i in nationality: #Procura e separa do dataframe playersbycountry com os 5 ultimos  q tem rating max

    data=playerbycountry.loc[i].sort_values(max).tail() #A variavel i escreve o nome dos paises e localiza 

    dataframe.append(data) #dataframe anexa a ele o valor. 

    

print(dataframe)  #imprime a lista. 

#Top 5 jogadores por pais 



(playerbycountry.loc['Brazil'] #Localiza "Brazil" no dataframe playerbycountry 

 .sort_values(max) #Ordena por valor maxima, ficando os minimos em cima e maximos abaixo. 

 .tail()#Coleta os 5 ultimos

 .plot(kind='bar',fontsize=20,color=Colors,legend=None)) #Plota os 5 ultimos jogadores do Brazil com seus nomes

                                                         #e Rating





plt.xlabel('Name',fontsize=30) #legenda eixo x

plt.ylabel('Rating',fontsize=30) #legenda eixo y

plt.title('Country Top 5 Players',fontsize=40) #Titulo grafico 



                                                       

plt.show() #Mostra o grafico.




