import sqlite3



banco = sqlite3.connect('/kaggle/input/soccer/database.sqlite')
import matplotlib.pyplot as plt

import numpy as np

import matplotlib.patheffects as path_effects

import pandas as pd

import seaborn as sns





plt.rcParams['figure.figsize'] = (11,5)

plt.style.use('bmh')
def gerar_bloxplot(linhas, tabela, condicao, titulo='NULL', gerar_graf=False):

  a = banco.execute("select %s from %s where %s"%(linhas, tabela, condicao))

  gols = []

  dados = a.fetchall()

  for linha in dados:

    gols.append(linha[0])



  if gerar_graf:

    npgols = np.array(gols)

    plt.text(0.7, 0.7, 'Média de gols: %.2f'%(np.mean(npgols)))

    plt.boxplot(gols, 0, 'rs',0)

    plt.title('%s'%titulo)

    plt.show()

  else:

    return gols
campeoes = ['10260', '8455', '10260', '8456', '10260', '8456', '8455', '8197']

campanhas_casa = {'2008/2009': [], '2009/2010': [], '2010/2011': [], '2011/2012': [], '2012/2013': [],

                  '2013/2014': [], '2014/2015': [], '2015/2016': []}

campanhas_fora = {'2008/2009': [], '2009/2010': [], '2010/2011': [], '2011/2012': [], '2012/2013': [],

                  '2013/2014': [], '2014/2015': [], '2015/2016': []}

j = 0

for i in campanhas_casa.keys():

  campanhas_casa[i] = gerar_bloxplot("home_team_goal", "match", "home_team_api_id = '%s' and season = '%s'"%(campeoes[j], i))

  campanhas_fora[i] = gerar_bloxplot("away_team_goal", "match", "away_team_api_id = '%s' and season = '%s'"%(campeoes[j], i))

  j += 1

data_casa = pd.DataFrame(campanhas_casa)

data_fora = pd.DataFrame(campanhas_fora)

data_casa.boxplot()

plt.title("Box plot dos campeões em casa")

plt.xlabel("Ano")

plt.ylabel('Gols')



plt.style.use('seaborn-paper')

plt.rcParams['figure.figsize'] = (10,6)

plt.show()
data_fora.boxplot()

plt.title("Box plot dos campeões fora de casa")

plt.xlabel("Ano")

plt.ylabel('Gols')

plt.rcParams['figure.figsize'] = (10,6)

plt.show()
data_casa.hist(color='#708090')

plt.rcParams['figure.figsize'] = (10,9)

plt.show()
data_fora.hist(color='#708090')

plt.rcParams['figure.figsize'] = (10,9)

plt.xlabel('Gols feitos')

plt.ylabel('Jogos')

plt.show()
golsC_temp = []

temp = []

for i in campanhas_casa: 

  golsC_temp.append(sum(campanhas_casa[i]))

  temp.append(i)

#pallete = sns.color_palette("Blues", 10)

pallete = sns.cubehelix_palette(8, 2, 0.4, 0.90, 0.8, 0.6)



sns.barplot(temp, golsC_temp, palette=pallete)

plt.rcParams['figure.figsize'] = (10,5)

plt.xlabel('Temporadas')

plt.ylabel('Gols feitos em casa')

plt.show()
golsF_temp = []

temp = []

for i in campanhas_fora: 

  golsF_temp.append(sum(campanhas_fora[i]))

  temp.append(i)

#pallete = sns.color_palette("Blues", 10)

pallete = sns.cubehelix_palette(8, 2, 0.4, 0.90, 0.8, 0.6)



sns.barplot(temp, golsF_temp, palette=pallete)

plt.rcParams['figure.figsize'] = (10,5)

plt.xlabel('Temporadas')

plt.ylabel('Gols feitos fora de casa')

plt.show()
medias_casa  = {'Media do campeão': 'Média casa'}

medias_fora  = {'Media do campeão': 'Média fora'}

medias_total = {'Media do campeão': 'Média total'}

j = 0

for i in campanhas_casa:

  medias_casa[i] = data_casa[i].mean()

  medias_fora[i] = data_fora[i].mean()

  medias_total[i] = (golsC_temp[j] + golsF_temp[j])/(len(campanhas_casa[i])*2)

  j+=1

data_media = pd.DataFrame([medias_casa, medias_fora, medias_total])

data_media.head()
vices = ['8650', '10260', '8455', '10260', '8456', '8650', '8456', '9825']

terceiros = ['8455', '9825', '8456', '9825', '8455', '8455', '9825', '8586']

j = 0

vices_total = []

teceriros_total = []

for i in campanhas_casa.keys():

  vices_total.append(gerar_bloxplot("home_team_goal", "match", "home_team_api_id = '%s' and season = '%s'"%(vices[j], i))

  + gerar_bloxplot("away_team_goal", "match", "away_team_api_id = '%s' and season = '%s'"%(vices[j], i)))

  teceriros_total.append(gerar_bloxplot("home_team_goal", "match", "home_team_api_id = '%s' and season = '%s'"%(terceiros[j], i)) 

  + gerar_bloxplot("away_team_goal", "match", "away_team_api_id = '%s' and season = '%s'"%(terceiros[j], i)))

  j += 1

medias_total_v = []

medias_total_t = []

vices_total

for i in range(len(campanhas_casa.keys())):

  medias_total_v.append(sum(vices_total[i])/len(vices_total[i]))

  medias_total_t.append(sum(teceriros_total[i])/len(teceriros_total[i]))



data_total = pd.DataFrame({'Temporada': list(campanhas_casa.keys()), 'Campeao': list(data_media.iloc[2])[1::], 'Vice': medias_total_v, 

              'Terceiro': medias_total_t})

data_total
fig, ax = plt.subplots()



line1, = ax.plot(data_total['Temporada'], data_total['Campeao'], label='Campeão')

line2, = ax.plot(data_total['Temporada'], data_total['Vice'], label='2º colocado')

line3, = ax.plot(data_total['Temporada'], data_total['Terceiro'], label='3º colocado')



ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=8, mode="expand", borderaxespad=0.)

plt.xlabel('Temporada')

plt.ylabel('Media total de gols')

plt.show()
gerar_bloxplot("home_team_goal", "match", "home_team_api_id = '10260' and season = '2008/2009'",\

               "Gols do campeão Manchester United em casa 2008/2009", True)
gerar_bloxplot("away_team_goal", "match", "away_team_api_id = '10260' and season = '2008/2009'",\

               "Gols do campeão Manchester United fora de casa 2008/2009", True)
gerar_bloxplot("home_team_goal", "match", "home_team_api_id = '8455' and season = '2009/2010'",\

               "Gols do campeão Chelsea em casa 2009/2010", True)
gerar_bloxplot("away_team_goal", "match", "away_team_api_id = '8455' and season = '2009/2010'",\

               "Gols do campeão Chelsea fora de casa 2009/2010", True)
gerar_bloxplot("home_team_goal", "match", "home_team_api_id = '10260' and season = '2010/2011'",\

               "Gols do campeão Manchester United em casa 2010/2011", True)
gerar_bloxplot("away_team_goal", "match", "away_team_api_id = '10260' and season = '2010/2011'",\

               "Gols do campeão Manchester United fora de casa 2010/2011", True)
gerar_bloxplot("home_team_goal", "match", "home_team_api_id = '8456' and season = '2011/2012'", \

               "Placares do campeão Manchester City em casa", True)
gerar_bloxplot("away_team_goal", "match", "away_team_api_id = '8456' and season = '2011/2012'", \

               "Placares do campeão Manchester City fora de casa", True)
gerar_bloxplot("home_team_goal", "match", "home_team_api_id = '10260' and season = '2012/2013'", \

              "Placares do campeão Manchester United em casa", True)
gerar_bloxplot("away_team_goal", "match", "away_team_api_id = '10260' and season = '2012/2013'", \

               "Placares do campeão Manchester United em casa", True)
def numero_em_cima(rects, ax):

    for rect in rects:

        height = rect.get_height()

        ax.annotate('{}'.format(height),

                    xy=(rect.get_x() + rect.get_width() / 2, height),

                    xytext=(0, 3),  

                    textcoords="offset points",

                    ha='center', va='bottom')

        

def gerar_grouped_bar(linhas1, tabela1, condicao1, linhas2, tabela2, condicao2, tibar1, tibar2, rep,titulo):

  a = banco.execute("select %s from %s where %s"%(linhas1, tabela2, condicao2))

  golsCa = []

  dados_a = a.fetchall()

  for linha in dados_a:

    golsCa.append(linha[0])



  b = banco.execute("select %s from %s where %s"%(linhas2, tabela2, condicao2))

  golsFo = []

  dados_b = b.fetchall()

  for linha in dados_b:

    golsFo.append(linha[0])



  jogo = ['J%d'%i for i in range(1, len(golsCa)+1)]



  x = np.arange(len(jogo))  

  width = 0.35  



  fig, ax = plt.subplots()

  rects1 = ax.bar(x - width/2, golsCa, width, label='%s'%tibar1)

  rects2 = ax.bar(x + width/2, golsFo, width, label='%s'%tibar2)



  ax.set_ylabel('%s'%(rep))

  ax.set_title('%s'%(titulo))

  ax.set_xticks(x)

  ax.set_xticklabels(jogo)

  ax.legend()



  numero_em_cima(rects1, ax)

  numero_em_cima(rects2, ax)



  fig.tight_layout()



  plt.show()
gerar_grouped_bar("home_team_goal", "match", "home_team_api_id = '10260' and season = '2008/2009'", \

                  "away_team_goal", "match", "home_team_api_id = '10260' and season = '2008/2009'", \

                  "Gols Casa", "Gols Fora", "Gols", "Placares do campeão Manchester United em casa")
gerar_grouped_bar("home_team_goal", "match", "away_team_api_id = '10260' and season = '2008/2009'", \

                  "away_team_goal", "match", "away_team_api_id = '10260' and season = '2008/2009'", \

                  "Gols Casa", "Gols Fora", "Gols", "Placares do campeão Manchester United fora de casa")
gerar_grouped_bar("home_team_goal", "match", "home_team_api_id = '8455' and season = '2009/2010'", \

                  "away_team_goal", "match", "home_team_api_id = '8455' and season = '2009/2010'", \

                  "Gols Casa", "Gols Fora", "Gols", "Placares do campeão Chelsea em casa")
gerar_grouped_bar("home_team_goal", "match", "away_team_api_id = '8455' and season = '2009/2010'", \

                  "away_team_goal", "match", "away_team_api_id = '8455' and season = '2009/2010'", \

                  "Gols Casa", "Gols Fora", "Gols", "Placares do campeão Chealsea fora de casa")
gerar_grouped_bar("home_team_goal", "match", "home_team_api_id = '10260' and season = '2010/2011'", \

                  "away_team_goal", "match", "home_team_api_id = '10260' and season = '2010/2011'", \

                  "Gols Casa", "Gols Fora", "Gols", "Placares do campeão Manchester United em casa")
gerar_grouped_bar("home_team_goal", "match", "away_team_api_id = '10260' and season = '2010/2011'", \

                  "away_team_goal", "match", "away_team_api_id = '10260' and season = '2010/2011'", \

                  "Gols Casa", "Gols Fora", "Gols", "Placares do campeão Manchester United fora de casa")
gerar_grouped_bar("home_team_goal", "match", "home_team_api_id = '8456' and season = '2011/2012'", \

                  "away_team_goal", "match", "home_team_api_id = '8456' and season = '2011/2012'", \

                  "Gols Casa", "Gols Fora", "Gols", "Placares do campeão Manchester City em casa")
gerar_grouped_bar("home_team_goal", "match", "away_team_api_id = '8456' and season = '2011/2012'", \

                  "away_team_goal", "match", "away_team_api_id = '8456' and season = '2011/2012'", \

                  "Gols Casa", "Gols Fora", "Gols", "Placares do campeão Manchester City fora de casa")
gerar_grouped_bar("home_team_goal", "match", "home_team_api_id = '10260' and season = '2012/2013'", \

                  "away_team_goal", "match", "home_team_api_id = '10260' and season = '2012/2013'", \

                  "Gols Casa", "Gols Fora", "Gols", "Placares do campeão Manchester United em casa")
gerar_grouped_bar("home_team_goal", "match", "away_team_api_id = '10260' and season = '2012/2013'", \

                  "away_team_goal", "match", "away_team_api_id = '10260' and season = '2012/2013'", \

                  "Gols Casa", "Gols Fora", "Gols", "Placares do campeão Manchester United fora de casa")
gerar_grouped_bar("home_team_goal", "match", "home_team_api_id = '8456' and season = '2013/2014'", \

                  "away_team_goal", "match", "home_team_api_id = '8456' and season = '2013/2014'", \

                  "Gols Casa", "Gols Fora", "Gols", "Placares do campeão Manchester City em casa")
gerar_grouped_bar("home_team_goal", "match", "away_team_api_id = '8456' and season = '2013/2014'", \

                  "away_team_goal", "match", "away_team_api_id = '8456' and season = '2013/2014'", \

                  "Gols Casa", "Gols Fora", "Gols", "Placares do campeão Manchester City fora de casa")
gerar_grouped_bar("home_team_goal", "match", "home_team_api_id = '8455' and season = '2014/2015'", \

                  "away_team_goal", "match", "home_team_api_id = '8455' and season = '2014/2015'", \

                  "Gols Casa", "Gols Fora", "Gols", "Placares do campeão Chealsea em casa")
gerar_grouped_bar("home_team_goal", "match", "away_team_api_id = '8455' and season = '2014/2015'", \

                  "away_team_goal", "match", "away_team_api_id = '8455' and season = '2014/2015'", \

                  "Gols Casa", "Gols Fora", "Gols", "Placares do campeão Chealsea fora de casa")
gerar_grouped_bar("home_team_goal", "match", "home_team_api_id = '8197' and season = '2015/2016'", \

                  "away_team_goal", "match", "home_team_api_id = '8197' and season = '2015/2016'", \

                  "Gols Casa", "Gols Fora", "Gols", "Placares do campeão Leicester City em casa")
gerar_grouped_bar("home_team_goal", "match", "away_team_api_id = '8197' and season = '2015/2016'", \

                  "away_team_goal", "match", "away_team_api_id = '8197' and season = '2015/2016'", \

                  "Gols Casa", "Gols Fora", "Gols", "Placares do campeão Leicester City fora de casa")