import numpy as np                       
import pandas as pd                      
import matplotlib.pyplot as plt          
from mpl_toolkits.basemap import Basemap # biblioteca para plotar mapas


#padroniza o tamanho dos quadros (polegadas)
plt.rcParams['figure.figsize']=(9,7)

#Leitura do arquivo
data = pd.read_csv('../input/meteorite-landings.csv', sep=',')

#Deixando somente os dados validos
data = data[(data.reclat != 0.0) & (data.reclong != 0.0)]
data = data[(data.reclong <= 180.0) & (data.reclong >= -180.0)]
data = data[(data.year >= 860 ) & (data.year <= 2016)]

print('Quantidade de entradas: {}'.format(len(data)))

map = Basemap(projection='cyl',llcrnrlat=-90,llcrnrlon=-180,urcrnrlat=90,urcrnrlon=180,resolution='c')
#valores para visualização geral -34/-76/6/-30

map.etopo() 
map.drawcountries() #desenha as linhas dos países
map.drawcoastlines(linewidth=0.5) #desenha as linhas das costas maritimas
#map.drawstates(linewidth=0.6) #desenha as linhas dos estados

#plota os dados no mapa
#edgecolor - cor do background
#color - cor do ponto k=preto
#marker - por default é um ponto
#alpha - tranparencia do ponto (0 - 1)

map.scatter(data.reclong,data.reclat,edgecolor='none',color='k',alpha=0.8, marker='.')
plt.title('Meteoritos no Mundo', fontsize=15)
#de subclasses para classes gerais(aerolitos(rochosos), sideritos(metalicos), siderolitos(mistos))
data.recclass.replace(to_replace=['Pallasite', 'Pallasite, PES','Pallasite, PMG',
           'Pallasite, PMG-an', 'Pallasite, ungrouped',
           'Pallasite?','Mesosiderite', 'Mesosiderite-A','Mesosiderite-A1',
           'Mesosiderite-A2', 'Mesosiderite-A3','Mesosiderite-A3/4',
           'Mesosiderite-A4', 'Mesosiderite-B','Mesosiderite-B1',
           'Mesosiderite-B2', 'Mesosiderite-B4','Mesosiderite-C',
           'Mesosiderite-C2', 'Mesosiderite-an','Mesosiderite?'],value='Siderolitos',inplace=True)

data.recclass.replace(to_replace=['Iron, IAB complex', 'Iron, IAB-MG','Iron, IAB-an', 'Iron, IAB-sHH',
           'Iron, IAB-sHL', 'Iron, IAB-sLH','Iron, IAB-sLL', 'Iron, IAB-sLM',
           'Iron, IAB-ung', 'Iron, IAB?','Iron, IIE',
           'Iron, IIE-an', 'Iron, IIE?','Iron','Iron?','Relict iron','Iron, ungrouped', 'Iron, IC', 'Iron, IC-an', 'Iron, IIAB', 'Iron, IIAB-an',
           'Iron, IIC', 'Iron, IID', 'Iron, IID-an','Iron, IIF', 'Iron, IIG',
           'Iron, IIIAB', 'Iron, IIIAB-an', 'Iron, IIIAB?', 'Iron, IIIE',
           'Iron, IIIE-an', 'Iron, IIIF', 'Iron, IVA', 'Iron, IVA-an',
           'Iron, IVB',  ],value='Sideritos',inplace=True)

data.recclass.replace(to_replace=['Acapulcoite', 'Acapulcoite/Lodranite', 'Acapulcoite/lodranite',
           'Lodranite','Lodranite-an','Winonaite','Achondrite-prim', 'Angrite', 'Aubrite','Aubrite-an','Ureilite', 'Ureilite-an','Ureilite-pmict',
           'Brachinite','Diogenite', 'Diogenite-an', 'Diogenite-olivine', 'Diogenite-pm',
           'Eucrite', 'Eucrite-Mg rich', 'Eucrite-an', 'Eucrite-br','Eucrite-cm',
           'Eucrite-mmict', 'Eucrite-pmict', 'Eucrite-unbr','Howardite', 'Lunar', 'Lunar (anorth)', 'Lunar (bas. breccia)',
           'Lunar (bas/anor)', 'Lunar (bas/gab brec)', 'Lunar (basalt)',
           'Lunar (feldsp. breccia)', 'Lunar (gabbro)', 'Lunar (norite)', 'Martian', 'Martian (OPX)','Martian (chassignite)', 'Martian (nakhlite)',
           'Martian (shergottite)', 'C','C2','C4','C4/5','C6','C1-ung', 'C1/2-ung','C2-ung',
           'C3-ung', 'C3/4-ung','C4-ung','C5/6-ung',
           'CB', 'CBa', 'CBb', 'CH/CBb', 'CH3', 'CH3 ', 'CI1', 'CK', 'CK3',
           'CK3-an', 'CK3.8', 'CK3/4', 'CK4', 'CK4-an', 'CK4/5', 'CK5',
           'CK5/6', 'CK6', 'CM', 'CM-an', 'CM1', 'CM1/2', 'CM2', 'CM2-an',
           'CO3', 'CO3 ', 'CO3.0', 'CO3.1', 'CO3.2', 'CO3.3', 'CO3.4', 'CO3.5',
           'CO3.6', 'CO3.7', 'CO3.8', 'CR', 'CR-an', 'CR1', 'CR2', 'CR2-an',
           'CV2', 'CV3', 'CV3-an', 'OC', 'OC3','H', 'H(5?)', 'H(?)4', 'H(L)3', 'H(L)3-an', 'H-an','H-imp melt',
           'H-melt rock', 'H-metal', 'H/L3', 'H/L3-4', 'H/L3.5',
           'H/L3.6', 'H/L3.7', 'H/L3.9', 'H/L4', 'H/L4-5', 'H/L4/5', 'H/L5',
           'H/L6', 'H/L6-melt rock', 'H/L~4', 'H3', 'H3 ', 'H3-4', 'H3-5',
           'H3-6', 'H3-an', 'H3.0', 'H3.0-3.4', 'H3.1', 'H3.10', 'H3.2',
           'H3.2-3.7', 'H3.2-6', 'H3.2-an', 'H3.3', 'H3.4', 'H3.4-5',
           'H3.4/3.5', 'H3.5', 'H3.5-4', 'H3.6', 'H3.6-6', 'H3.7', 'H3.7-5',
           'H3.7-6', 'H3.7/3.8', 'H3.8', 'H3.8-4', 'H3.8-5', 'H3.8-6',
           'H3.8-an', 'H3.8/3.9', 'H3.8/4', 'H3.9', 'H3.9-5', 'H3.9-6',
           'H3.9/4', 'H3/4', 'H4', 'H4 ', 'H4(?)', 'H4-5', 'H4-6', 'H4-an',
           'H4/5', 'H4/6', 'H5', 'H5 ', 'H5-6', 'H5-7', 'H5-an',
           'H5-melt breccia', 'H5/6', 'H6', 'H6 ', 'H6-melt breccia', 'H6/7',
           'H7', 'H?','H~4', 'H~4/5', 'H~5', 'H~6','L', 'L(?)3',
           'L(H)3', 'L(LL)3', 'L(LL)3.05', 'L(LL)3.5-3.7', 'L(LL)5', 'L(LL)6',
           'L(LL)~4', 'L-imp melt', 'L-melt breccia', 'L-melt rock', 'L-metal',
           'L/LL', 'L/LL(?)3', 'L/LL-melt rock', 'L/LL3', 'L/LL3-5', 'L/LL3-6',
           'L/LL3.10', 'L/LL3.2', 'L/LL3.4', 'L/LL3.5', 'L/LL3.6/3.7', 'L/LL4',
           'L/LL4-6', 'L/LL4/5', 'L/LL5', 'L/LL5-6', 'L/LL5/6', 'L/LL6',
           'L/LL6-an', 'L/LL~4', 'L/LL~5', 'L/LL~6', 'L3', 'L3-4', 'L3-5',
           'L3-6', 'L3-7', 'L3.0', 'L3.0-3.7', 'L3.0-3.9', 'L3.05', 'L3.1',
           'L3.10', 'L3.2', 'L3.2-3.5', 'L3.2-3.6', 'L3.3', 'L3.3-3.5',
           'L3.3-3.6', 'L3.3-3.7', 'L3.4', 'L3.4-3.7', 'L3.5', 'L3.5-3.7',
           'L3.5-3.8', 'L3.5-3.9', 'L3.5-5', 'L3.6', 'L3.6-4', 'L3.7',
           'L3.7-3.9', 'L3.7-4', 'L3.7-6', 'L3.7/3.8', 'L3.8', 'L3.8-5',
           'L3.8-6', 'L3.8-an', 'L3.9', 'L3.9-5', 'L3.9-6', 'L3.9/4', 'L3/4',
           'L4', 'L4 ', 'L4-5', 'L4-6', 'L4-an', 'L4-melt rock', 'L4/5', 'L5',
           'L5 ', 'L5-6', 'L5-7', 'L5/6', 'L6', 'L6 ', 'L6-melt breccia',
           'L6-melt rock', 'L6/7', 'L7', 'LL', 'LL(L)3', 'LL-melt rock', 'LL3',
           'LL3-4', 'LL3-5', 'LL3-6', 'LL3.0', 'LL3.00', 'LL3.1', 'LL3.1-3.5',
           'LL3.10', 'LL3.15', 'LL3.2', 'LL3.3', 'LL3.4', 'LL3.5', 'LL3.6',
           'LL3.7', 'LL3.7-6', 'LL3.8', 'LL3.8-6', 'LL3.9', 'LL3.9/4', 'LL3/4',
           'LL4', 'LL4-5', 'LL4-6', 'LL4/5', 'LL4/6', 'LL5', 'LL5-6', 'LL5-7',
           'LL5/6', 'LL6', 'LL6 ', 'LL6(?)', 'LL6/7', 'LL7', 'LL7(?)',
           'LL<3.5', 'LL~3', 'LL~4', 'LL~4/5', 'LL~5', 'LL~6',
           'L~3', 'L~4', 'L~5', 'L~6','Relict H','Relict OC', 'EH','EH-imp melt', 'EH3', 'EH3/4-an', 'EH4', 'EH4/5', 'EH5', 'EH6',
           'EH6-an', 'EH7', 'EH7-an', 'EL3', 'EL3/4', 'EL4', 'EL4/5', 'EL5',
           'EL6', 'EL6 ', 'EL6/7', 'EL7','E','E3','E4', 'E5','E6', 'K', 'K3','R', 'R3', 'R3-4', 'R3-5', 'R3-6', 'R3.4', 'R3.5-6',
           'R3.6', 'R3.7', 'R3.8', 'R3.8-5', 'R3.8-6', 'R3.9', 'R3/4', 'R4',
           'R4/5', 'R5', 'R6', 'Chondrite-fusion crust', 'Enst achon-ung', 'Achondrite-ung','Chondrite-ung',
           'Enst achon','E-an',  'E3-an',  'E5-an'],value='Aerolitos',inplace=True)

data.recclass.replace(to_replace=['Stone-uncl', 'Unknown', 'Fusion crust', 'Impact melt breccia', 'Stone-ung'],value='Desconhecidos',inplace=True)

data['recclass'].value_counts()

#separando 'Quedas' e 'Achados' totais
data_quedas = data.groupby('fall').get_group('Fell').copy()

data_achados = data.groupby('fall').get_group('Found').copy()


#antes de 1900
dados_XIX = data[(data.year < 1900 )]

dados_XIX_quedas = dados_XIX.groupby('fall').get_group('Fell').copy()

dados_XIX_achados = dados_XIX.groupby('fall').get_group('Found').copy()



#apos 1900
dados_XX = data[(data.year >= 1900 )]

dados_XX_quedas = dados_XX.groupby('fall').get_group('Fell').copy()

dados_XX_achados = dados_XX.groupby('fall').get_group('Found').copy()


print('Quantidade de entradas: {}'.format(len(data)))
print('Quantidade total de meteoritos tipo "queda": {}'.format(len(data_quedas)))
print('Quantidade total de meteoritos tipo "achados": {}\n'.format(len(data_achados)))

print('Quantidade de meteoros antes de 1900: {}'.format(len(dados_XIX)))
print('quantidade de meteoros "queda" antes de 1900: {}'.format(len(dados_XIX_quedas)))
print('quantidade de meteoros "achado" antes de 1900: {}\n'.format(len(dados_XIX_achados)))

print('quantidade de meteoros depois de 1900: {}'.format(len(dados_XX)))
print('quantidade de meteoros "queda" depois de 1900: {}'.format(len(dados_XX_quedas)))
print('quantidade de meteoros "achado" depois de 1900: {}'.format(len(dados_XX_achados)))


#grafico tipo pizza
plt.title("Total de Meteoritos separados por tipo")
values = [len(data_achados), len(data_quedas)] 
labels = ['Achados', 'Quedas']

plt.pie(values, labels=labels, autopct='%1.1f%%')
plt.show()


plt.title("Meteoritos do tipo 'Queda' (antes e depois de 1900)")
values = [len(dados_XIX_quedas), len(dados_XX_quedas)] 
labels = ['seculo XIX', 'seculo XX']

plt.pie(values, labels=labels, autopct='%1.1f%%')
plt.show()

plt.title("Meteoritos antes e depois do seculo XX)")
values = [len(dados_XIX), len(dados_XX)] 
labels = ['antes do seculo XX', 'seculo XX']

plt.pie(values, labels=labels, autopct='%1.1f%%')
plt.show()
#isolando os meteoritos por classe
geral_aerolito = data.groupby('recclass').get_group('Aerolitos').copy()

geral_siderito = data.groupby('recclass').get_group('Sideritos').copy()

geral_siderolito = data.groupby('recclass').get_group('Siderolitos').copy()

geral_desconhecidos = data.groupby('recclass').get_group('Desconhecidos').copy()

plt.title("Proporcao de Meteoritos por classe")
values = [len(geral_aerolito), len(geral_siderito), len(geral_siderolito), len(geral_desconhecidos)] 
labels = ['Aerolitos', 'Sideritos', 'Siderolitos', 'Desconhecidos']

plt.pie(values, labels=labels, autopct='%1.1f%%')
plt.show()

#isolando os meteoros brasileiros
dataBr = data[(data.reclat >=-34) & (data.reclat <= 6) & (data.reclong >=-55) & (data.reclong <= -30)]
print('Quantidade de Meteoritos Brasileiros: {}\n'.format(len(dataBr)))

brasil_aerolito = dataBr.groupby('recclass').get_group('Aerolitos').copy()

brasil_siderito = dataBr.groupby('recclass').get_group('Sideritos').copy()

brasil_siderolito = dataBr.groupby('recclass').get_group('Siderolitos').copy()


plt.title("Proporcao de Meteoritos por classe no Brasil")
values = [len(brasil_aerolito), len(brasil_siderito), len(brasil_siderolito)] 
labels = ['Aerolitos', 'Sideritos', 'Siderolitos']

plt.pie(values, labels=labels, autopct='%1.1f%%')
plt.show()

#plotando mapa do brasil com os meteoritos 'achados' e 'Quedas'

#projeção do tipo cilindrica equidistante https://www.suapesquisa.com/geografia_do_brasil/pontos_extremos.htm
#pontos mais ao norte e mais ao sul do Brasil (Lat 6/ Lat -34)
#pontos mais ao oeste e mais ao leste do Brasil (Lon -34 / Lon -73)
map = Basemap(projection='cyl',llcrnrlat=-34,llcrnrlon=-76,urcrnrlat=6,urcrnrlon=-30,resolution='c')
#valores para visualização geral -34/-76/6/-30

map.etopo() 
map.drawcountries() #desenha as linhas dos países
map.drawcoastlines(linewidth=0.6) #desenha as linhas das costas maritimas
map.drawstates(linewidth=0.6) #desenha as linhas dos estados

#plota os dados no mapa
#edgecolor - cor do background
#color - cor do ponto k=preto
#marker - por default é um ponto
#alpha - tranparencia do ponto (0 - 1)

map.scatter(dataBr.reclong,dataBr.reclat,edgecolor='none',color='k',alpha=0.8)
plt.title('Meteoritos no Brasil', fontsize=15)
#separando por tamanho
classes_3 = data[(data.mass <= 100000)] #menores ou iguais à 100kg
classes_3['recclass'].value_counts()
classes_4 = data[(data.mass > 100000)] #maiores que 100kg
classes_4['recclass'].value_counts()
#chances de ser atingido no Brasil

#area da superficie terrestre 510 072 000km2
#sendo destes 70,9% mares e oceanos
#e 29.1% de terra firme correspondente a 148,940,000km2

ext_mundo = 148940000.0

#meteoritos considerados validos pro levantamento (114 anos de dados, de 1900 a 2014)
met_val_ano = len(dados_XX_quedas)/114.0

print("Caem pelo menos {} meteoritos ao ano, grandes o suficiente para causar danos.".format(met_val_ano))

#meteoros por kilometro quadrado no mundo
met_km2 = met_val_ano/ext_mundo

print("Caem {} meteoritos por kilometro quadrado ao ano no mundo".format(met_km2))

#extensao do territorio brasileiro
ext_brasil = 8516000.0

met_br_ano = met_km2*ext_brasil

print("Sendo assim caem {} meteoritos ao ano no Brasil.".format(met_br_ano))

met_km2_br = met_br_ano/ext_brasil

print("Caem {} meteoritos por kilometro quadrado ao ano no Brasil.".format(met_br_ano))

#chance de atingir alguem no mundo densidade demografica mundial 50,79 pessoas/km2

chance_mundo = met_km2*50.79

print("a chance de atingir alguem no mundo eh de {}".format(chance_mundo))
print("Ou de 1 em {}".format(int(1/chance_mundo)))

#chance de atingir alguem no Brasil - densidade demografica brasileira 24,7 pessoas/km2

chance_brasil = met_km2_br*24.7

print("a chance de atingir alguem no brasil eh de {}".format(chance_brasil))
print("Ou de 1 em {}".format(int(1/chance_brasil)))
dataBr['recclass'].value_counts()


