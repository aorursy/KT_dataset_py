import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os
sns.set()
path_root = "../input/"
files = os.listdir(path_root) 
print(' Quantidade de arquivos : ', len(files), ' Arquivos ')
print('-'*30)
for i,v in enumerate(files):
    print(i,' - ',v)
print('-'*30)
temperatura = pd.read_csv(path_root+'a1_temperatura.csv', index_col=['Ano'], parse_dates=['Ano'])
plt.figure(figsize=(15, 7))
plt.plot(temperatura.Cananeia, label='Cananeia')
plt.plot(temperatura.Ubatuba, label='Ubatuba')
plt.title(' Temperaturas Mensais ')
plt.grid(True)
plt.legend()
plt.show()
manchas = pd.read_csv(path_root+'a2_MANCHAS.csv', index_col=['Ano'], parse_dates=['Ano'])
plt.figure(figsize=(15, 7))
plt.plot(manchas.manchas)
plt.title(' Número de Manchas solares de Wolfer - obs. ANUAIS ')
plt.grid(True)
plt.show()
chuvas_for = pd.read_csv(path_root+'a3_fortaleza.csv', index_col=['Ano'], parse_dates=['Ano'])
plt.figure(figsize=(15, 7))
plt.plot(chuvas_for.Fort)
plt.title(' (a) Precipitação atmosférica em Fortalieza - obs. ANUAIS ')
plt.grid(True)
plt.show()
chuvas_lav = pd.read_csv(path_root+'a3_LAVRAS.csv', index_col=['Ano'], parse_dates=['Ano'])
plt.figure(figsize=(15, 7))
plt.plot(chuvas_lav.Precipitacao)
plt.title(' Precipitação atmosférica em Lavras - obs. MENSAIS  ')
plt.grid(True)
plt.show()
data = pd.read_csv(path_root+'a4_OZONIO.csv', index_col=['Ano'], parse_dates=['Ano'])
plt.figure(figsize=(15, 7))
plt.plot(data.Ozonio)
plt.title(' Concentração de ozônio em Azuza - obs. MENSAIS  ')
plt.grid(True)
plt.show()
data = pd.read_csv(path_root+'a5_ENERGIA.csv', index_col=['Ano'], parse_dates=['Ano'])
plt.figure(figsize=(15, 7))
plt.plot(data.Energia)
plt.title(' Consumo de Energia no Estado do Espírito Santo - obs. MENSAIS  ')
plt.grid(True)
plt.show()
data = pd.read_csv(path_root+'a6_poluicao.csv', index_col=['DATA'], parse_dates=['DATA'])
plt.figure(figsize=(15, 7))
plt.plot(data.co, label='CO - gás carbônico')
plt.plot(data.no2, label='NO2 - dióxido de nitrogênio')
plt.plot(data.PM10, label='PM10 - material particulado')
plt.plot(data.o3, label='O3 - Ozônio')
plt.plot(data.so2, label='SO2 - Dióxido de Enxofre')
plt.plot(data.RES65, label='RES65')
plt.title(' Poluição na cidade de São Paulo ')
plt.grid(True)
plt.legend()
plt.show()

conjuntos = [data.co,data.no2,data.PM10,data.o3,data.so2]
titulos = ['CO - gás carbônico','NO2 - dióxido de nitrogênio','PM10 - material particulado','O3 - Ozônio','SO2 - Dióxido de Enxofre']
fig, ax = plt.subplots(len(titulos), 1, figsize=(15,20))
fig.subplots_adjust(hspace=0.8)

for i in range(len(titulos)):
    ax[i].plot(conjuntos[i])
    ax[i].set_title(titulos[i])
data = pd.read_csv(path_root+'a7_atmosfera.csv', index_col=['Time'], parse_dates=['Time'])
plt.figure(figsize=(15, 7))
plt.plot(data.temperatura, label='Temperatura')
plt.plot(data.umidade, label='Umidade')
plt.title(' Temperaturas Mensais ')
plt.grid(True)
plt.legend()
plt.show()
data = pd.read_csv(path_root+'a8_a_pibanual.csv', index_col=['ano'], parse_dates=['ano'])
plt.figure(figsize=(15, 7))

plt.plot(data.pib1949)
plt.title(' (a) - Produto Interno Bruto do Brasil ')
plt.show()
data = pd.read_csv(path_root+'a8_b_IPI.csv', index_col=['Time'], parse_dates=['Time'])
plt.figure(figsize=(15, 7))

plt.plot(data.ipialiment)
plt.title(' (b) - Produção de Produtos Alimentares ')
plt.show()
data = pd.read_csv(path_root+'a8_c_PFI.csv', index_col=['Time'], parse_dates=['Time'])
plt.figure(figsize=(15, 7))

plt.plot(data.PFI)
plt.title(' (c) - Industrial Geral - obs. MENSAL ')
plt.show()
data = pd.read_csv(path_root+'a8_d_Bebida.csv', index_col=['Time'], parse_dates=['Time'])
plt.figure(figsize=(15, 7))

plt.plot(data.Bebida)
plt.title(' (d) - Produção de Alimentos e Bebidas - obs. MENSAL ')
plt.show()
data = pd.read_csv(path_root+'day_IBV-PETRO-BANESPA-CEMIG.csv', index_col=['Time'], parse_dates=['Time'])
plt.figure(figsize=(15, 7))

titulos =  data.keys()

fig, ax = plt.subplots(len(titulos), 1, figsize=(15,20))
fig.subplots_adjust(hspace=0.8)

for i in range(len(titulos)):
    ax[i].plot(data[titulos[i]])
    ax[i].set_title(titulos[i])

plt.title(' Mercado Financeiro - obs. Diária ')
plt.show()
# M-IBV-SP.csv
data = pd.read_csv(path_root+'M-IBV-SP.csv', index_col=['Time'], parse_dates=['Time'])
plt.figure(figsize=(15, 7))

titulos = data.keys()

fig, ax = plt.subplots(len(titulos), 1, figsize=(15,20))
fig.subplots_adjust(hspace=0.8)

for i in range(len(titulos)):
    ax[i].plot(data[titulos[i]])
    ax[i].set_title(titulos[i])

plt.title(' Mercado Financeiro - obs. Mensal ')
plt.show()
data = pd.read_csv(path_root+'D-GLOBO.csv', index_col=['Time'], parse_dates=['Time'])
plt.figure(figsize=(15, 7))

chaves = data.keys()

plt.plot(data[chaves[0]])
plt.title(' (g) - Mercado de ações da GLOBO - obs. Diária  ')
plt.show()
data = pd.read_csv(path_root+'D-TAM.csv', index_col=['Time'], parse_dates=['Time'])
plt.figure(figsize=(15, 7))

chaves = data.keys()

plt.plot(data[chaves[0]])
plt.title(' (g) - Mercado de ações da TAM - obs. Diária  ')
plt.show()
data = pd.read_csv(path_root+'a10_ICV.csv', index_col=['Time'], parse_dates=['Time'])
plt.figure(figsize=(15, 7))

chaves = data.keys()

plt.plot(data[chaves[0]])
plt.title('  Custo de Vida em SP - obs. MENSAL  ')
plt.show()
data = pd.read_csv(path_root+'a11_CONSUMO.csv', index_col=['data'], parse_dates=['data'])
plt.figure(figsize=(15, 7))

chaves = data.keys()

plt.plot(data[chaves[0]])
plt.title('  Vendas Físicas em SP - obs. MENSAL  ')
plt.show()
