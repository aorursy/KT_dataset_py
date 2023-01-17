# Importando

import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))
# Funções
def load_database(custom_query='SELECT * from football_data'):
    #with sqlite3.connect('../input/database.sqlite') as con:
    with sqlite3.connect('database.db') as con:
        all_data = pd.read_sql_query(custom_query, con)
        
    return all_data

def plota_histograma(x):
    %matplotlib inline
    #x = data['Goals'].values.tolist()
    plt.xlabel('Total de gols')
    plt.ylabel('Frequência')
    plt.title('Histograma de Gols')
    return plt.hist(x, bins='auto')
from shutil import copyfile #I've faced a problem with loading database.sqlite file, with database.db it works well
copyfile('../input/database.sqlite', 'database.db')
  
data = load_database("SELECT (FTHG+FTAG) as Goals from football_data")
print(data.head())
x = data['Goals'].values.tolist()
print("Mínimo de total de gols registrado:",min(x))
print("Máximo de total de gols registrado:",max(x))
data = load_database("SELECT Season, Date, Div, Country, League, HomeTeam, AwayTeam, FTHG, FTAG, (FTHG+FTAG) as Goals from football_data WHERE FTHG+FTAG<0 ")
print(data.head())
data = load_database("SELECT (FTHG+FTAG) as Goals from football_data WHERE FTHG+FTAG>=0")
print(data.head())

x = data['Goals'].values.tolist()
print("Mínimo de total de gols registrado:",min(x))
print("Máximo de total de gols registrado:",max(x))
x = data['Goals'].values.tolist()
Null, Null, Null = plota_histograma(x)
data = load_database("SELECT (FTHG+FTAG) as Goals from football_data WHERE FTHG+FTAG>4")
print(data.head())

x = data['Goals'].values.tolist()
print("Mínimo de total de gols registrado:",min(x))
print("Máximo de total de gols registrado:",max(x))
x = data['Goals'].values.tolist()
Null, Null, Null = plota_histograma(x)
data = load_database("SELECT (FTHG+FTAG) as Goals from football_data WHERE FTHG+FTAG>-1")
print(data.head())

x = data['Goals'].values.tolist()
print("Mínimo de total de gols registrado:",min(x))
print("Máximo de total de gols registrado:",max(x))
Null, Null, Null = plota_histograma(x)
counts, bins, bars = plota_histograma(x)
#print("C=",counts, "Bi=",bins, "Ba=", bars)
#print("Counts=",counts, len(counts))
#print("Bins=",bins, len(bins))
#print("Bars=", bars[0]) #Rectangle??
frequencias_gols = [cg for cg in counts if cg > 0]
print(frequencias_gols)
for gol in range(3,13):
    if gol != 3:
        print("Gols=",gol, " frequencia=", frequencias_gols[gol-3], ", proporção=", frequencias_gols[gol-3-1]/frequencias_gols[gol-3] )
    else:
        print("Gols=",gol, " frequencia=", frequencias_gols[gol-3])
# Não funciona com valores negativos - em construção
def dobreProporcoesEImprime(frequencias, faixas):
    maximo = max(faixas)
    #minimo = min(faixas)
    minimo=1
    for g in range(1,int((max(faixas)+1)/2)+1):
        #print(g, g-minimo, g*2-minimo, len(frequencias), minimo, maximo  )
        print("Dobrando gols, de",g,"para ", g*2, "proporção=", frequencias[g-minimo]/frequencias[g*2-minimo])

#frequencias_gols = [cg for cg in counts if cg > 0]
#dobreProporcoesEImprime(frequencias_gols)
print("Dobrando gols, de 3 para 6 proporção=", frequencias_gols[3-1]/frequencias_gols[6-1] )
print("Dobrando gols, de 4 para 8 proporção=", frequencias_gols[4-1]/frequencias_gols[8-1] )
print("Dobrando gols, de 5 para 10 proporção=", frequencias_gols[5-1]/frequencias_gols[10-1] )
print("Dobrando gols, de 6 para 12 proporção=", frequencias_gols[6-1]/frequencias_gols[12-1] )
#dobreProporcoesEImprime(frequencias_gols, range(min(x),max(x)) )
data = load_database("SELECT FTHG as Gol_Man from football_data WHERE FTHG+FTAG>0")
print(data.head())

x = data['Gol_Man'].values.tolist()
print("Mínimo de total de gols registrado:",min(x))
print("Máximo de total de gols registrado:",max(x))

counts, bins, bars = plota_histograma(x)
frequencias_gols = [cg for cg in counts if cg > 0]

#print("Dobrando gols, de 3 para 6 proporção=", frequencias_gols[3-1]/frequencias_gols[6-1] )
#print("Dobrando gols, de 4 para 8 proporção=", frequencias_gols[4-1]/frequencias_gols[8-1] )
#print("Dobrando gols, de 5 para 10 proporção=", frequencias_gols[5-1]/frequencias_gols[10-1] )
dobreProporcoesEImprime(frequencias_gols, range(min(x),max(x)+1) )
data = load_database("SELECT FTAG as Gol_Vis from football_data WHERE FTHG+FTAG>0")
print(data.head())

x = data['Gol_Vis'].values.tolist()
print("Mínimo de total de gols registrado:",min(x))
print("Máximo de total de gols registrado:",max(x))

counts, bins, bars = plota_histograma(x)
frequencias_gols = [cg for cg in counts if cg > 0]

dobreProporcoesEImprime(frequencias_gols, range(min(x),max(x)+1) )
data = load_database("SELECT (FTHG-FTAG) as SG from football_data WHERE FTHG+FTAG>-1")
#print(data.head())

x = data['SG'].values.tolist()
print("Mínimo de saldo registrado:",min(x))
print("Máximo de saldo registrado:",max(x))

counts, bins, bars = plota_histograma(x)
frequencias_saldos = [cg for cg in counts if cg > 0]

print("Dobrando saldo, de 1 para 2 proporção=", frequencias_saldos[1+9]/frequencias_saldos[2+9] )
print("Dobrando saldo, de 2 para 4 proporção=", frequencias_saldos[2+9]/frequencias_saldos[4+9] )
print("Dobrando saldo, de 3 para 6 proporção=", frequencias_saldos[3+9]/frequencias_saldos[6+9] )
print("Dobrando saldo, de 4 para 8 proporção=", frequencias_saldos[4+9]/frequencias_saldos[8+9] )
print("Dobrando saldo, de 5 para 10 proporção=", frequencias_saldos[5+9]/frequencias_saldos[10+9] )
#dobreProporcoesEImprime(frequencias_gols, range(min(x),max(x)+1) )

print("Dobrando saldo, de -1 para -2 proporção=", frequencias_saldos[-1+9]/frequencias_saldos[-2+9] )
print("Dobrando saldo, de -2 para -4 proporção=", frequencias_saldos[-2+9]/frequencias_saldos[-4+9] )
print("Dobrando saldo, de -3 para -6 proporção=", frequencias_saldos[-3+9]/frequencias_saldos[-6+9] )
print("Dobrando saldo, de -4 para -8 proporção=", frequencias_saldos[-4+9]/frequencias_saldos[-8+9] )
print("Dobrando saldo, de -5 para -10 proporção=", frequencias_saldos[-5+9]/frequencias_saldos[-10+9] )
#dobreProporcoesEImprime(frequencias_gols, range(-1*max(x)+1,-1*min(x)) )
print("Simetria de saldos, entre +1 e -1 proporção=", frequencias_saldos[+1+9]/frequencias_saldos[-1+9] )
print("Simetria de saldos, entre +2 e -2 proporção=", frequencias_saldos[+2+9]/frequencias_saldos[-2+9] )
print("Simetria de saldos, entre +3 e -3 proporção=", frequencias_saldos[+3+9]/frequencias_saldos[-3+9] )
print("Simetria de saldos, entre +4 e -4 proporção=", frequencias_saldos[+4+9]/frequencias_saldos[-4+9] )
print("Simetria de saldos, entre +5 e -5 proporção=", frequencias_saldos[+5+9]/frequencias_saldos[-5+9] )
print("Simetria de saldos, entre +6 e -6 proporção=", frequencias_saldos[+6+9]/frequencias_saldos[-6+9] )
print("Simetria de saldos, entre +7 e -7 proporção=", frequencias_saldos[+7+9]/frequencias_saldos[-7+9] )
print("Simetria de saldos, entre +8 e -8 proporção=", frequencias_saldos[+8+9]/frequencias_saldos[-8+9] )
print("Simetria de saldos, entre +9 e -9 proporção=", frequencias_saldos[+9+9]/frequencias_saldos[-9+9] )
data0x0 = load_database("SELECT COUNT(*) as Total from football_data WHERE FTHG+FTAG>-1 AND FTHG=0 AND FTAG=0")
#data = load_database("SELECT FTHG || '-' || FTAG as Placar from football_data WHERE FTHG+FTAG>-1 AND FTHG-FTAG=5")
#data = load_database("SELECT 100*FTHG+FTAG as Placar from football_data WHERE FTHG+FTAG>-1")
#print(data0x0.head())

data1x0 = load_database("SELECT COUNT(*) as Total from football_data WHERE FTHG+FTAG>-1 AND FTHG=1 AND FTAG=0")
data2x0 = load_database("SELECT COUNT(*) as Total from football_data WHERE FTHG+FTAG>-1 AND FTHG=2 AND FTAG=0")
data2x1 = load_database("SELECT COUNT(*) as Total from football_data WHERE FTHG+FTAG>-1 AND FTHG=2 AND FTAG=1")
data3x0 = load_database("SELECT COUNT(*) as Total from football_data WHERE FTHG+FTAG>-1 AND FTHG=3 AND FTAG=0")
data3x1 = load_database("SELECT COUNT(*) as Total from football_data WHERE FTHG+FTAG>-1 AND FTHG=3 AND FTAG=1")
data3x2 = load_database("SELECT COUNT(*) as Total from football_data WHERE FTHG+FTAG>-1 AND FTHG=3 AND FTAG=2")
data4x0 = load_database("SELECT COUNT(*) as Total from football_data WHERE FTHG+FTAG>-1 AND FTHG=4 AND FTAG=0")
data4x1 = load_database("SELECT COUNT(*) as Total from football_data WHERE FTHG+FTAG>-1 AND FTHG=4 AND FTAG=1")
data5x0 = load_database("SELECT COUNT(*) as Total from football_data WHERE FTHG+FTAG>-1 AND FTHG=5 AND FTAG=0")

data0x0 = load_database("SELECT COUNT(*) as Total from football_data WHERE FTHG+FTAG>-1 AND FTHG=0 AND FTAG=0")
data1x1 = load_database("SELECT COUNT(*) as Total from football_data WHERE FTHG+FTAG>-1 AND FTHG=1 AND FTAG=1")
data2x2 = load_database("SELECT COUNT(*) as Total from football_data WHERE FTHG+FTAG>-1 AND FTHG=2 AND FTAG=2")

data0x1 = load_database("SELECT COUNT(*) as Total from football_data WHERE FTHG+FTAG>-1 AND FTHG=0 AND FTAG=1")
data0x2 = load_database("SELECT COUNT(*) as Total from football_data WHERE FTHG+FTAG>-1 AND FTHG=0 AND FTAG=2")
data1x2 = load_database("SELECT COUNT(*) as Total from football_data WHERE FTHG+FTAG>-1 AND FTHG=1 AND FTAG=2")
data0x3 = load_database("SELECT COUNT(*) as Total from football_data WHERE FTHG+FTAG>-1 AND FTHG=0 AND FTAG=3")
data1x3 = load_database("SELECT COUNT(*) as Total from football_data WHERE FTHG+FTAG>-1 AND FTHG=1 AND FTAG=3")
data2x3 = load_database("SELECT COUNT(*) as Total from football_data WHERE FTHG+FTAG>-1 AND FTHG=2 AND FTAG=3")
data0x4 = load_database("SELECT COUNT(*) as Total from football_data WHERE FTHG+FTAG>-1 AND FTHG=0 AND FTAG=4")
data1x4 = load_database("SELECT COUNT(*) as Total from football_data WHERE FTHG+FTAG>-1 AND FTHG=1 AND FTAG=4")

p1x0 = data1x0['Total'].values[0]
p2x0 = data2x0['Total'].values[0]
p2x1 = data2x1['Total'].values[0]
p3x0 = data3x0['Total'].values[0]
p3x1 = data3x1['Total'].values[0]
p3x2 = data3x2['Total'].values[0]
p4x0 = data4x0['Total'].values[0]
p4x1 = data4x1['Total'].values[0]
p5x0 = data5x0['Total'].values[0]

p0x0 = data0x0['Total'].values[0]
p1x1 = data1x1['Total'].values[0]
p2x2 = data2x2['Total'].values[0]

p0x1 = data0x1['Total'].values[0]
p0x2 = data0x2['Total'].values[0]
p1x2 = data1x2['Total'].values[0]
p0x3 = data0x3['Total'].values[0]
p1x3 = data1x3['Total'].values[0]
p2x3 = data2x3['Total'].values[0]
p0x4 = data0x4['Total'].values[0]
p1x4 = data1x4['Total'].values[0]

print(p1x1)

print("Dobrando placar, de 1x1 para 2x2 proporção=", p1x1/p2x2 )
print("Dobrando placar, de 1x0 para 2x0 proporção=", p1x0/p2x0 )
print("Dobrando placar, de 0x1 para 0x2 proporção=", p0x1/p0x2 )

print("Simetria de saldos, entre 2x1 e 1x2 proporção=", p2x1/p1x2 )
print("Simetria de saldos, entre 3x1 e 1x3 proporção=", p3x1/p1x3 )
print("Simetria de saldos, entre 3x2 e 2x3 proporção=", p3x2/p2x3 )
print("Simetria de saldos, entre 4x1 e 1x4 proporção=", p4x1/p1x4 )