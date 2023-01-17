import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
traindata = pd.read_csv("../input/train.csv", usecols=['Id','field','age','type','harvest_year','harvest_month','production'])
traindata.head()
field_min = 0
field_max = 27
fieldData = {}
for i in range(field_min,field_max+1):
    file = "../input/field-"+str(i)+".csv"
    fieldData[i] = pd.read_csv(file, usecols=['month','year','temperature','dewpoint','windspeed','Soilwater_L1','Soilwater_L2','Soilwater_L3','Soilwater_L4','Precipitation']).values
def getDataField(field, month, year):
    #todos os dados possuem tamanho 192, e vão de 01/2002 a 12/2017, o que significa que com base no mês e ano podemos calcular o índice de onde queremos os dados
    offsetano = int(year) - 2002
    offsetmes = int(month) #
    return fieldData[int(field)][((offsetano*12) + offsetmes)-1]

#provando que a função funciona
print(getDataField(1, 1, 2002)[0],getDataField(1, 1, 2002)[1])
print(getDataField(1, 1, 2003)[0],getDataField(1, 1, 2003)[1])
print(getDataField(1, 1, 2004)[0],getDataField(1, 1, 2004)[1])
###
print(getDataField(1, 4, 2002)[0],getDataField(1, 4, 2002)[1])
print(getDataField(1, 4, 2003)[0],getDataField(1, 4, 2003)[1])
print(getDataField(1, 4, 2004)[0],getDataField(1, 4, 2004)[1])
#avg temperature x production
data = traindata.values
dataClimatico = []
print(data[:, 1])
production = traindata['production'].values
for i in range(len(data)):
    dataClimatico.append(getDataField(data[i][1], data[i][5], data[i][4]))
dataClimatico = np.array(dataClimatico)
#index 2 -> temperature
plt.scatter(production, dataClimatico[:, 2], c="g", alpha=0.5,label="A")
plt.xlabel("Produção")
plt.ylabel("Média Temperatura")
plt.legend(loc=2)
plt.show()
#index 3 -> Dewpoint
plt.scatter(production, dataClimatico[:, 3], c="purple", alpha=0.5,label="A")
plt.xlabel("Produção")
plt.ylabel("Dewpoint")
plt.legend(loc=2)
plt.show()
#index 4 -> Windspeed
plt.scatter(production, dataClimatico[:, 4], c="b", alpha=0.5,label="A")
plt.xlabel("Produção")
plt.ylabel("Velocidade do vento")
plt.legend(loc=2)
plt.show()
#index 5 -> SoilwaterL1
plt.scatter(production, dataClimatico[:, 5], c="g", alpha=0.5,label="A")
plt.xlabel("Produção")
plt.ylabel("Umidade do solo L1")
plt.legend(loc=2)
plt.show()
#index 6 -> SoilwaterL2
plt.scatter(production, dataClimatico[:, 6], c="g", alpha=0.5,label="A")
plt.xlabel("Produção")
plt.ylabel("Umidade do solo L2")
plt.legend(loc=2)
plt.show()
#index 7 -> SoilwaterL3
plt.scatter(production, dataClimatico[:, 7], c="g", alpha=0.5,label="A")
plt.xlabel("Produção")
plt.ylabel("Umidade do solo L3")
plt.legend(loc=2)
plt.show()
#index 8 -> SoilwaterL4
plt.scatter(production, dataClimatico[:, 8], c="g", alpha=0.5,label="A")
plt.xlabel("Produção")
plt.ylabel("Umidade do solo L4")
plt.legend(loc=2)
plt.show()
#index 9 -> Precipitation
plt.scatter(production, dataClimatico[:, 9], c="r", alpha=0.5,label="A")
plt.xlabel("Produção")
plt.ylabel("Precipitação")
plt.legend(loc=2)
plt.show()
