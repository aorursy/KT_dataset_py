import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
pas = pd.read_csv('../input/AirPassengers.csv')

pas.head()
def ano():

    anos = []

    for i in pas['Month']:

        anos.append(int(i[:4]))

    return anos



def mes():

    meses = []

    for i in pas['Month']:

        meses.append(int(i[5:]))

    return meses



anos = ano()

meses = mes()
datas = pd.DataFrame({'mes': meses, 'ano': anos})
pas_cl = datas.join(pas['#Passengers'])

pas_cl.head()
porAno = pas_cl.groupby('ano').describe()['#Passengers']

porAno
plt.style.use('bmh')



fig = plt.figure()



ax1 = fig.add_axes([1.0,1.0,1.5,1.5])



ax1.plot(porAno.index ,porAno['mean'], label= 'Média')

ax1.plot(porAno.index ,porAno['50%'], label= 'Mediana')



ax1.set_xlabel('Anos')

ax1.set_ylabel('N° de passageiros')





ax1.set_title('Passageiros em função dos anos')

ax1.legend()
fig2 = plt.figure()



ax2 = fig2.add_axes([1.0,1.0,1.5,1.5])

ax2.boxplot(porAno)



ax2.set_xticks([i for i in range(13) if i != 0])

ax2.set_xticklabels([1949, 1950, 1951, 1952, 1953, 1954, 1955, 1956, 1957, 1958, 1959, 1960])
pas_cl.hist(column= '#Passengers', by= pas_cl['ano'], figsize= (12,12), sharex= True)
porMes = pas_cl.groupby(by = 'mes', sort = True).describe()

porMes
fig3 = plt.figure(figsize= (9,5))

ax3 = fig3.subplots(1,2)



ax3[0].bar(x= [i for i in range(13) if i != 0], height= porMes['#Passengers']['mean'])

ax3[1].bar(x= [i for i in range(13) if i != 0], height= porMes['#Passengers']['50%'])