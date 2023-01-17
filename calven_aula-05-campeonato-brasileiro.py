import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
cols = ['Horario', 'Dia', 'Data', 'Clube_1', 'Clube_2', 'Vencedor', 'Rodada', 'Arena', 'p1', 'p2', 'C1_Estado', 'C2_Estado', 'Vencedor_Estado']

df = pd.read_csv('../input/campeonato-brasileiro-full.csv', names=cols)
df.Clube_1 = df.Clube_1.apply(lambda x: x.capitalize())

df.Clube_2 = df.Clube_2.apply(lambda x: x.capitalize())

df.Vencedor = df.Vencedor.apply(lambda x: x.capitalize())
# Qual a chance, em %, de um time x vencer um time y?





jogosXeY = df.loc[((df.Clube_1 == 'Flamengo') & (df.Clube_2 == 'Cruzeiro') | (df.Clube_1 == 'Cruzeiro') & (df.Clube_2 == 'Flamengo'))]

#df['Clube_1'].filter('Guarani')

#display(jogosXeY)



total = jogosXeY.Dia.count()

#df.loc[df.Vencedor == '']

print("Total de Jogos: " + str(total))



flamengoVenceu = jogosXeY.loc[jogosXeY.Vencedor == 'Flamengo']

vitFla = flamengoVenceu.Dia.count()

vitCru = total - vitFla

print("Vitórias Flamengo: " + str(vitFla))

print("Vitórias Cruzeiro: " + str(vitCru))



porcentagemFla = (vitFla * 100) / total 

print("Porcentagem Flamengo: " + str(porcentagemFla) + "%")
#Simule uma partida entre dois times determinando o resultado com base no histórico de gols marcados usando média simples e média móvel





#display(jogosXeY)

flaC1 = jogosXeY.loc[(jogosXeY.Clube_1 == 'Flamengo')]

flaC2 = jogosXeY.loc[(jogosXeY.Clube_2 == 'Flamengo')]



#display(flaC1)

flaC1.p1 = pd.to_numeric(flaC1.p1)

p1 = flaC1.p1.sum()

flaC2.p2 = pd.to_numeric(flaC2.p2)

p2 = flaC2.p2.sum()

media = (p1+p2).mean()

print(p1)

print(media)
#Demonstre graficamente qual o volume de partidas por região do país.

plt.figure()

plt.rcParams["figure.figsize"] = [12,7]

plt.plot(df['C1_Estado'].value_counts())
#Crie um ranking dos times vencedores por rodada

df.groupby((['Rodada', 'Vencedor'])).count()