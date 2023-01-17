import pandas as pd
import matplotlib.pyplot as plt
resposta = [
    ["Paises Fabricantes", "Qualitativa Nominal"],
    ["Motores","Qualitativa Nominal"],
    ["Tipo de aeronave","Qualitativa Nominal"],
    ["Fatalidades","Quantitativa Discreta"],
    ["Ano de Fabricação","Quantitativa Discreta"],
    ["Tipo de dano","Qualitativa Nominal"],
    ["Fase de operação","Qualitativa Nominal"],
]
resposta = pd.DataFrame(resposta, columns=["Variavel", "Classificação"])
resposta
df = pd.read_csv('../input/anv.csv', delimiter=',')
df.head(5)

df_fabr = df[df.aeronave_pais_fabricante != 'NÃO IDENTIFICADO']
data = df_fabr['aeronave_pais_fabricante'].value_counts()[0:7]
paises = data.keys().tolist()

print(data)

plt.rcdefaults()
fig, ax = plt.subplots()
plt.xticks(rotation='vertical')

ax.bar(paises, data, align='center', ecolor='black')
ax.set_title('Os 7 Maiores fabricantes de aeronaves')

plt.show()

df_motor = df[df.aeronave_motor_quantidade != '***']
data = df_motor['aeronave_motor_quantidade'].value_counts()
labels = data.keys().tolist()
print(data)

fig1, ax1 = plt.subplots()
ax1.pie(data, explode=(0.05, 0, 0, 0, 0), autopct='%1.1f%%',shadow=False, startangle=90)
ax1.axis('equal')
ax1.legend(labels = labels,loc="center left",bbox_to_anchor=(1, 0, 0.5, 1))
ax1.set_title("Quantidade de Motores utilizados nos aviões")

plt.show()

data = df[df.aeronave_tipo_veiculo != 'INDETERMINADA'].groupby(['aeronave_tipo_veiculo'])['total_fatalidades'].sum()
data = data.sort_values()[data > 0]
labels = data.keys().tolist()
print(data)

plt.rcdefaults()
fig, ax = plt.subplots()
plt.xticks(rotation='vertical')

ax.bar(labels, data, align='center', ecolor='black', color='#ff4422')
ax.set_title('Os tipos de aeronaves com mais acidentes fatais')

plt.show()

data = df[df.aeronave_ano_fabricacao > 0].groupby((2018 - df.aeronave_ano_fabricacao)//5*5)['total_fatalidades'].sum()
labels = data.keys().tolist()
print(data)

plt.plot(labels,data, color='#ff4422')
plt.ylabel('Acidentes fatais')
plt.xlabel('Idade da aeronave')
plt.title('Acidentes fatais relacionados a idade da aeronave');
plt.show()


data = df[(df.total_fatalidades == 0) & (df.aeronave_nivel_dano != 'INDETERMINADO')]['aeronave_nivel_dano'].value_counts()
#df.aeronave_nivel_dano != 'INDETERMINADO'
labels = data.keys().tolist()
print(data)

fig1, ax1 = plt.subplots()
ax1.pie(data, explode=(0, 0, 0, 0.1), autopct='%1.1f%%',shadow=False, startangle=90)
ax1.axis('equal')
ax1.legend(labels = labels,loc="center left",bbox_to_anchor=(1, 0, 0.5, 1))
ax1.set_title("Acidentes sem vitimas fatais por tipo dano na aeronave")

plt.show()
data = df[(df.aeronave_fase_operacao != '***') & (df.aeronave_fase_operacao != 'INDETERMINADA')].groupby(['aeronave_fase_operacao'])['total_fatalidades'].sum()
data = data.sort_values()[data > 0].sort_values(ascending=False)[0:12].sort_values()
labels = data.keys().tolist()
print(data)

plt.rcdefaults()
fig, ax = plt.subplots()
plt.xticks(rotation='vertical')

ax.bar(labels, data, align='center', ecolor='black', color='#ff4422')
ax.set_title('As fases de operação de voo com maior número de mortes')

plt.show()