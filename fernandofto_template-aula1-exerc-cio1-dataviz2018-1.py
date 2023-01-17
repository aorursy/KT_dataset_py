import pandas as pd
import matplotlib.pyplot as plt
resposta = [["codigo_ocorrencia", "Quantitativa Discreta"],["aeronave_tipo_veiculo","Qualitativa Nominal"],["aeronave_nivel_dano","Qualitativa Ordinal"],["aeronave_motor_tipo","Qualitativa Nominal"],["aeronave_motor_quantidade","Qualitativa Ordinal"],["aeronave_ano_fabricacao","Quantitativa Discreta"],["aeronave_pais_fabricante","Qualitativa Nominal"],["total_fatalidades","Quantitativa Discreta"]] #variáveis relacionadas a tempo são contínuas, mas podem ser discretas pois não há perdas -- (discretização)
resposta = pd.DataFrame(resposta, columns=["Variavel", "Classificação"])
resposta.head(8)
df = pd.read_csv('../input/anv.csv', delimiter=',')
df.loc[:, ['codigo_ocorrencia','aeronave_tipo_veiculo','aeronave_nivel_dano','aeronave_motor_tipo','aeronave_motor_quantidade','aeronave_ano_fabricacao','aeronave_pais_fabricante','total_fatalidades']]
f_aeronave_tipo_veiculo = df['aeronave_tipo_veiculo'].value_counts()
f_aeronave_tipo_veiculo.head()
f_aeronave_nivel_dano = df['aeronave_nivel_dano'].value_counts()
f_aeronave_nivel_dano.head()
f_aeronave_motor_tipo = df['aeronave_motor_tipo'].value_counts()
f_aeronave_motor_tipo.head()
f_aeronave_motor_quantidade = df['aeronave_motor_quantidade'].value_counts()
f_aeronave_motor_quantidade.head()
f_aeronave_pais_fabricante = df['aeronave_pais_fabricante'].value_counts()
f_aeronave_pais_fabricante.head()
fig1,ax1 = plt.subplots()
labels = f_aeronave_nivel_dano.keys().tolist()

ax1.set_title('Nível de dano nas aeronaves acidentadas')
ax1.pie(f_aeronave_nivel_dano,autopct='%1.1f%%',shadow=False, startangle=90)

ax1.axis('equal')
ax1.legend(labels = labels, bbox_to_anchor=(1,0,0.5,1))

plt.show()
fig1,ax1 = plt.subplots()
labels = f_aeronave_motor_tipo.keys().tolist()

ax1.set_title('Tipos motores de aeronaves acidentadas')
ax1.pie(f_aeronave_motor_tipo,autopct='%1.1f%%',shadow=False, startangle=90)

ax1.axis('equal')
ax1.legend(labels = labels, bbox_to_anchor=(1,0,0.5,1))

plt.show()
plt.rcdefaults()
fig, ax = plt.subplots()
plt.xticks(rotation='vertical')
labels = f_aeronave_tipo_veiculo.keys().tolist()

ax.bar(labels, f_aeronave_tipo_veiculo, align='center', ecolor='black', color='#120a8f')
ax.set_title('Os acidentes por tipos de aeronaves')

plt.show() 
plt.rcdefaults()
fig, ax = plt.subplots()
plt.xticks(rotation='vertical')
labels = f_aeronave_motor_quantidade.keys().tolist()

ax.bar(labels, f_aeronave_motor_quantidade, align='center', ecolor='black', color='#120a8f')
ax.set_title('Motores das aeronaves acidentadas')

plt.show() 
plt.rcdefaults()
fig, ax = plt.subplots()
plt.xticks(rotation='vertical')
labels = f_aeronave_pais_fabricante.keys().tolist()

ax.bar(labels, f_aeronave_pais_fabricante, align='center', ecolor='black', color='#120a8f')
ax.set_title('Origem de fabricação das aeronaves acidentadas')

plt.show() 
df = pd.read_csv('../input/anv.csv', delimiter=',')