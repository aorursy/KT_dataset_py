import pandas as pd #Versão utilizada 0.23.4
df = pd.read_csv('../input/anv.csv', delimiter=',')
df.info()
df.isna().sum()
df.aeronave_assentos.fillna(0,inplace=True)
df.describe()
df.head(3)
df.aeronave_modelo.value_counts().sort_index()
df.aeronave_motor_tipo.value_counts().sort_index()
df.aeronave_pmd_categoria.value_counts().sort_index()
#Realizando o replace para os campos '***' para INDETERMINADO
df.aeronave_modelo.replace("***","INDETERMINADO",inplace=True)
df.aeronave_motor_tipo.replace("***","INDETERMINADO",inplace=True)
df.aeronave_pmd_categoria.replace("***","INDETERMINADA",inplace=True)
resposta = []
resposta.append(["aeronave_tipo_veiculo", "Qualitativa Nominal"])
resposta.append(["aeronave_pmd_categoria","Qualitativa Nominal"])# por causa do indefinido, deixou de ser discreta
resposta.append(["aeronave_assentos","Quantitativa Discreta"])
resposta.append(["aeronave_motor_tipo","Qualitativa Nonimal"])
resposta.append(["aeronave_ano_fabricacao","Quantitativa Discreta"])
resposta.append(["aeronave_tipo_operacao","Qualitativa Nonimal"])
resposta.append(["total_fatalidades","Quantitativa Discreta"])
resposta.append(["aeronave_pais_fabricante","Qualitativa Nonimal"])
resposta.append(["aeronave_nivel_dano","Qualitativa Nonimal"])

resposta = pd.DataFrame(resposta, columns=["Variavel", "Classificação"])
resposta
#aeronave_tipo_veiculo
df.aeronave_tipo_veiculo.value_counts().sort_index()
#aeronave_pmd_categoria
df.aeronave_pmd_categoria.value_counts().sort_index()
#aeronave_motor_tipo
df.aeronave_motor_tipo.value_counts().sort_index()
#aeronave_tipo_operacao
df.aeronave_tipo_operacao.value_counts().sort_index()
#aeronave_pais_fabricante
df.aeronave_pais_fabricante.value_counts().sort_index()
#aeronave_nivel_dano
df.aeronave_nivel_dano.value_counts().sort_index()
#Versão utilizada do matplotlib 2.2.3
#Versão utilizada do numpy 1.7.1 presente no matplotlib
import matplotlib as mpl
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
mpl.rcParams.update(mpl.rcParamsDefault)
data = df[df['aeronave_ano_fabricacao']!=0].aeronave_ano_fabricacao.value_counts().sort_index()

pylab.figure(figsize=(14,7))
pylab.plot(data)

pylab.xlabel("Ano de fabricação")
pylab.ylabel("Acidentes")
pylab.title('Acidentes pelo ano de fabricação da aeronave')
pylab.show()
df.aeronave_nivel_dano.unique()
mpl.rcParams.update(mpl.rcParamsDefault)
data = pd.DataFrame(df.aeronave_nivel_dano.value_counts().sort_index())

ax = data.plot(kind='bar', color='#AAAA00',figsize=(16,7))
total = ["{}".format(int(row.aeronave_nivel_dano)) for name,row in data.iterrows()]
for i,child in enumerate(ax.get_children()[:data.index.size]):
    ax.text(i,child.get_bbox().y1+40,total[i], horizontalalignment ='center')

ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
ax.patch.set_facecolor('#FFFFFF')
ax.spines['bottom'].set_color('#CCCCCC')
ax.spines['bottom'].set_linewidth(1)
ax.spines['left'].set_color('#CCCCCC')
ax.spines['left'].set_linewidth(1)
plt.title("Total de acidentes por nível de dano")
plt.xticks(rotation=90)
plt.show()
data = df.aeronave_tipo_operacao.value_counts().sort_index()
data
mpl.rcParams.update(mpl.rcParamsDefault)
data = df.aeronave_tipo_operacao.sort_index()
prob = data.value_counts(normalize=True)
plt.figure(figsize=(15, 5))
prob.plot(kind='bar',title="Percentual de tipos de Voos")
plt.xticks(rotation=45)
plt.ylabel("Porcentagem")

plt.show()
plt.style.use('ggplot')
data = pd.DataFrame(df.aeronave_pmd_categoria.value_counts()).reset_index()

sizes = list(data.aeronave_pmd_categoria)
labels = list(data['index'])

explode = (0.05, 0, 0, 0)
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode, labels=labels, autopct='%1.1f%%', shadow=False,pctdistance=0.6, startangle=90)
ax1.axis('equal')
plt.title('Percentual de categorias de aeronaves')
plt.show()
data = pd.DataFrame(df.aeronave_tipo_veiculo.value_counts()).reset_index()
data.head()
data = df.aeronave_tipo_veiculo.value_counts().sort_index()

plt.style.use('bmh')
limit = 17
mask = data > limit
count = data.loc[~mask].sum()
data = data.loc[mask]
data['OUTROS'] = count
plt.figure(figsize=(15,5))
data.plot(kind='bar')

plt.xlabel("Tipos")
plt.ylabel("Quantidades")
plt.title("Quantidade dos principais tipos de aeronaves")
plt.xticks(rotation=90)

plt.show()
data = pd.DataFrame(df.aeronave_motor_tipo.value_counts()).reset_index()

plt.style.use('ggplot')

engineType = data['index']
engineQtd = data['aeronave_motor_tipo']
plt.figure(figsize=(14,7))
xPos = [i for i, _ in enumerate(engineType)]
plt.barh(xPos, engineQtd)
plt.xlabel("Quantidade")
plt.ylabel("Tipo")
plt.title("Tipos de motores por aeronaves acidentadas")

plt.yticks(xPos, engineType)

plt.show()
df.total_fatalidades.sum()
df.aeronave_assentos.sum()
data = df[['aeronave_assentos','total_fatalidades']]
data.head()
data = df[['aeronave_assentos','total_fatalidades']]
plt.style.use('ggplot')
data = data[data.total_fatalidades > 0]
data.reset_index(inplace=True)
data.drop('index', inplace=True, axis=1)
data.plot(kind='area',stacked=False,figsize=(15,7));
plt.title("Proporção entre assentos e fatalidade por registro de acidente aéreo")
plt.show()
df.groupby('aeronave_pais_fabricante')['total_fatalidades'].sum()
data = df.groupby('aeronave_pais_fabricante')['total_fatalidades'].sum()
data = data.reset_index()
data1 = data.total_fatalidades
x = data.aeronave_pais_fabricante
plt.style.use('ggplot')

plt.plot( x, data1, 'go', color ='red')
plt.plot( x, data1, 'k--', color='orange')
plt.title("Países fabricantes por fatalidades")
plt.grid(True)
plt.xlabel("Países")
plt.ylabel("Fatalidades")
plt.xticks(rotation=90)
plt.show()
df = pd.read_csv('../input/anv.csv', delimiter=',')
df.head(1)