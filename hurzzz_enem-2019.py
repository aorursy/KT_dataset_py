#import das bibliotecas utilizadas no notebook

import numpy as np

import pandas as pd

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn import metrics



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#lista de colunas da tabela

data = pd.read_csv("/kaggle/input/microdados-enem-2019/MICRODADOS_ENEM_2019.csv", nrows=10, encoding='iso-8859-1', sep=';')

cols = data.columns.tolist()

# print('Todas as colunas disponíveis: ',cols)

print('\nNúmero total de colunas disponíveis: ', len(cols))



#jogando fora colunas que não incluiremos en nosso modelo

drop_cols = ('IN', 'NU_ANO', 'TX_', 'NO_', 'SG_')

for col in cols[:]:

    if col.startswith(drop_cols):

        cols.remove(col)

print('\n')

print('colunas importadas:')

print(cols)

print('\nNúmero total de colunas importadas: ', len(cols))



#importando nosso dataset inteiro, somente com as colunas de interesse

data = pd.read_csv("/kaggle/input/microdados-enem-2019/MICRODADOS_ENEM_2019.csv", encoding='iso-8859-1', sep=';', usecols=cols)



#eliminando os estudantes que faltaram qualquer uma das provas

for col in ['TP_PRESENCA_CN', 'TP_PRESENCA_CH']:

    data = data[data[col] == 1]

    

#eliminando os estudantes que ficaram com nota 0 por qualquer outro motivo (e.g. eliminados)

data = data[data['NU_NOTA_MT'] != 0]
#imprimindo a tabela de estatística descritiva

dependentVariable = 'NU_NOTA_MT'

print(data[dependentVariable].describe(percentiles=[.25, .5, .75, .99]))



#distribuição da variável dependente

f, ax = plt.subplots()

ax.set(title="Distribuição de Notas")

sns.distplot(data[dependentVariable], axlabel="Nota")



f.savefig("summary_plot1.png", bbox_inches='tight', dpi=600)









#imprimindo assimetria e curtose

print("\nSkewness: %f" % data[dependentVariable].skew())

print("Kurtosis: %f" % data[dependentVariable].kurt())
#bar plot: Qual é a renda mensal de sua família? (Some a sua renda com a dos seus familiares.) x NU_NOTA_MT

var = 'Q006'

A = 'Nenhuma renda'

B = 'Até 998'

C = 'De 998 a 1.497'

D = 'De 1.497 a 1.996'

E = 'De 1.996 a 2.495'

F = 'De 2.495 a 2.994'

G = 'De 2.994 a 3.992'

H = 'De 3.992 a 4.990'

I = 'De 4.990 a 5.988'

J = 'De 5.988 a 6.986'

K = 'De 6.986 a 7.984'

L = 'De 7.984 a 8.982'

M = 'De 8.982 a 9.980'

N = 'De 9.980 a 11.976'

O = 'De 11.976 a 14.970'

P = 'De 14.970 a 19.960'

Q = 'Mais de 19.960'

plotdata = pd.concat([data[dependentVariable], data[var].map({'A': A, 'B': B, 'C': C, 'D': D, 'E': E, 'F': F, 'G': G, 'H': H, 'I': I, 'J': J, 'K': K, 'L': L, 'M': M, 'N': N, 'O': O, 'P': P, 'Q': Q})], axis=1)

f, ax = plt.subplots(figsize=(14, 6))

fig = sns.barplot(x=var, y=dependentVariable, data=plotdata, order=[A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q], estimator=np.mean)

fig.axis(ymin=320);

ax.set_xticklabels(ax.get_xticklabels(),rotation=80)

ax.set(ylabel="Nota de Matemática")

ax.set(xlabel="")

ax.set(title="Qual é a renda mensal de sua família, em reais?")



#box plot: Qual é a renda mensal de sua família? (Some a sua renda com a dos seus familiares.) x NU_NOTA_MT

var = 'Q006'

A = 'Nenhuma renda'

B = 'Até 998'

C = 'De 998 a 1.497'

D = 'De 1.497 a 1.996'

E = 'De 1.996 a 2.495'

F = 'De 2.495 a 2.994'

G = 'De 2.994 a 3.992'

H = 'De 3.992 a 4.990'

I = 'De 4.990 a 5.988'

J = 'De 5.988 a 6.986'

K = 'De 6.986 a 7.984'

L = 'De 7.984 a 8.982'

M = 'De 8.982 a 9.980'

N = 'De 9.980 a 11.976'

O = 'De 11.976 a 14.970'

P = 'De 14.970 a 19.960'

Q = 'Mais de 19.960'

plotdata = pd.concat([data[dependentVariable], data[var].map({'A': A, 'B': B, 'C': C, 'D': D, 'E': E, 'F': F, 'G': G, 'H': H, 'I': I, 'J': J, 'K': K, 'L': L, 'M': M, 'N': N, 'O': O, 'P': P, 'Q': Q})], axis=1)

f, ax = plt.subplots(figsize=(14, 6))

fig = sns.boxplot(x=var, y=dependentVariable, data=plotdata, order=[A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q], showmeans=True)

fig.axis(ymin=320);

ax.set_xticklabels(ax.get_xticklabels(),rotation=80)

ax.set(ylabel="Nota de Matemática")

ax.set(xlabel="")

ax.set(title="Qual é a renda mensal de sua família, em reais?")



f.savefig("renda.png", bbox_inches='tight', dpi=600)
#1.2% dos alunos da menor classe de renda conseguem tirar mais que 700 pontos na prova de Matemática do ENEM

print(data[data['Q006'] == 'B']['NU_NOTA_MT'].quantile(.98755))

#46.7% dos alunos da maior classe de renda conseguem tirar mais que 700 pontos na prova de Matemática do ENEM

print(data[data['Q006'] == 'Q']['NU_NOTA_MT'].quantile(.533))



#0.009% dos alunos da menor classe de renda conseguem tirar mais que 800 pontos na prova de Matemática do ENEM

print(data[data['Q006'] == 'B']['NU_NOTA_MT'].quantile(.9990915))

#mais de 13% dos alunos da maior classe de renda conseguem tirar mais que 800 pontos na prova de Matemática do ENEM

print(data[data['Q006'] == 'Q']['NU_NOTA_MT'].quantile(.8666))
#box plot: Em sua residência trabalha empregado(a) doméstico(a)? x NU_NOTA_MT

var = 'Q007'

A = 'Nenhum'

B = '1 ou 2 dias'

C = '3 ou 4 dias'

D = 'pelo menos 5 dias'

plotdata = pd.concat([data[dependentVariable], data[var].map({'A': A, 'B': B, 'C': C, 'D': D})], axis=1)

f, ax = plt.subplots(figsize=(10, 6))

fig = sns.boxplot(x=var, y=dependentVariable, data=plotdata, order=[A, B, C, D], showmeans=True)

fig.axis(ymin=320);



ax.set_xticklabels(ax.get_xticklabels(),rotation=0)

ax.set(ylabel="Nota de Matemática")

ax.set(xlabel="")

ax.set(title="Em sua residência trabalha empregado(a) doméstico(a) quantos dias por semana?")



f.savefig("empregado.png", bbox_inches='tight', dpi=600)
#box plot: Na sua residência tem carro?? x NU_NOTA_MT

var = 'Q010'

A = 'Nenhum'

B = 'Sim, 1'

C = 'Sim, 2'

D = 'Sim, 3'

E = 'Sim, 4 ou mais'

plotdata = pd.concat([data[dependentVariable], data[var].map({'A': A, 'B': B, 'C': C, 'D': D, 'E': E})], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y=dependentVariable, data=plotdata, order=[A, B, C, D, E])

fig.axis(ymin=320);

ax.set(ylabel="Nota de Matemática")

ax.set(xlabel="")

ax.set(title="Na sua residência tem carro?")
#box plot: Na sua residência tem TV por assinatura? x NU_NOTA_MT

var = 'Q021'

A = 'Não'

B = 'Sim'

plotdata = pd.concat([data[dependentVariable], data[var].map({'A': A, 'B': B})], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y=dependentVariable, data=plotdata, order=[A, B], showmeans=True)

fig.axis(ymin=320);

ax.set(ylabel="Nota de Matemática")

ax.set(xlabel="")

ax.set(title="Na sua residência tem TV por assinatura?")
#box plot: Na sua residência tem acesso à Internet? x NU_NOTA_MT

var = 'Q025'

A = 'Não'

B = 'Sim'

plotdata = pd.concat([data[dependentVariable], data[var].map({'A': A, 'B': B})], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y=dependentVariable, data=plotdata, order=[A, B], showmeans=True)

fig.axis(ymin=320);

ax.set(ylabel="Nota de Matemática")

ax.set(xlabel="")

ax.set(title="Na sua residência tem acesso à Internet?")
#box plot: Até que série seu pai, ou o homem responsável por você, estudou? x NU_NOTA_MT

var = 'Q001'

A = 'Nunca estudou'

B = 'Não completou 5o ano Fundamental'

C = 'Completou 5o ano Fundamental mas não completou 9o ano'

D = 'Completou 5o ano Fundamental mas não completou Médio'

E = 'Completou Médio mas não completou Faculdade'

F = 'Completou Faculdade mas não completou Pós Graduação'

G = 'Completou Pós Graduação'

H = 'Não sei'

plotdata = pd.concat([data[dependentVariable], data[var].map({'A': A, 'B': B, 'C': C, 'D': D, 'E': E, 'F': F, 'G': G, 'H': H})], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y=dependentVariable, data=plotdata, order=[A, B, C, D, E, F, G, H], showmeans=True)

fig.axis(ymin=320);

ax.set_xticklabels(ax.get_xticklabels(),rotation=86)

ax.set(ylabel="Nota de Matemática")

ax.set(xlabel="")

ax.set(title="Até que série seu pai, ou o homem responsável por você, estudou?")



f.savefig("escolaridade.png", bbox_inches='tight', dpi=300)
#box plot: Até que série sua mãe, ou a mulher responsável por você, estudou? x NU_NOTA_MT

var = 'Q002'

A = 'Nunca estudou'

B = 'Não completou 5o ano Fundamental'

C = 'Completou 5o ano Fundamental mas não completou 9o ano'

D = 'Completou 5o ano Fundamental mas não completou Médio'

E = 'Completou Médio mas não completou Faculdade'

F = 'Completou Faculdade mas não completou Pós Graduação'

G = 'Completou Pós Graduação'

H = 'Não sei'

plotdata = pd.concat([data[dependentVariable], data[var].map({'A': A, 'B': B, 'C': C, 'D': D, 'E': E, 'F': F, 'G': G, 'H': H})], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y=dependentVariable, data=plotdata, order=[A, B, C, D, E, F, G, H], showmeans=True)

fig.axis(ymin=320);

ax.set_xticklabels(ax.get_xticklabels(),rotation=86)

ax.set(ylabel="Nota de Matemática")

ax.set(xlabel="")

ax.set(title="Até que série sua mãe, ou a mulher responsável por você, estudou?")
#box plot: Ocupação do Pai x NU_NOTA_MT

var = 'Q003'

A = 'lavrador, etc'

B = 'empregado doméstico, etc'

C = 'padeiro, etc'

D = 'professor, microempresário, etc'

E = 'médico, engenheiro, etc'

F = 'não sei'

plotdata = pd.concat([data[dependentVariable], data[var].map({'A': A, 'B': B, 'C': C, 'D': D, 'E': E, 'F': F})], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y=dependentVariable, data=plotdata, order=[A, B, C, D, E, F], showmeans=True)

fig.axis(ymin=320);

fig.axis(ymin=320);

ax.set_xticklabels(ax.get_xticklabels(),rotation=85)

ax.set(ylabel="Nota de Matemática")

ax.set(xlabel="")

ax.set(title="Ocupação do Pai")
#box plot: Ocupação da Mãe x NU_NOTA_MT

var = 'Q004'

A = 'lavrador, etc'

B = 'empregado doméstico, etc'

C = 'padeiro, etc'

D = 'professor, microempresário, etc'

E = 'médico, engenheiro, etc'

F = 'não sei'

plotdata = pd.concat([data[dependentVariable], data[var].map({'A': A, 'B': B, 'C': C, 'D': D, 'E': E, 'F': F})], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y=dependentVariable, data=plotdata, order=[A, B, C, D, E, F], showmeans=True)

fig.axis(ymin=320);

fig.axis(ymin=320);

ax.set_xticklabels(ax.get_xticklabels(),rotation=85)

ax.set(ylabel="Nota de Matemática")

ax.set(xlabel="")

ax.set(title="Ocupação da Mãe")
#box plot: Incluindo você, quantas pessoas moram atualmente em sua residência? x NU_NOTA_MT

var = 'Q005'

plotdata = pd.concat([data[dependentVariable], data[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y=dependentVariable, data=plotdata)

fig.axis(ymin=320);

ax.set(ylabel="Nota de Matemática")

ax.set(xlabel="")

ax.set(title="Quantas pessoas moram em sua residência?")



#feature engineering: fazer renda per capita
#box plot: Dependência administrativa (Escola) x NU_NOTA_MT

#1 federal, 2 estadual, 3 municipal, 4 privada

var = 'TP_DEPENDENCIA_ADM_ESC'

A = 'lavrador, etc'

B = 'empregado doméstico, etc'

C = 'padeiro, etc'

D = 'professor, microempresário, etc'

plotdata = pd.concat([data[dependentVariable], data[var].map({1.0: 'Federal', 2.0: 'Estadual', 3.0: 'Municipal', 4.0: 'Privada'})], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y=dependentVariable, data=plotdata, order=['Municipal', 'Estadual', 'Federal', 'Privada'], showmeans=True)

fig.axis(ymin=320);

ax.set(ylabel="Nota de Matemática")

ax.set(xlabel="")

ax.set(title="Dependência administrativa da sua Escola")



f.savefig("adm.png", bbox_inches='tight', dpi=600)
#box plot: Idade x NU_NOTA_MT



var = 'NU_IDADE'

plotdata = pd.concat([data[dependentVariable], data[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.scatterplot(x=var, y=dependentVariable, data=plotdata)

fig.axis(ymin=320);

ax.set(ylabel="Nota de Matemática")

ax.set(xlabel="Idade")

ax.set(title="Idade")
#box plot: Sexo x NU_NOTA_MT

var = 'TP_SEXO'

plotdata = pd.concat([data[dependentVariable], data[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y=dependentVariable, data=plotdata, showmeans=True)

fig.axis(ymin=320);

ax.set(ylabel="Nota de Matemática")

ax.set(xlabel="")

ax.set(title="Sexo")



f.savefig("sexo.png", bbox_inches='tight', dpi=400)
#box plot: Cor/raça x NU_NOTA_MT

var = 'TP_COR_RACA'

A = 'Não declarado'

B = 'Branca'

C = 'Preta'

D = 'Parda'

E = 'Amarela'

F = 'Indígena'

plotdata = pd.concat([data[dependentVariable], data[var].map({0: A, 1: B, 2: C, 3: D, 4: E, 5: F})], axis=1)

f, ax = plt.subplots(figsize=(10, 6))

fig = sns.boxplot(x=var, y=dependentVariable, data=plotdata, order=[A, B, C, D, E, F], showmeans=True)

fig.axis(ymin=320);

ax.set(ylabel="Nota de Matemática")

ax.set(xlabel="")

ax.set(title="Cor/Raça")
#box plot: Tipo de escola do Ensino Médio x NU_NOTA_MT

#1 nao responder 2pública 3privada

var = 'TP_ESCOLA'

A = 'Não respondeu'

B = 'Pública'

C = 'Privada'

plotdata = pd.concat([data[dependentVariable], data[var].map({1: A, 2: B, 3:C})], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y=dependentVariable, data=plotdata, showmeans=True, order=[A,B,C])

fig.axis(ymin=320);

ax.set(ylabel="Nota de Matemática")

ax.set(xlabel="")

ax.set(title="Tipo de escola do Ensino Médio")
#box plot: Código da Unidade da Federação de residência x NU_NOTA_MT



var = 'CO_UF_RESIDENCIA'

plotdata = pd.concat([data[dependentVariable], data[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y=dependentVariable, data=plotdata)

fig.axis(ymin=320);

ax.set(ylabel="Nota de Matemática")

ax.set(xlabel="")

ax.set(title="Estado de residência")
#box plot: Estado Civil x NU_NOTA_MT

var = 'TP_ESTADO_CIVIL'

A = 'Não informado'

B = 'Solteiro'

C = 'Casado/mora junto'

D = 'Separado'

E = 'Viúvo'

plotdata = pd.concat([data[dependentVariable], data[var].map({0: A, 1: B, 2: C, 3: D, 4: E})], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y=dependentVariable, data=plotdata, order=[A, B, C, D, E])

fig.axis(ymin=320);

ax.set(ylabel="Nota de Matemática")

ax.set(xlabel="")

ax.set(title="Estado Civil")
#box plot: Nacionalidade x NU_NOTA_MT

var = 'TP_NACIONALIDADE'

A = 'Não informado'

B = 'brasileiro'

C = 'brasileiro naturalizado'

D = 'estrangeiro'

E = 'brasileiro nato nascido no exterior'

plotdata = pd.concat([data[dependentVariable], data[var].map({0: A, 1: B, 2: C, 3: D, 4: E})], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y=dependentVariable, data=plotdata, order=[A, B, C, D, E])

fig.axis(ymin=320);

ax.set_xticklabels(ax.get_xticklabels(),rotation=80)

ax.set(ylabel="Nota de Matemática")

ax.set(xlabel="")

ax.set(title="Nacionalidade")
#box plot: Situação de conclusão do Ensino Médio x NU_NOTA_MT

#já concluiu ensino médio?está cursando?

var = 'TP_ST_CONCLUSAO'

A = 'Já concluí'

B = 'Concluirei em 2019 '

C = 'Concluirei após 2019'

D = 'Não concluí nem estou cursando'

plotdata = pd.concat([data[dependentVariable], data[var].map({1: A, 2: B, 3: C, 4: D})], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y=dependentVariable, data=plotdata, order=[A, B, C, D])

fig.axis(ymin=320);

ax.set_xticklabels(ax.get_xticklabels(),rotation=80)

ax.set(ylabel="Nota de Matemática")

ax.set(xlabel="")

ax.set(title="Situação de conclusão do Ensino Médio")
cols_candidatos = ['NU_INSCRICAO', 'NU_NOTA_MT', 'Q001', 'Q002', 'Q003', 'Q004','Q005', 'Q006', 'Q007', 'Q010', 'Q021', 'TP_SEXO', 'TP_COR_RACA', 'TP_DEPENDENCIA_ADM_ESC']



#missing data

model_data = data[cols_candidatos]



total = model_data.isnull().sum().sort_values(ascending=False)

print('missing data:\n', total)

#percent = (model_data.isnull().sum()/model_data.isnull().count()).sort_values(ascending=False)

#missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

#missing_data



percent = model_data['TP_DEPENDENCIA_ADM_ESC'].isnull().sum()/len(model_data.index)*100

print('\n missing TP_DEPENDENCIA_ADM_ESC: '+ str(round(percent,1)) +'%')
total = len(model_data.index)



#str(round(percent,1)) +'%'



r1 = model_data[(model_data['Q002'] == 'H') & (model_data['Q001'] == 'H')]['Q002'].count()

r1_percent = ' ('+str(round(r1*100/total,1))+'%)'

print('respondentes que não sabem a escolaridade da mãe nem do pai: '+ str(r1)+r1_percent)



r0 = model_data[(model_data['Q002'] == 'H') | (model_data['Q001'] == 'H')]['Q002'].count() - 2*r1

r0_percent = ' ('+str(round(r0*100/total,1))+'%)'

print('respondentes que não sabem a escolaridade da mãe mas sabem do pai, ou vice versa: '+str(r0)+r0_percent)



r2 = model_data[(model_data['Q004'] == 'F') & (model_data['Q003'] == 'F')]['Q002'].count()

r2_percent = ' ('+str(round(r2*100/total,1))+'%)'

print('\nrespondentes que não sabem a ocupação da mãe nem do pai: '+ str(r2)+r2_percent)



#r3 = model_data[((model_data['Q002'] == 'H') & (data['Q001'] == 'H') & (model_data['Q004'] == 'F') & (data['Q003'] == 'F'))]['Q002'].count()

#print('\nunião entre os grupos acima: ', r1+r2-r3)



r4 = model_data[(model_data['TP_COR_RACA'] == 0)]['Q002'].count()

r4_percent = ' ('+str(round(r4*100/total,1))+'%)'

print('\nrespondentes que não declararam raça: '+ str(r4)+r4_percent)
#Eliminando candidatos que nao sabem a escolaridade do pai nem da mae

model_data2 = model_data[(model_data['Q002'] != 'H') | (model_data['Q001'] != 'H')]



#Manteremos as seguintesc colunas: escolaridade do pai e mãe, renda, número de residentes, empregado doméstico, carro, TV a cabo e sexo.

cols = ['Q001', 'Q002', 'Q006','Q005', 'Q007', 'Q010', 'Q021', 'TP_SEXO', 'TP_COR_RACA']



X3 = model_data2[cols]
print('Valores existentes e quantidade de cada valor em cada coluna')

print(X3['Q001'].value_counts())

print(X3['Q002'].value_counts())

print(X3['Q005'].value_counts())

print(X3['Q006'].value_counts())

print(X3['Q007'].value_counts())

print(X3['Q010'].value_counts())

print(X3['Q021'].value_counts())

print(X3['TP_SEXO'].value_counts())

print(X3['TP_COR_RACA'].value_counts())
X3_labels = X3.copy()



#feature eng 1: max_escol

#primeiro vamos mapear os níveis de escolaridade A, ..., G, H em números. Como H = "não sei", colocaremos como -1

X3_labels['Q001'] = X3_labels['Q001'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H':-1})

X3_labels['Q002'] = X3_labels['Q002'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H':-1})

#agora vamos criar uma coluna que é a maior escolaridade entre pai e mae

X3_labels['max_escol'] = X3_labels[['Q001', 'Q002']].max(axis=1)

#e jogar fora as duas colunas originais de escolaridade do pai e da mãe

X3_labels.drop(['Q001', 'Q002'], axis=1, inplace=True)



#X3_labels.head()



#FAZENDO OS LABELS DAS OUTRAS COLUNAS



X3_labels['TP_SEXO'] = X3_labels['TP_SEXO'].map({'M': 0, 'F': 1})

#X3_labels['TP_SEXO'].value_counts()

X3_labels['Q006'] = X3_labels['Q006'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H':7, 'I': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16})

#X3_labels['Q006'].value_counts()

X3_labels['Q007'] = X3_labels['Q007'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3})

#X3_labels['Q007'].value_counts()

X3_labels['Q010'] = X3_labels['Q010'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4})

#X3_labels['Q010'].value_counts()

X3_labels['Q021'] = X3_labels['Q021'].map({'A': 0, 'B': 1})

#X3_labels['Q021'].value_counts()



#feature eng 2: renda per capita

rendaDict = {

    0: [0,0],

    1: [0,998],

    2: [998.01, 1497],

    3: [1497.01, 1996],

    4: [1996.01, 2495],

    5: [2495.01, 2994],

    6: [2994.01, 3992],

    7: [3992.01, 4990],

    8: [4990.01, 5988],

    9: [5988.01, 6986],

    10: [6986.01, 7984],

    11: [7984.01, 8982],

    12: [8982.01, 9980],

    13: [9980.01, 11976],

    14: [11976.01, 14970],

    15: [14970.01, 19960],

    16: [19960.01, 9000000]    

}





rendaPerCapitaDict = {

0: [0, 0],

1: [0, 200],

2: [200.01, 400],

3: [400.01, 600],

4: [600.01, 800],

5: [800.01, 1000],

6: [1000.01, 1200],

7: [1200.01, 1400],

8: [1400.01, 1600],

9: [1600.01, 1800],

10: [1800.01, 2000],

11: [2000.01, 2400],

12: [2400.01, 2800],

13: [2800.01, 3200] ,   

14: [3200.01, 3600],

15: [3600.01, 4000],

16: [4000.01, 4800],

17: [4800.01, 5600],

18: [5600.01, 9000000]

}



def calcularRendaPerCapitaMinima(classeRenda, residentes):

    if classeRenda == 0:

        return -1

    else:

        renda = np.mean([rendaDict[classeRenda][0], rendaDict[classeRenda][1]])

        rendaPerCapita = renda/residentes

        return rendaPerCapita



#r = calcularRendaPerCapitaMinima(16, 1)

#print(r)



def calcularclassePerCapita(rendaPerCapita):

    if rendaPerCapita == -1:

        return 0

    else:

        for classe in rendaPerCapitaDict:

            #print(str(classe)+' '+str(rendaPerCapitaDict[classe][0])+' '+str(rendaPerCapitaDict[classe][1]))

            if rendaPerCapita >= rendaPerCapitaDict[classe][0] and rendaPerCapita < rendaPerCapitaDict[classe][1]:

                return classe



def criarClassePerCapita(cols):

    classeRenda = cols[0]

    residentes = cols[1]

    #determina a renda per capita mínima desse estudante

    rendaPerCapitaMinima = calcularRendaPerCapitaMinima(classeRenda, residentes)

    #determina qual é a classe de renda per capita onde aquele aluno deve ser colocado

    if rendaPerCapitaMinima == -1:

        return 0

    else:

        return calcularclassePerCapita(rendaPerCapitaMinima)



#print('rendaPerCapitaMinima', calcularRendaPerCapitaMinima(10,6))

#print('classe per capita', calcularclassePerCapita(calcularRendaPerCapitaMinima(10,6)))

#print('criar classe per capita', criarClassePerCapita([10,6]))



X3_labels['perCapita'] = X3_labels[['Q006', 'Q005']].apply(criarClassePerCapita, axis=1)





#dummies de raça

X3_labels = pd.get_dummies(X3_labels, columns=['TP_COR_RACA'])

#eliminando a coluna de one hot encoding de quem respondeu "não declarado" para raça

X3_labels.drop('TP_COR_RACA_0', axis=1, inplace=True)

#X3_labels.head()
#DELETAR

#feature eng 2: renda per capita

rendaDict = {

    0: [0,0],

    1: [0,998],

    2: [998.01, 1497],

    3: [1497.01, 1996],

    4: [1996.01, 2495],

    5: [2495.01, 2994],

    6: [2994.01, 3992],

    7: [3992.01, 4990],

    8: [4990.01, 5988],

    9: [5988.01, 6986],

    10: [6986.01, 7984],

    11: [7984.01, 8982],

    12: [8982.01, 9980],

    13: [9980.01, 11976],

    14: [11976.01, 14970],

    15: [14970.01, 19960],

    16: [19960.01, 9000000]    

}





rendaPerCapitaDict = {

0: [0, 0],

1: [0, 200],

2: [200.01, 400],

3: [400.01, 600],

4: [600.01, 800],

5: [800.01, 1000],

6: [1000.01, 1200],

7: [1200.01, 1400],

8: [1400.01, 1600],

9: [1600.01, 1800],

10: [1800.01, 2000],

11: [2000.01, 2400],

12: [2400.01, 2800],

13: [2800.01, 3200] ,   

14: [3200.01, 3600],

15: [3600.01, 4000],

16: [4000.01, 4800],

17: [4800.01, 5600],

18: [5600.01, 9000000]

}



def calcularRendaPerCapitaMinima(classeRenda, residentes):

    if classeRenda == 0:

        return -1

    else:

        renda = (rendaDict[classeRenda][0]+ rendaDict[classeRenda][1])/2

        rendaPerCapita = renda/residentes

        return rendaPerCapita



#r = calcularRendaPerCapitaMinima(16, 1)

#print(r)



def calcularclassePerCapita(rendaPerCapita):

    if rendaPerCapita == -1:

        return 0

    else:

        for classe in rendaPerCapitaDict:

            #print(str(classe)+' '+str(rendaPerCapitaDict[classe][0])+' '+str(rendaPerCapitaDict[classe][1]))

            if rendaPerCapita >= rendaPerCapitaDict[classe][0] and rendaPerCapita < rendaPerCapitaDict[classe][1]:

                return classe



def criarClassePerCapita(classeRenda, residentes):

    #determina a renda per capita mínima desse estudante

    rendaPerCapitaMinima = calcularRendaPerCapitaMinima(classeRenda, residentes)

    #determina qual é a classe de renda per capita onde aquele aluno deve ser colocado

    if rendaPerCapitaMinima == -1:

        return 0

    else:

        return calcularclassePerCapita(rendaPerCapitaMinima)



criarClassePerCapita(10,2)
#box plot: renda per capita x NU_NOTA_MT

var = 'perCapita'

plotdata = pd.concat([model_data2['NU_NOTA_MT'], X3_labels[var]], axis=1)

f, ax = plt.subplots(figsize=(14, 6))

fig = sns.boxplot(x=var, y='NU_NOTA_MT', data=plotdata)

fig.axis(ymin=320);

ax.set_xticklabels(ax.get_xticklabels(),rotation=80)

ax.set(ylabel="Nota de Matemática")

ax.set(xlabel="")

ax.set(title="Renda per capita")

f.savefig("perCapita.png", bbox_inches='tight', dpi=600)





#box plot: max_escol x NU_NOTA_MT

var = 'max_escol'

A = 'Nunca estudou'

B = 'Não completou 5o ano Fundamental'

C = 'Completou 5o ano Fundamental mas não completou 9o ano'

D = 'Completou 5o ano Fundamental mas não completou Médio'

E = 'Completou Médio mas não completou Faculdade'

F = 'Completou Faculdade mas não completou Pós Graduação'

G = 'Completou Pós Graduação'

H = 'Não sei'

plotdata = pd.concat([model_data2['NU_NOTA_MT'], X3_labels[var].map({0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, -1: H})], axis=1)

f, ax = plt.subplots(figsize=(14, 6))

fig = sns.boxplot(x=var, y='NU_NOTA_MT', data=plotdata, order=[A, B, C, D, E, F, G])

fig.axis(ymin=320);

ax.set_xticklabels(ax.get_xticklabels(),rotation=80)

ax.set(ylabel="Nota de Matemática")

ax.set(xlabel="")

ax.set(title="Maior escolaridade entre pai e mãe")

f.savefig("max_escol.png", bbox_inches='tight', dpi=400)
#Separando nosso dataset entre treinamento e validação



#Notas de Matemática (variável dependente)

y = model_data2['NU_NOTA_MT']



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X3_labels, y, test_size=0.3, random_state=101)
from xgboost import XGBRegressor



xgbr = XGBRegressor(n_estimators=1000)

xgbr_model = xgbr.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_test, y_test)], verbose=0)

            



#plotando as feature importances

from xgboost import plot_importance



plot_importance(xgbr, show_values=False, importance_type="weight")

plt.title("Feature Importances by weight")

plt.show()

f.savefig("perCapita.png", bbox_inches='tight', dpi=600)





plot_importance(xgbr, show_values=False, importance_type="cover")

plt.title("Feature Importances by cover")

plt.show()





plot_importance(xgbr, show_values=False, importance_type="gain")

plt.title("Feature Importances by gain")

plt.show()



#avaliando métricas do modelo

predictions = xgbr.predict(X_test)



print('Resultado XGBoost')

print('MAE:', metrics.mean_absolute_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))



sns.distplot((y_test-predictions),bins=50)
import lightgbm as lgb



#treinando o modelo

train_data = lgb.Dataset(X_train, label=y_train)

validation_data = lgb.Dataset(X_test, label=y_test)



params = {}

#params['learning_rate'] = 0.01



params['metric'] = 'rmse'

params['verbose'] = 0



bst = lgb.train(params, train_data, 1000, valid_sets=validation_data, early_stopping_rounds=5, verbose_eval=False)



#plotando as feature importances

from lightgbm import plot_importance

plot_importance(bst, importance_type="split", figsize=(10,3), grid=False)

plt.title("Feature Importances by split")

plt.show()



plot_importance(bst, importance_type="gain", figsize=(10,3), grid=False, precision=0)

plt.title("Feature Importances by gain")

plt.show()





#avaliando métricas do modelo

predictions = bst.predict(X_test)



print('Resultado LightGBM')

print('MAE:', metrics.mean_absolute_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))



sns.distplot((y_test-predictions),bins=50)
import shap



bst.params['objective'] = 'regression'

explainer = shap.TreeExplainer(bst)

shap_values = explainer.shap_values(X_train[:30000])



shap.summary_plot(shap_values, X_train.columns, plot_type="bar")
#work around para o problema do codec

#https://github.com/slundberg/shap/issues/1215



mybooster = xgbr_model.get_booster()





model_bytearray = mybooster.save_raw()[4:]

def myfun(self=None):

    return model_bytearray



mybooster.save_raw = myfun



import shap



# this takes a minute or two since we are explaining over 30 thousand samples in a model with over a thousand trees

explainer2 = shap.TreeExplainer(xgbr_model)

shap_values = explainer.shap_values(X_train[:30000])



shap.summary_plot(shap_values, X_train.columns, plot_type="bar")
X_final = X3_labels[['Q005', 'Q006', 'TP_SEXO', 'max_escol', 'perCapita', 'Q010']].copy()

X_final.head()

y = model_data2['NU_NOTA_MT']



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.3, random_state=101)



import lightgbm as lgb



params = {}

#params['learning_rate'] = 0.01

#params['objective'] = 'regression'

params['metric'] = 'rmse'

params['verbose'] = 0



#incluindo 'carros'

train_data = lgb.Dataset(X_train, label=y_train)

validation_data = lgb.Dataset(X_test, label=y_test)

bst = lgb.train(params, train_data, 1000, valid_sets=validation_data, early_stopping_rounds=5, verbose_eval=False)



#avaliando métricas do modelo com carros

predictions = bst.predict(X_test)

print('incluindo a feature quantidade de carros:')

print('MAE:', metrics.mean_absolute_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))



#excluindo 'carros'

X_final = X3_labels[['Q005', 'Q006', 'TP_SEXO', 'max_escol', 'perCapita']].copy()

X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.3, random_state=101)



train_data = lgb.Dataset(X_train, label=y_train)

validation_data = lgb.Dataset(X_test, label=y_test)

bst = lgb.train(params, train_data, 1000, valid_sets=validation_data, early_stopping_rounds=5, verbose_eval=False)



#avaliando métricas do modelo com carros

predictions = bst.predict(X_test)

print('\nexcluindo a feature quantidade de carros:')

print('MAE:', metrics.mean_absolute_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))



sns.distplot((y_test-predictions),bins=50)
##exportando os datasets finais

#X_final = X3_labels[['Q005', 'Q006', 'TP_SEXO', 'max_escol', 'perCapita']].copy()

#y_final = model_data2['NU_NOTA_MT']



#y_final.to_csv('y_final.csv', index=False)

#X_final.to_csv('X_final.csv', index=False)
#Função Objetivo

import lightgbm as lgb

from hyperopt import STATUS_OK



X_final = X3_labels[['Q005', 'Q006', 'TP_SEXO', 'max_escol', 'perCapita']].copy()

y_final = model_data2['NU_NOTA_MT']



#split 80/10/10

X_train, X_val, y_train, y_val = train_test_split(X_final, y_final, test_size=0.1, random_state=101)

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1111111111, random_state=101)





train_set = lgb.Dataset(X_train, y_train)

val_set = lgb.Dataset(X_val, y_val)

test_set = lgb.Dataset(X_test, y_test)



def objective(params):

    """Objective function for Gradient Boosting Machine Hyperparameter Tuning"""

    

    params['metric'] = 'rmse'

    

    params['subsample'] = params['boosting_type']['subsample']

    boost_type = params['boosting_type']['boosting_type']

    del params['boosting_type']

    params['boosting_type'] = boost_type

    

    params['num_leaves'] = int(params['num_leaves'])

    params['min_child_samples'] = int(params['min_child_samples'])

    params['subsample_for_bin'] = int(params['subsample_for_bin'])

    

    

    

    

    

    eval_metrics = {}

    bst = lgb.train(params, train_set, 1000, valid_sets=val_set, early_stopping_rounds=10, verbose_eval=False, evals_result=eval_metrics, valid_names=['validation_data'])

  

    # Extract the best score

    loss = eval_metrics['validation_data']['rmse'][len(eval_metrics['validation_data']['rmse']) - 1]

    

    

    

    # Dictionary with information for evaluation

    return {'loss': loss, 'params': params, 'status': STATUS_OK}
f, ax = plt.subplots()



sns.distplot(y_train, label='train')

sns.distplot(y_val, label='dev')

sns.distplot(y_test, label='test')

plt.legend()

plt.title('train/dev/test set possuem mesma distribuição de notas')

plt.show()

f.savefig("trantestdev2", bbox_inches='tight', dpi=500)
#Espaço de hiperparâmetros a ser explorado

from hyperopt import hp



# Define the search space

space = {

    'boosting_type': hp.choice('boosting_type', [{'boosting_type': 'gbdt', 'subsample': hp.uniform('gdbt_subsample', 0.5, 1)}, 

                                                 {'boosting_type': 'goss', 'subsample': 1.0}]),

    'num_leaves': hp.quniform('num_leaves', 20, 150, 1),

    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),

    'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),

    'min_child_samples': hp.quniform('min_child_samples', 20, 500, 5),

    'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),

    'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),

    'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0)

}
#teste: sorteando um conjunto de hiperparâmetros

from hyperopt.pyll.stochastic import sample



# Sample from the full space

example = sample(space)



# Dictionary get method with default

subsample = example['boosting_type'].get('subsample', 1.0)



# Assign top-level keys

example['boosting_type'] = example['boosting_type']['boosting_type']

example['subsample'] = subsample



example
#Fazendo a busca

MAX_EVALS = 500



from hyperopt import tpe

# Algorithm

tpe_algorithm = tpe.suggest



from hyperopt import Trials

# Trials object to track progress

bayes_trials = Trials()



from hyperopt import fmin



# Optimize

best = fmin(fn = objective, space = space, algo = tpe.suggest, 

            max_evals = MAX_EVALS, trials = bayes_trials)
#treinando o melhor modelo



#best['boosting_type'] = 'gbdt'

#best['min_child_samples'] = int(best['min_child_samples'])

#best['num_leaves'] = int(best['num_leaves'])

#best['subsample_for_bin'] = int(best['subsample_for_bin'])



#hiperparâmetros ótimos obtidos após 500 modelos treinados com a busca bayesiana

opt_params = {'boosting_type': 'gbdt',

          'colsample_by_tree': 0.7971305836421847,

          'gdbt_subsample': 0.7849750176469992,

          'learning_rate': 0.1226204673784012,

          'min_child_samples': 275,

          'num_leaves': 20,

          'reg_alpha': 0.28240810762516616,

          'reg_lambda': 0.8413708118330061,

          'subsample_for_bin': 300000,

          'metric': 'rmse'}



#para não precisar rodar a busca bayesiana completa, coloquei acima, estaticamente, os hiperparâmetros ótimos

#opt_params = best

#opt_params['metric'] = 'rmse'



eval_metrics = {}

modelo_final = lgb.train(opt_params, train_set, 1000, valid_sets=val_set, early_stopping_rounds=10, verbose_eval=False, evals_result=eval_metrics, valid_names=['validation_data'])

  

    # Extract the best score

loss = eval_metrics['validation_data']['rmse'][len(eval_metrics['validation_data']['rmse']) - 1]

    

print({'loss': loss, 'params': opt_params})
#avaliando métricas do modelo

predictions = modelo_final.predict(X_test)



print('Performance final no test set (exemplos nunca vistos pelo modelo):')

print('MAE:', metrics.mean_absolute_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))



sns.distplot((y_test-predictions),bins=50)
import time

from joblib import dump, load





model_name = 'LightGBM_005.joblib'

#dump(modelo_final, model_name)
import shap



#porção do train_set usada para estudar o SHAP

X_train_for_shap = X_train



#é necessário declararmos explicitamente o parâmetro 'objective' para a biblioteca SHAP saber que tarefa estamos realizando

modelo_final.params['objective'] = 'regression'



#calculando os valores de SHAP para cada feature de cada candidato

explainer = shap.TreeExplainer(modelo_final)

shap_values = explainer.shap_values(X_train_for_shap)



#valores de shap para cada usuário

shap_df = pd.DataFrame(shap_values, columns=X_train.columns)



#somando os shap values de todas as features de cada candidato

shap_sum = np.round(shap_df.sum(axis=1),2)



#fazendo a previsão da nota para cada candidato

predictions = np.round(modelo_final.predict(X_train_for_shap),2)



#criando dataframe com os shaps, shap_sum e prediction para cada ponto amostral

df = pd.concat([np.round(shap_df,2), pd.Series(shap_sum, name='SHAP_sum'), pd.Series(predictions, name='prediction')], axis=1)

df.head()
#plot da soma dos valores de shap contra a previsão da nota

#sns.scatterplot(x=shap_sum, y=predictions)



plt.scatter(shap_sum, predictions)

plt.xlabel('Soma do SHAP para o candidato',fontsize=13)

plt.ylabel('Nota Prevista para o candidato ',fontsize=13)

plt.title('Previsão de nota a partir do SHAP',fontsize=14)
shap.summary_plot(shap_values, X_train.columns, plot_type="bar")
#plotando os valores de SHAP para cada feature de cada candidato

shap.summary_plot(shap_values, X_train_for_shap)
shap.dependence_plot('TP_SEXO', shap_values, X_train_for_shap)
shap.dependence_plot('Q006', shap_values, X_train_for_shap)
shap.initjs()

shap.force_plot(explainer.expected_value, shap_values[2], X_train_for_shap.iloc[2])
#vendo os valores de shap para cada usuário

shap_df = pd.DataFrame(shap_values, columns=X_train.columns)

shap_df.head()
#IGNORAR

Q006 = 10

Q005 = 2

TP_SEXO = 1

max_escol = 1

perCapita = 15



novoExemplo = {'Q006': [Q006], 'Q005': [Q005],'TP_SEXO': [TP_SEXO], 'max_escol': [max_escol], 'perCapita': [perCapita]}

novoExemplo = pd.DataFrame(novoExemplo)



r = modelo_final.predict(novoExemplo)

r = r[0]

r = round(r,2)

r