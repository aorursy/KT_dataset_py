import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style()
cores_esporte = ['#BE81F7','#81F781','#F7BE81','#F78181','#F3F781']
paleta_esporte = sns.color_palette(cores_esporte)
cores_genero = ['#9999ff','#ff99cc']
paleta_genero = sns.color_palette(cores_genero)
color_orange = '#F7BE81'
color_red = '#F78181'
color_yellow = '#F3F781'
df=pd.read_csv("../input/olympic-games/athletes.csv")
df.head()

#Excluindo NaN
df.columns= ['id','nome','nacionalidade','genero','data_nascimento','altura','peso','esporte','ouro','prata','bronze']
df.fillna(df.mean())
df['peso'] = df['peso'].fillna(df['peso'].mean())
df['altura'] = df['altura'].fillna(df['altura'].mean())
df
df['esporte'].value_counts()[:5]
df_agrupado = df[['altura', 'peso']].groupby(by=df['esporte']).mean().round(2)
df_agrupado
print(f'''Máximos e mínimos:
Altura: Máximo ({df_agrupado['altura'].max()}m) --- Mínimo ({df_agrupado['altura'].min()}m)
Peso: Máximo ({df_agrupado['peso'].max()}kg) --- Mínimo ({df_agrupado['peso'].min()}kg)''')
df_agrupado_medalhas = df[['ouro', 'prata', 'bronze']].groupby(by=df['esporte']).sum()
df_agrupado_medalhas.sort_values('ouro', ascending= False)
df_agrupado_genero = df[['altura', 'peso']].groupby(by=df['genero']).mean().round(2)
df_agrupado_genero.T
df_agrupado_genero_medalha = df[['ouro', 'prata', 'bronze']].groupby(by=df['genero']).sum()
df_agrupado_genero_medalha.T
df_agrupado_genero_medalha = df[['ouro', 'prata', 'bronze']].groupby(by=df['genero']).mean()
df_agrupado_genero_medalha.T
#Genero dos competidores do atletismo
homens_atl = df[(df['esporte'] == 'athletics') & (df['genero'] == 'male')]
mulheres_atl = df[(df['esporte'] == 'athletics') & (df['genero'] == 'female')]
homens_atl_porc = homens_atl.count()[0] / (mulheres_atl.count()[0] + homens_atl.count()[0])*100
mulheres_atl_porc = mulheres_atl.count()[0] / (mulheres_atl.count()[0] + homens_atl.count()[0])*100
print(f'''Homens no atletismo: {homens_atl.count()[0]} ({homens_atl_porc:.2f}%)
Mulheres no atletismo: {mulheres_atl.count()[0]} ({mulheres_atl_porc:.2f}%)''')
homens_atl = df[(df['esporte'] == 'athletics') & (df['genero'] == 'male')]
mulheres_atl = df[(df['esporte'] == 'athletics') & (df['genero'] == 'female')]
masc_atl = homens_atl.count()[0]
fem_atl = mulheres_atl.count()[1]
if masc_atl > fem_atl:
  primeiro = 'MASCULINO'
  segundo = 'FEMININO'
else:
  primeiro = 'FEMININO'
  segundo = 'MASCULINO'
gen_atl = [masc_atl, fem_atl]
fig = plt.figure(figsize=(7,7))
sns.set_style('ticks')
plt.pie(gen_atl, labels=[primeiro, segundo], colors= paleta_genero, autopct='%1.1f%%',shadow=True)
plt.title('Gênero dos atletas (Atletismo)',fontsize=21);
homens_atl['ouro'].sum()
homens_atl['prata'].sum()
homens_atl['bronze'].sum()
mulheres_atl['ouro'].sum()
mulheres_atl['prata'].sum()
mulheres_atl['bronze'].sum()
print(f'''HOMENS:
Ouro:    {homens_atl['ouro'].sum()}
Prata:   {homens_atl['prata'].sum()}
Bronze:  {homens_atl['bronze'].sum()}
Total:   {homens_atl['ouro'].sum() + homens_atl['prata'].sum() + homens_atl['bronze'].sum()}
Média por atleta: {(homens_atl['ouro'].sum() + homens_atl['prata'].sum() + homens_atl['bronze'].sum()) / homens_atl.count()[0]:.3f}

-----------
MULHERES:
Ouro:    {mulheres_atl['ouro'].sum()}
Prata:   {mulheres_atl['prata'].sum()}
Bronze:  {mulheres_atl['bronze'].sum()}
Total:   {mulheres_atl['ouro'].sum() + mulheres_atl['prata'].sum() + mulheres_atl['bronze'].sum()}
Média por atleta: {(mulheres_atl['ouro'].sum() + mulheres_atl['prata'].sum() + mulheres_atl['bronze'].sum()) / mulheres_atl.count()[0]:.3f}
''')
df_agrupado_medalha_atl = df[['ouro', 'prata', 'bronze']].groupby(by=df['esporte'] == 'athletics').sum()
df_agrupado_medalha_atl = df_agrupado_medalha_atl.drop(df_agrupado_medalha_atl[df_agrupado_medalha_atl.ouro > 100].index)
df_agrupado_medalha_atl
homens_atl['altura'].mean()
homens_atl['peso'].mean()
print(f'''HOMENS:
Altura média: {homens_atl['altura'].mean():.2f}
Peso médio:   {homens_atl['peso'].mean():.2f}

MULHERES:
Altura média: {mulheres_atl['altura'].mean():.2f}
Peso médio:   {mulheres_atl['peso'].mean():.2f}''')
df[df['esporte']=='athletics']
df_aquat = df[(df['esporte']=='aquatics')]
aquat_nacionalidade =df_aquat['nacionalidade'].value_counts()[:10]
aquat_nacionalidade
fig = plt.figure(figsize=(15,5))
sns.countplot(x='nacionalidade',palette=sns.light_palette("purple",13,reverse=True), order = aquat_nacionalidade.index, data= df_aquat)
sns.despine(left=True)
plt.xlabel('')
plt.title('Países com mais competidores no atletismo',fontsize=21)
plt.ylabel('')
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.xlim(-0.5, 9.5);
df_atl = df[(df['esporte']=='athletics')]
atl_nacionalidade =df_atl['nacionalidade'].value_counts()[:10]
atl_nacionalidade
sns.set_style('white')
color_green = '#81F781'
color_purple = '#BE81F7'
fig =plt.figure(figsize=(15,5))
sns.boxplot(x='nacionalidade', y='altura',linewidth= 2, order = atl_nacionalidade.index,color = color_purple, fliersize= False, data=df_atl)
plt.xlim(-0.5,9.5)
sns.despine(left=True)
plt.xlabel('Nacionalidade',fontsize=15)
plt.title('Distribuição das alturas dos atletas no atletismo',fontsize=21)
plt.ylabel('Altura em metros',fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.ylim(1.3,2.2);
sns.set_style('white')
color_green = '#81F79F'
fig =plt.figure(figsize=(15,5))
sns.boxplot(x='nacionalidade', y='altura',linewidth= 2,hue='genero', order = atl_nacionalidade.index,palette=paleta_genero, fliersize= False, data=df_atl)
plt.xlim(-0.5,9.5)
sns.despine(left=True)
plt.xlabel('Nacionalidade',fontsize=15)
plt.title('Distribuição da altura dos atletas no atletismo (por gênero)',fontsize=21)
plt.ylabel('Altura em metros',fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.ylim(1.3,2.2);
sns.set_style('white')
color_green = '#81F79F'
fig =plt.figure(figsize=(15,5))
sns.boxplot(x='nacionalidade', y='peso',linewidth= 2.5, order = atl_nacionalidade.index,color = color_purple, fliersize= False, data=df_atl)
plt.xlim(-0.5,9.5)
sns.despine(left=True)
plt.xlabel('Nacionalidade',fontsize=15)
plt.title('Distribuição dos pesos dos atletas no atletismo',fontsize=21)
plt.ylabel('Peso em Kg',fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.ylim(0,130);
sns.set_style('white')
color_green = '#81F79F'
fig =plt.figure(figsize=(15,5))
sns.boxplot(x='nacionalidade', y='peso',linewidth= 2,hue='genero', order = atl_nacionalidade.index,palette=paleta_genero, fliersize= False, data=df_atl)
plt.xlim(-0.5,9.5)
sns.despine(left=True)
plt.xlabel('Nacionalidade',fontsize=15)
plt.ylabel('Peso em Kg',fontsize=15)
plt.title('Distribuição dos pesos dos atletas do atletismo (por gênero)')
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.ylim(0,130);
homens_aqu = df[(df['esporte'] == 'aquatics') & (df['genero'] == 'male')]
mulheres_aqu = df[(df['esporte'] == 'aquatics') & (df['genero'] == 'female')]
homens_aqu_porc = homens_aqu.count()[0] / (mulheres_aqu.count()[0] + homens_aqu.count()[0])*100
mulheres_aqu_porc = mulheres_aqu.count()[0] / (mulheres_aqu.count()[0] + homens_aqu.count()[0])*100
print(f'''Homens no esporte aquático: {homens_aqu.count()[0]} ({homens_aqu_porc:.2f}%)
Mulheres no esporte aquático: {mulheres_aqu.count()[0]} ({mulheres_aqu_porc:.2f}%)''')
homens_aqu = df[(df['esporte'] == 'aquatics') & (df['genero'] == 'male')]
mulheres_aqu = df[(df['esporte'] == 'aquatics') & (df['genero'] == 'female')]
masc_aqu = homens_aqu.count()[0]
fem_aqu = mulheres_aqu.count()[1]
if masc_aqu > fem_aqu:
  primeiro = 'MASCULINO'
  segundo = 'FEMININO'
else:
  primeiro = 'FEMININO'
  segundo = 'MASCULINO'
gen_aqu = [masc_aqu, fem_aqu]
fig = plt.figure(figsize=(7,7))
sns.set_style('ticks')
plt.pie(gen_aqu, labels=[primeiro, segundo], colors= paleta_genero, autopct='%1.1f%%',shadow=True)
plt.title('Gênero dos atletas (esporte aquático)',fontsize=21);
homens_aqu['ouro'].sum()
homens_aqu['prata'].sum()
homens_aqu['bronze'].sum()
mulheres_aqu['ouro'].sum()
mulheres_aqu['prata'].sum()
mulheres_aqu['bronze'].sum()
print(f'''HOMENS:
Ouro:    {homens_aqu['ouro'].sum()}
Prata:   {homens_aqu['prata'].sum()}
Bronze:  {homens_aqu['bronze'].sum()}
Total:   {homens_aqu['ouro'].sum() + homens_aqu['prata'].sum() + homens_aqu['bronze'].sum()}
Média por atleta: {(homens_aqu['ouro'].sum() + homens_aqu['prata'].sum() + homens_aqu['bronze'].sum()) / homens_aqu.count()[0]:.3f}

-----------
MULHERES:
Ouro:    {mulheres_aqu['ouro'].sum()}
Prata:   {mulheres_aqu['prata'].sum()}
Bronze:  {mulheres_aqu['bronze'].sum()}
Total:   {mulheres_aqu['ouro'].sum() + mulheres_aqu['prata'].sum() + mulheres_aqu['bronze'].sum()}
Média por atleta: {(mulheres_aqu['ouro'].sum() + mulheres_aqu['prata'].sum() + mulheres_aqu['bronze'].sum()) / mulheres_aqu.count()[0]:.3f}
''')
df_agrupado_medalha_aqu = df[['ouro', 'prata', 'bronze']].groupby(by=df['esporte'] == 'aquatics').sum()
df_agrupado_medalha_aqu = df_agrupado_medalha_aqu.drop(df_agrupado_medalha_aqu[df_agrupado_medalha_aqu.ouro > 200].index)
df_agrupado_medalha_aqu
homens_aqu['altura'].mean()
homens_aqu['peso'].mean()
print(f'''HOMENS:
Altura média: {homens_atl['altura'].mean():.2f}
Peso médio:   {homens_atl['peso'].mean():.2f}

MULHERES:
Altura média: {mulheres_atl['altura'].mean():.2f}
Peso médio:   {mulheres_atl['peso'].mean():.2f}''')
df_aquat = df[(df['esporte']=='aquatics')]
aquat_nacionalidade = df[(df['esporte']=='aquatics')]['nacionalidade'].value_counts()[:10]
aquat_nacionalidade
fig = plt.figure(figsize=(15,5))
sns.countplot(x='nacionalidade',palette=sns.light_palette("green",13,reverse=True), order = aquat_nacionalidade.index, data= df_aquat)
sns.despine(left=True)
plt.xlabel('')
plt.title('Países com mais competidores no esporte aquático',fontsize=21)
plt.ylabel('')
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.xlim(-0.5, 9.5);
sns.set_style('white')
color_green = '#81F79F'
fig =plt.figure(figsize=(15,5))
sns.boxplot(x='nacionalidade', y='altura', order = aquat_nacionalidade.index,color = color_green, fliersize= False, data=df_aquat)
plt.xlim(-0.5,9.5)
sns.despine(left=True)
plt.xlabel('Nacionalidade',fontsize=15)
plt.title('Distribuição das alturas dos atletas no esporte aquático',fontsize=21)
plt.ylabel('Altura em metros',fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.ylim(1.3,2.3);
sns.set_style('white')
color_green = '#81F79F'
fig =plt.figure(figsize=(15,5))
sns.boxplot(x='nacionalidade', y='altura',hue='genero', order = aquat_nacionalidade.index,palette=paleta_genero, fliersize= False, data=df_aquat)
plt.xlim(-0.5,9.5)
sns.despine(left=True)
plt.xlabel('Nacionalidade',fontsize=15)
plt.title('Distribuição das alturas dos atletas no esporte aquático (por gênero)',fontsize=21)
plt.ylabel('Altura em metros',fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.ylim(1.3,2.3);
sns.set_style('white')
color_green = '#81F79F'
fig =plt.figure(figsize=(15,5))
sns.boxplot(x='nacionalidade', y='peso', order = aquat_nacionalidade.index,color = color_green, fliersize= False, data=df_aquat)
plt.xlim(-0.5,9.5)
sns.despine(left=True)
plt.xlabel('Nacionalidade',fontsize=15)
plt.title('Distribuição dos pesos dos atletas no esporte aquático',fontsize=21)
plt.ylabel('Peso em Kg',fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.ylim(0,120);
sns.set_style('white')
color_green = '#81F79F'
fig =plt.figure(figsize=(15,5))
sns.boxplot(x='nacionalidade', y='peso',hue='genero', order = aquat_nacionalidade.index,palette=paleta_genero, fliersize= False, data=df_aquat)
plt.xlim(-0.5,9.5)
sns.despine(left=True)
plt.xlabel('Nacionalidade',fontsize=15)
plt.title('Distribuição dos pesos dos atletas no esporte aquático (por gênero)',fontsize=21)
plt.ylabel('Altura em metros',fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.ylim(0,130);
homens_fut = df[(df['esporte'] == 'football') & (df['genero'] == 'male')]
mulheres_fut = df[(df['esporte'] == 'football') & (df['genero'] == 'female')]
homens_fut_porc = homens_fut.count()[0] / (mulheres_fut.count()[0] + homens_fut.count()[0])*100
mulheres_fut_porc = mulheres_fut.count()[0] / (mulheres_fut.count()[0] + homens_fut.count()[0])*100
print(f'''Homens no futebol: {homens_fut.count()[0]} ({homens_fut_porc:.2f}%)
Mulheres no futebol: {mulheres_fut.count()[0]} ({mulheres_fut_porc:.2f}%)''')
homens_fut = df[(df['esporte'] == 'football') & (df['genero'] == 'male')]
mulheres_fut = df[(df['esporte'] == 'football') & (df['genero'] == 'female')]
masc_fut = homens_fut.count()[0]
fem_fut = mulheres_fut.count()[1]
if masc_fut > fem_fut:
  primeiro = 'MASCULINO'
  segundo = 'FEMININO'
else:
  primeiro = 'FEMININO'
  segundo = 'MASCULINO'
gen_fut = [masc_fut, fem_fut]
fig = plt.figure(figsize=(7,7))
sns.set_style('ticks')
plt.pie(gen_fut, labels=[primeiro, segundo], colors= paleta_genero, autopct='%1.1f%%',shadow=True)
plt.title('Gênero dos atletas (football)',fontsize=21);
homens_fut['ouro'].sum()
homens_fut['prata'].sum()
homens_fut['bronze'].sum()
mulheres_fut['ouro'].sum()
mulheres_fut['prata'].sum()
mulheres_fut['bronze'].sum()
print(f'''HOMENS:
Ouro:    {homens_fut['ouro'].sum()}
Prata:   {homens_fut['prata'].sum()}
Bronze:  {homens_fut['bronze'].sum()}
Total:   {homens_fut['ouro'].sum() + homens_fut['prata'].sum() + homens_fut['bronze'].sum()}
Média por atleta: {(homens_fut['ouro'].sum() + homens_fut['prata'].sum() + homens_fut['bronze'].sum()) / homens_fut.count()[0]:.3f}

-----------
MULHERES:
Ouro:    {mulheres_fut['ouro'].sum()}
Prata:   {mulheres_fut['prata'].sum()}
Bronze:  {mulheres_fut['bronze'].sum()}
Total:   {mulheres_fut['ouro'].sum() + mulheres_fut['prata'].sum() + mulheres_fut['bronze'].sum()}
Média por atleta: {(mulheres_fut['ouro'].sum() + mulheres_fut['prata'].sum() + mulheres_fut['bronze'].sum()) / mulheres_fut.count()[0]:.3f}
''')
df_agrupado_medalha_fut = df[['ouro', 'prata', 'bronze']].groupby(by=df['esporte'] == 'football').sum()
df_agrupado_medalha_fut = df_agrupado_medalha_fut.drop(df_agrupado_medalha_fut[df_agrupado_medalha_fut.ouro > 100].index)
df_agrupado_medalha_fut
homens_fut['altura'].mean()
homens_fut['peso'].mean()
print(f'''HOMENS:
Altura média: {homens_atl['altura'].mean():.2f}
Peso médio:   {homens_atl['peso'].mean():.2f}

MULHERES:
Altura média: {mulheres_atl['altura'].mean():.2f}
Peso médio:   {mulheres_atl['peso'].mean():.2f}''')
df_fut = df[df['esporte']=='football']
fut_nacionalidade = df_fut['nacionalidade'].value_counts()[:10]
fut_nacionalidade.index
df_fut['nacionalidade']
fig = plt.figure(figsize=(15,5))
sns.countplot(x='nacionalidade',palette=sns.light_palette("orange",13,reverse=True), order = fut_nacionalidade.index, data= df_fut)
sns.despine(left=True)
plt.xlabel('')
plt.title('Países com mais competidores no futebol',fontsize=21)
plt.ylabel('')
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.xlim(-0.5, 9.5);
sns.set_style('white')
color_green = '#81F79F'
fig =plt.figure(figsize=(15,5))
sns.boxplot(x='nacionalidade', y='altura',linewidth= 2, order = fut_nacionalidade.index,color = color_yellow, fliersize= False, data=df_fut)
plt.xlim(-0.5,9.5)
sns.despine(left=True)
plt.xlabel('Nacionalidades',fontsize=15)
plt.title('Distribuição das alturas dos atletas no futebol',fontsize=21)
plt.ylabel('Altura em metros',fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.ylim(1.3,2.2);
sns.set_style('white')
color_green = '#81F79F'
fig =plt.figure(figsize=(15,5))
sns.boxplot(x='nacionalidade', y='altura',linewidth= 2,hue='genero', order = fut_nacionalidade.index,palette=paleta_genero, fliersize= False, data=df_fut)
plt.xlim(-0.5,9.5)
sns.despine(left=True)
plt.xlabel('Nacionalidades',fontsize=15)
plt.title('Distribuição das alturas dos atletas no futebol (por gênero)',fontsize=21)
plt.ylabel('Altura em metros',fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.ylim(1.5,2);
sns.set_style('white')
color_green = '#81F79F'
fig =plt.figure(figsize=(15,5))
sns.boxplot(x='nacionalidade', y='peso', order = fut_nacionalidade.index,color = color_yellow, fliersize= False, data=df_fut)
plt.xlim(-0.5,9.5)
sns.despine(left=True)
plt.xlabel('Nacionalidade',fontsize=15)
plt.title('Distribuição dos pesos dos atletas no futebol',fontsize=21)
plt.ylabel('Peso em Kg',fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.ylim(45,110);
sns.set_style('white')
color_green = '#81F79F'
fig =plt.figure(figsize=(15,5))
sns.boxplot(x='nacionalidade', y='peso',hue='genero', order = fut_nacionalidade.index,palette=paleta_genero, fliersize= False, data=df_fut)
plt.xlim(-0.5,9.5)
sns.despine(left=True)
plt.xlabel('Nacionalidades',fontsize=15)
plt.title('Distribuição dos pesos dos atletas no futebol (por gênero)',fontsize=21)
plt.ylabel('Peso em Kg',fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.ylim(40,120);
homens_rem = df[(df['esporte'] == 'rowing') & (df['genero'] == 'male')]
mulheres_rem = df[(df['esporte'] == 'rowing') & (df['genero'] == 'female')]
homens_rem_porc = homens_rem.count()[0] / (mulheres_rem.count()[0] + homens_rem.count()[0])*100
mulheres_rem_porc = mulheres_rem.count()[0] / (mulheres_rem.count()[0] + homens_rem.count()[0])*100
print(f'''Homens no remo: {homens_rem.count()[0]} ({homens_rem_porc:.2f}%)
Mulheres no remo: {mulheres_rem.count()[0]} ({mulheres_rem_porc:.2f}%)''')
homens_rem.count()[0]
homens_rem = df[(df['esporte'] == 'rowing') & (df['genero'] == 'male')]
mulheres_rem = df[(df['esporte'] == 'rowing') & (df['genero'] == 'female')]
masc_rem = homens_rem.count()[0]
fem_rem = mulheres_rem.count()[1]
if masc_rem > fem_rem:
  primeiro = 'MASCULINO'
  segundo = 'FEMININO'
else:
  primeiro = 'FEMININO'
  segundo = 'MASCULINO'
gen_rem = [masc_rem, fem_rem]
fig = plt.figure(figsize=(7,7))
sns.set_style('ticks')
plt.pie(gen_rem, labels=[primeiro, segundo], colors= paleta_genero, autopct='%1.1f%%',shadow=True)
plt.title('Gênero dos atletas (remo)',fontsize=21);
homens_rem['ouro'].sum()
homens_rem['prata'].sum()
homens_rem['bronze'].sum()
mulheres_rem['ouro'].sum()
mulheres_rem['prata'].sum()
mulheres_rem['bronze'].sum()
print(f'''HOMENS:
Ouro:    {homens_rem['ouro'].sum()}
Prata:   {homens_rem['prata'].sum()}
Bronze:  {homens_rem['bronze'].sum()}
Total:   {homens_rem['ouro'].sum() + homens_rem['prata'].sum() + homens_rem['bronze'].sum()}
Média por atleta: {(homens_rem['ouro'].sum() + homens_rem['prata'].sum() + homens_rem['bronze'].sum()) / homens_rem.count()[0]:.3f}

-----------
MULHERES:
Ouro:    {mulheres_rem['ouro'].sum()}
Prata:   {mulheres_rem['prata'].sum()}
Bronze:  {mulheres_rem['bronze'].sum()}
Total:   {mulheres_rem['ouro'].sum() + mulheres_rem['prata'].sum() + mulheres_rem['bronze'].sum()}
Média por atleta: {(mulheres_rem['ouro'].sum() + mulheres_rem['prata'].sum() + mulheres_rem['bronze'].sum()) / mulheres_rem.count()[0]:.3f}
''')
df_agrupado_medalha_rem = df[['ouro', 'prata', 'bronze']].groupby(by=df['esporte'] == 'rowing').sum()
df_agrupado_medalha_rem = df_agrupado_medalha_rem.drop(df_agrupado_medalha_rem[df_agrupado_medalha_rem.ouro > 100].index)
df_agrupado_medalha_rem
homens_rem['altura'].mean()
homens_rem['peso'].mean()
print(f'''HOMENS:
Altura média: {homens_atl['altura'].mean():.2f}
Peso médio:   {homens_atl['peso'].mean():.2f}

MULHERES:
Altura média: {mulheres_atl['altura'].mean():.2f}
Peso médio:   {mulheres_atl['peso'].mean():.2f}''')
df_remo = df[(df['esporte']=='rowing')]
remo_nacionalidade = df_remo['nacionalidade'].value_counts()[:10]
remo_nacionalidade.index
df.head()
altura_nacionalidade = df[(df['esporte'] == 'rowing')]
altura_nacionalidade.nlargest(10,'ouro')
altura_nacionalidade = df.nlargest(10,'ouro')
altura_remo_pais = altura_nacionalidade.groupby(by=df['nacionalidade'])
altura_remo_pais.tail()
#AGORA FALTA O ESPORTE
fig = plt.figure(figsize=(15,5))
sns.countplot(x='nacionalidade',palette=sns.light_palette("red",13,reverse=True), order = remo_nacionalidade.index, data= df_remo)
sns.despine(left=True)
plt.xlabel('')
plt.title('Países com mais competidores no remo',fontsize=21)
plt.ylabel('')
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.xlim(-0.5, 9.5);
sns.set_style('white')
color_green = '#81F79F'
fig =plt.figure(figsize=(15,5))
sns.boxplot(x='nacionalidade', y='altura',linewidth= 2, order = remo_nacionalidade.index,color = color_red, fliersize= False, data=df_remo)
plt.xlim(-0.5,9.5)
sns.despine(left=True)
plt.xlabel('Nacionalidade',fontsize=15)
plt.title('Distribuição das alturas dos atletas no remo',fontsize=21)
plt.ylabel('Altura em metros',fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.ylim(1.3,2.25);
sns.set_style('white')
color_green = '#81F79F'
fig =plt.figure(figsize=(15,5))
sns.boxplot(x='nacionalidade', y='altura',hue= 'genero',linewidth=2, order = remo_nacionalidade.index,palette=paleta_genero, fliersize= False, data=df_remo)
plt.xlim(-0.5,9.5)
sns.despine(left=True)
plt.xlabel('Nacionalidade',fontsize=15)
plt.title('Distribuição das alturas dos atletas no esporte aquático (por gênero)',fontsize=21)
plt.ylabel('Altura em metros',fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.ylim(1.3,2.2);
sns.set_style('white')
color_green = '#81F79F'
fig =plt.figure(figsize=(15,5))
sns.boxplot(x='nacionalidade', y='peso', order = remo_nacionalidade.index,color = color_red, fliersize= False, data=df_remo)
plt.xlim(-0.5,9.5)
sns.despine(left=True)
plt.xlabel('Nacionalidade',fontsize=15)
plt.title('Distribuição dos pesos dos atletas no esporte aquático',fontsize=21)
plt.ylabel('Peso em Kg',fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.ylim(0,130);
sns.set_style('white')
color_green = '#81F79F'
fig =plt.figure(figsize=(15,5))
sns.boxplot(x='nacionalidade', y='peso',hue= 'genero', order = remo_nacionalidade.index,palette=paleta_genero, fliersize= False, data=df_remo)
plt.xlim(-0.5,9.5)
sns.despine(left=True)
plt.xlabel('Nacionalidade',fontsize=15)
plt.title('Distribuição dos pesos dos atletas no esporte aquático (por gênero)',fontsize=21)
plt.ylabel('Peso em Kg',fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.ylim(0,120);
homens_cic = df[(df['esporte'] == 'cycling') & (df['genero'] == 'male')]
mulheres_cic = df[(df['esporte'] == 'cycling') & (df['genero'] == 'female')]
homens_cic_porc = homens_cic.count()[0] / (mulheres_cic.count()[0] + homens_cic.count()[0])*100
mulheres_cic_porc = mulheres_cic.count()[0] / (mulheres_cic.count()[0] + homens_cic.count()[0])*100
print(f'''Homens no ciclismo: {homens_cic.count()[0]} ({homens_cic_porc:.2f}%)
Mulheres no ciclismo: {mulheres_cic.count()[0]} ({mulheres_cic_porc:.2f}%)''')
homens_cic = df[(df['esporte'] == 'cycling') & (df['genero'] == 'male')]
mulheres_cic = df[(df['esporte'] == 'cycling') & (df['genero'] == 'female')]
masc_cic = homens_cic.count()[0]
fem_cic = mulheres_cic.count()[1]
if masc_cic > fem_cic:
  primeiro = 'MASCULINO'
  segundo = 'FEMININO'
else:
  primeiro = 'FEMININO'
  segundo = 'MASCULINO'
gen_cic = [masc_cic, fem_cic]
fig = plt.figure(figsize=(7,7))
sns.set_style('ticks')
plt.pie(gen_cic, labels=[primeiro, segundo], colors= paleta_genero, autopct='%1.1f%%',shadow=True)
plt.title('Gênero dos atletas (ciclismo)',fontsize=21);
homens_cic['ouro'].sum()
homens_cic['prata'].sum()
homens_cic['bronze'].sum()
mulheres_cic['ouro'].sum()
mulheres_cic['prata'].sum()
mulheres_cic['bronze'].sum()
print(f'''HOMENS:
Ouro:    {homens_cic['ouro'].sum()}
Prata:   {homens_cic['prata'].sum()}
Bronze:  {homens_cic['bronze'].sum()}
Total:   {homens_cic['ouro'].sum() + homens_cic['prata'].sum() + homens_cic['bronze'].sum()}
Média por atleta: {(homens_cic['ouro'].sum() + homens_cic['prata'].sum() + homens_cic['bronze'].sum()) / homens_cic.count()[0]:.3f}

-----------
MULHERES:
Ouro:    {mulheres_cic['ouro'].sum()}
Prata:   {mulheres_cic['prata'].sum()}
Bronze:  {mulheres_cic['bronze'].sum()}
Total:   {mulheres_cic['ouro'].sum() + mulheres_cic['prata'].sum() + mulheres_cic['bronze'].sum()}
Média por atleta: {(mulheres_cic['ouro'].sum() + mulheres_cic['prata'].sum() + mulheres_cic['bronze'].sum()) / mulheres_cic.count()[0]:.3f}
''')
df_agrupado_medalha_cic = df[['ouro', 'prata', 'bronze']].groupby(by=df['esporte'] == 'cycling').sum()
df_agrupado_medalha_cic = df_agrupado_medalha_cic.drop(df_agrupado_medalha_cic[df_agrupado_medalha_cic.ouro > 100].index)
df_agrupado_medalha_cic
homens_cic['altura'].mean()
homens_cic['peso'].mean()
print(f'''HOMENS:
Altura média: {homens_atl['altura'].mean():.2f}
Peso médio:   {homens_atl['peso'].mean():.2f}

MULHERES:
Altura média: {mulheres_atl['altura'].mean():.2f}
Peso médio:   {mulheres_atl['peso'].mean():.2f}''')
df_cic = df[(df['esporte']=='cycling')]
ciclismo_nacionalidade = df_cic['nacionalidade'].value_counts()[:10]
ciclismo_nacionalidade.index
fig = plt.figure(figsize=(15,5))
sns.countplot(x='nacionalidade',palette=sns.light_palette("yellow",13,reverse=True), order = ciclismo_nacionalidade.index, data= df_cic)
sns.despine(left=True)
plt.xlabel('')
plt.title('Países com mais competidores no ciclismo',fontsize=21)
plt.ylabel('')
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.xlim(-0.5, 9.5);
sns.set_style('white')
color_green = '#81F79F'
fig =plt.figure(figsize=(15,5))
sns.boxplot(x='nacionalidade', y='altura', order = ciclismo_nacionalidade.index,color = color_yellow, fliersize= False, data=df_cic)
plt.xlim(-0.5,9.5)
sns.despine(left=True)
plt.xlabel('Nacionalidade',fontsize=15)
plt.title('Distribuição das alturas dos atletas no ciclismo',fontsize=21)
plt.ylabel('Altura em metros',fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.ylim(1.3,2.1);
sns.set_style('white')
color_green = '#81F79F'
fig =plt.figure(figsize=(15,5))
sns.boxplot(x='nacionalidade', y='altura',hue='genero', order = ciclismo_nacionalidade.index,palette=paleta_genero, fliersize= False, data=df_cic)
plt.xlim(-0.5,9.5)
sns.despine(left=True)
plt.xlabel('Nacionalidade',fontsize=15)
plt.title('Distribuição das alturas dos atletas no ciclismo',fontsize=21)
plt.ylabel('Altura em metros',fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.ylim(1.3,2.2);
sns.set_style('white')
color_green = '#81F79F'
fig =plt.figure(figsize=(15,5))
sns.boxplot(x='nacionalidade', y='peso', order = ciclismo_nacionalidade.index,color = color_yellow, fliersize= False, data=df)
plt.xlim(-0.5,9.5)
sns.despine(left=True)
plt.xlabel('Nacionalidade',fontsize=15)
plt.title('Distribuição dos pesos dos atletas no ciclismo',fontsize=21)
plt.ylabel('Quilo em kg',fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.ylim(30,140);
sns.set_style('white')
color_green = '#81F79F'
fig =plt.figure(figsize=(15,5))
sns.boxplot(x='nacionalidade', y='peso',hue='genero', order = ciclismo_nacionalidade.index,palette=paleta_genero, fliersize= False, data=df_cic)
plt.xlim(-0.5,9.5)
sns.despine(left=True)
plt.xlabel('Nacionalidades',fontsize=15)
plt.title('Distribuição dos pesos dos atletas no esporte aquático (por gênero)',fontsize=21)
plt.ylabel('Peso em Kg',fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.ylim(30,110);
sns.set_style('white')
color_green = '#81F79F'
fig =plt.figure(figsize=(10,5))
sns.boxplot(x='esporte', y='altura', order = df['esporte'].value_counts()[:5].index,palette= paleta_esporte, fliersize= False, data=df)
plt.xlim(-0.5,4.5)
sns.despine(left=True)
plt.xlabel('Esportes',fontsize=15)
plt.ylabel('Altura em metros',fontsize=15)
plt.title('Distribuição das alturas dos atletas',fontsize=21)
plt.xticks(fontsize=12)
plt.yticks(fontsize=11)
plt.ylim(1.3,2.2);
sns.set_style('white')
color_green = '#81F79F'
fig =plt.figure(figsize=(15,5))
sns.boxplot(x='esporte', y='altura',hue='genero', order = df['esporte'].value_counts()[:5].index,palette= paleta_genero, fliersize= False, data=df)
plt.xlim(-0.5,4.5)
sns.despine(left=True)
plt.xlabel('Nacionalidades',fontsize=15)
plt.ylabel('Altura em metros',fontsize=15)
plt.title('Distribuição das alturas os atletas (separados por gênero)',fontsize=21)
plt.xticks(fontsize=12)
plt.yticks(fontsize=11)
plt.ylim(1.44,2.2);
sns.set_style('white')
color_green = '#81F79F'
fig =plt.figure(figsize=(10,5))
sns.boxplot(x='esporte', y='peso', order = df['esporte'].value_counts()[:5].index,palette= paleta_esporte, fliersize= False, data=df)
plt.xlim(-0.5,4.5)
sns.despine(left=True)
plt.xlabel('')
plt.title('Distribuição dos pesos dos atletas de cada esporte',fontsize=21)
plt.ylabel('Peso em Kg',fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.ylim(0,130);
sns.set_style('white')
color_green = '#81F79F'
fig =plt.figure(figsize=(15,5))
sns.boxplot(x='esporte', y='peso',hue='genero', order = df['esporte'].value_counts()[:5].index,palette= paleta_genero, fliersize= False, data=df)
plt.xlim(-0.5,4.5)
sns.despine(left=True)
plt.xlabel('')
plt.title('Distribuição dos pesos dos atletas de cada esporte (por gênero)',fontsize=21)
plt.ylabel('Peso em Kg',fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.ylim(30,130);