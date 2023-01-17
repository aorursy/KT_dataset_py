import pandas as pd
classificacao = [['aeronave_operador_categoria', 'Qualitativa Nominal'],
    ['aeronave_tipo_veiculo', 'Qualitativa Nominal'],
    ['aeronave_fabricante', 'Qualitativa Nominal'],
    ['aeronave_modelo', 'Qualitativa Nominal'],
    ['aeronave_motor_tipo', 'Qualitativa Nominal'],
    ['aeronave_motor_quantidade', 'Quantitativa Discreta'],
    ['aeronave_pmd', 'Quantitativa Continua'],
    ['aeronave_ano_fabricacao', 'Quantitativa Discreta'],#discretização
    ['aeronave_fase_operacao', 'Qualitativa Ordinal'],
    ['aeronave_tipo_operacao', 'Qualitativa Nominal'],
    ['aeronave_nivel_dano', 'Qualitativa Ordinal']]
classificacao = pd.DataFrame(classificacao, columns=["Variavel", "Classificação"])
classificacao
df = pd.read_csv('../input/anv.csv')
dfaux = classificacao.where(classificacao['Classificação'].apply(lambda x : 'Qualitativa' in x)).dropna()
for var in dfaux['Variavel']:
    dfvar = pd.DataFrame(df[var].value_counts())
    dfvar.rename(columns={ var: 'Frequencia Absoluta'}, inplace=True)
    dfvar['Frequencia Relativa'] = dfvar['Frequencia Absoluta']/dfvar['Frequencia Absoluta'].sum()
    dfvar['Frequencia Relativa %'] = dfvar['Frequencia Relativa'].apply((lambda x : '{:.2%}'.format(x)))
    a = pd.DataFrame([[dfvar['Frequencia Absoluta'].sum(),dfvar['Frequencia Relativa'].sum(),
                       '{:.2%}'.format(dfvar['Frequencia Relativa'].sum())]], 
                     columns=['Frequencia Absoluta', 'Frequencia Relativa', 'Frequencia Relativa %'], index=['Total'])
    print(dfvar.append(a)) #Print fica feio no kaggle
dfvar = pd.DataFrame(df['aeronave_operador_categoria'].value_counts())
dfvar.rename(columns={ 'aeronave_operador_categoria': 'Frequencia Absoluta'}, inplace=True)
dfvar['Frequencia Relativa'] = dfvar['Frequencia Absoluta']/dfvar['Frequencia Absoluta'].sum()
dfvar['Frequencia Relativa %'] = dfvar['Frequencia Relativa'].apply((lambda x : '{:.2%}'.format(x)))
a = pd.DataFrame([[dfvar['Frequencia Absoluta'].sum(),dfvar['Frequencia Relativa'].sum(),
                   '{:.2%}'.format(dfvar['Frequencia Relativa'].sum())]], 
                 columns=['Frequencia Absoluta', 'Frequencia Relativa', 'Frequencia Relativa %'], index=['Total'])
dfvar.append(a)
dfvar = pd.DataFrame(df['aeronave_tipo_veiculo'].value_counts())
dfvar.rename(columns={ 'aeronave_tipo_veiculo': 'Frequencia Absoluta'}, inplace=True)
dfvar['Frequencia Relativa'] = dfvar['Frequencia Absoluta']/dfvar['Frequencia Absoluta'].sum()
dfvar['Frequencia Relativa %'] = dfvar['Frequencia Relativa'].apply((lambda x : '{:.2%}'.format(x)))
a = pd.DataFrame([[dfvar['Frequencia Absoluta'].sum(),dfvar['Frequencia Relativa'].sum(),
                   '{:.2%}'.format(dfvar['Frequencia Relativa'].sum())]], 
                 columns=['Frequencia Absoluta', 'Frequencia Relativa', 'Frequencia Relativa %'], index=['Total'])
dfvar.append(a)
dfvar = pd.DataFrame(df['aeronave_fabricante'].value_counts())
dfvar.rename(columns={ 'aeronave_fabricante': 'Frequencia Absoluta'}, inplace=True)
dfvar['Frequencia Relativa'] = dfvar['Frequencia Absoluta']/dfvar['Frequencia Absoluta'].sum()
dfvar['Frequencia Relativa %'] = dfvar['Frequencia Relativa'].apply((lambda x : '{:.2%}'.format(x)))
a = pd.DataFrame([[dfvar['Frequencia Absoluta'].sum(),dfvar['Frequencia Relativa'].sum(),
                   '{:.2%}'.format(dfvar['Frequencia Relativa'].sum())]], 
                 columns=['Frequencia Absoluta', 'Frequencia Relativa', 'Frequencia Relativa %'], index=['Total'])
dfvar.append(a)
dfvar = pd.DataFrame(df['aeronave_modelo'].value_counts())
dfvar.rename(columns={ 'aeronave_modelo': 'Frequencia Absoluta'}, inplace=True)
dfvar['Frequencia Relativa'] = dfvar['Frequencia Absoluta']/dfvar['Frequencia Absoluta'].sum()
dfvar['Frequencia Relativa %'] = dfvar['Frequencia Relativa'].apply((lambda x : '{:.2%}'.format(x)))
a = pd.DataFrame([[dfvar['Frequencia Absoluta'].sum(),dfvar['Frequencia Relativa'].sum(),
                   '{:.2%}'.format(dfvar['Frequencia Relativa'].sum())]], 
                 columns=['Frequencia Absoluta', 'Frequencia Relativa', 'Frequencia Relativa %'], index=['Total'])
dfvar.append(a)
dfvar = pd.DataFrame(df['aeronave_motor_tipo'].value_counts())
dfvar.rename(columns={ 'aeronave_motor_tipo': 'Frequencia Absoluta'}, inplace=True)
dfvar['Frequencia Relativa'] = dfvar['Frequencia Absoluta']/dfvar['Frequencia Absoluta'].sum()
dfvar['Frequencia Relativa %'] = dfvar['Frequencia Relativa'].apply((lambda x : '{:.2%}'.format(x)))
a = pd.DataFrame([[dfvar['Frequencia Absoluta'].sum(),dfvar['Frequencia Relativa'].sum(),
                   '{:.2%}'.format(dfvar['Frequencia Relativa'].sum())]], 
                 columns=['Frequencia Absoluta', 'Frequencia Relativa', 'Frequencia Relativa %'], index=['Total'])
dfvar.append(a)
dfvar = pd.DataFrame(df['aeronave_fase_operacao'].value_counts())
dfvar.rename(columns={ 'aeronave_fase_operacao': 'Frequencia Absoluta'}, inplace=True)
dfvar['Frequencia Relativa'] = dfvar['Frequencia Absoluta']/dfvar['Frequencia Absoluta'].sum()
dfvar['Frequencia Relativa %'] = dfvar['Frequencia Relativa'].apply((lambda x : '{:.2%}'.format(x)))
a = pd.DataFrame([[dfvar['Frequencia Absoluta'].sum(),dfvar['Frequencia Relativa'].sum(),
                   '{:.2%}'.format(dfvar['Frequencia Relativa'].sum())]], 
                 columns=['Frequencia Absoluta', 'Frequencia Relativa', 'Frequencia Relativa %'], index=['Total'])
dfvar.append(a)
dfvar = pd.DataFrame(df['aeronave_tipo_operacao'].value_counts())
dfvar.rename(columns={ 'aeronave_tipo_operacao': 'Frequencia Absoluta'}, inplace=True)
dfvar['Frequencia Relativa'] = dfvar['Frequencia Absoluta']/dfvar['Frequencia Absoluta'].sum()
dfvar['Frequencia Relativa %'] = dfvar['Frequencia Relativa'].apply((lambda x : '{:.2%}'.format(x)))
a = pd.DataFrame([[dfvar['Frequencia Absoluta'].sum(),dfvar['Frequencia Relativa'].sum(),
                   '{:.2%}'.format(dfvar['Frequencia Relativa'].sum())]], 
                 columns=['Frequencia Absoluta', 'Frequencia Relativa', 'Frequencia Relativa %'], index=['Total'])
dfvar.append(a)
dfvar = pd.DataFrame(df['aeronave_nivel_dano'].value_counts())
dfvar.rename(columns={ 'aeronave_nivel_dano': 'Frequencia Absoluta'}, inplace=True)
dfvar['Frequencia Relativa'] = dfvar['Frequencia Absoluta']/dfvar['Frequencia Absoluta'].sum()
dfvar['Frequencia Relativa %'] = dfvar['Frequencia Relativa'].apply((lambda x : '{:.2%}'.format(x)))
a = pd.DataFrame([[dfvar['Frequencia Absoluta'].sum(),dfvar['Frequencia Relativa'].sum(),
                   '{:.2%}'.format(dfvar['Frequencia Relativa'].sum())]], 
                 columns=['Frequencia Absoluta', 'Frequencia Relativa', 'Frequencia Relativa %'], index=['Total'])
dfvar.append(a)
df['aeronave_operador_categoria'].value_counts()
df['aeronave_tipo_veiculo'].value_counts()
df['aeronave_fabricante'].value_counts()
df['aeronave_modelo'].value_counts()
df['aeronave_motor_tipo'].value_counts()
df['aeronave_fase_operacao'].value_counts()
df['aeronave_tipo_operacao'].value_counts()
df1 = pd.DataFrame(df['aeronave_nivel_dano'].value_counts())
df1.rename(columns={"aeronave_nivel_dano": "Frequencia Absoluta"}, inplace=True)
df1['Frequencia Relativa'] = df1['Frequencia Absoluta']/df1['Frequencia Absoluta'].sum()
df1['Frequencia Relativa %'] = df1['Frequencia Relativa'].apply((lambda x : '{:.2%}'.format(x)))
df1
a = pd.DataFrame([[df1['Frequencia Absoluta'].sum(),df1['Frequencia Relativa'].sum(),
                   '{:.2%}'.format(df1['Frequencia Relativa'].sum())]], 
                 columns=['Frequencia Absoluta', 'Frequencia Relativa', 'Frequencia Relativa %'], index=['Total'])
df1.append(a)
import matplotlib.pyplot as plt
%matplotlib inline  
vc = df['aeronave_operador_categoria'].value_counts()
plt.bar(vc.index, vc)
plt.xticks(rotation='vertical')
plt.show()
dfaux = df.groupby(['aeronave_nivel_dano', 'aeronave_operador_categoria']).size()
gridsize = (3, 2) # 4 rows, 2 columns
fig = plt.figure(figsize=(10, 15)) # this creates a figure without axes
ax = []
for i in range(len(dfaux.index.levels[0])):
    #import pdb; pdb.set_trace()
    ax.append(plt.subplot2grid(gridsize, (i//2, i%2)))
    x = dfaux[dfaux.index.levels[0][i]].index
    y = dfaux[dfaux.index.levels[0][i]]
    ax[i].bar(x, y.values)
    ax[i].set_title(dfaux.index.levels[0][i])
    plt.xticks(rotation='vertical')
    #plt.xticks([])
#plt.xticks(dfaux.index.levels[1])

gridsize = (3, 2) # 3 rows, 2 columns
fig = plt.figure(figsize=(10, 15)) # this creates a figure without axes
ax = []
for i in range(len(dfaux.index.levels[0])):
    if(i<4):
        plt.xticks([])
    dano = dfaux.index.levels[0][i]
    
    dic = {cat:dfaux[dano][cat] for cat in dfaux[dano].index}
    for cat in dfaux.index.levels[1]:
        if(cat not in dic):
            dic[cat] = 0
    #import pdb; pdb.set_trace()
    sorted_keys = sorted(dic)
    x = []
    y = []
    for key in sorted_keys:
        x += [key]
        y += [dic[key]]
    #print(dano)
    #print(x)
    #print(y)
    ax.append(plt.subplot2grid(gridsize, (i//2, i%2)))
    ax[i].bar(x, y)
    ax[i].set_title(dfaux.index.levels[0][i])
    plt.xticks(rotation='vertical')
vc = df['aeronave_tipo_veiculo'].value_counts()
plt.bar(vc.index, vc)
plt.xticks(rotation='vertical')
plt.show()
dfaux = df.groupby(['aeronave_nivel_dano', 'aeronave_tipo_veiculo']).size()
gridsize = (2, 3) # 3 rows, 2 columns
fig = plt.figure(figsize=(20, 10)) # this creates a figure without axes
ax = []
for i in range(len(dfaux.index.levels[0])):
    if(i<3):
        plt.xticks([])
    dano = dfaux.index.levels[0][i]
    
    dic = {tipo:dfaux[dano][tipo] for tipo in dfaux[dano].index}
    for tipo in dfaux.index.levels[1]:
        if(tipo not in dic):
            dic[tipo] = 0
    #import pdb; pdb.set_trace()
    sorted_keys = sorted(dic)
    x = []
    y = []
    for key in sorted_keys:
        x += [key]
        y += [dic[key]]
    #print(dano)
    #print(x)
    #print(y)
    ax.append(plt.subplot2grid(gridsize, (i//3, i%3)))
    ax[i].bar(x, y)
    ax[i].set_title(dfaux.index.levels[0][i])
    plt.xticks(rotation='vertical')
vc = df['aeronave_fabricante'].value_counts().head(40)
plt.figure(figsize=(20, 10))
plt.bar(vc.index, vc)
plt.xticks(rotation='vertical')
plt.show()
vc = df['aeronave_modelo'].value_counts().head(40)
plt.figure(figsize=(20, 10))
plt.bar(vc.index, vc)
plt.xticks(rotation='vertical')
plt.show()
vc = df['aeronave_motor_tipo'].value_counts()
plt.bar(vc.index, vc)
plt.xticks(rotation='vertical')
plt.show()
dfaux = df.groupby(['aeronave_nivel_dano', 'aeronave_motor_tipo']).size()
gridsize = (2, 3) # 3 rows, 2 columns
fig = plt.figure(figsize=(20, 10)) # this creates a figure without axes
ax = []
for i in range(len(dfaux.index.levels[0])):
    if(i<3):
        plt.xticks([])
    dano = dfaux.index.levels[0][i]
    
    dic = {qtd:dfaux[dano][qtd] for qtd in dfaux[dano].index}
    for qtd in dfaux.index.levels[1]:
        if(qtd not in dic):
            dic[qtd] = 0
    #import pdb; pdb.set_trace()
    sorted_keys = sorted(dic)
    x = []
    y = []
    for key in sorted_keys:
        x += [key]
        y += [dic[key]]
    #print(dano)
    #print(x)
    #print(y)
    ax.append(plt.subplot2grid(gridsize, (i//3, i%3)))
    ax[i].bar(x, y)
    ax[i].set_title(dfaux.index.levels[0][i])
    plt.xticks(rotation='vertical')
vc = df['aeronave_motor_quantidade'].value_counts()
plt.bar(vc.index, vc)
plt.xticks(rotation='vertical')
plt.show()
dfaux = df.groupby(['aeronave_nivel_dano', 'aeronave_motor_quantidade']).size()
gridsize = (2, 3) # 3 rows, 2 columns
fig = plt.figure(figsize=(20, 10)) # this creates a figure without axes
ax = []
for i in range(len(dfaux.index.levels[0])):
    if(i<3):
        plt.xticks([])
    dano = dfaux.index.levels[0][i]
    
    dic = {tipo:dfaux[dano][tipo] for tipo in dfaux[dano].index}
    for tipo in dfaux.index.levels[1]:
        if(tipo not in dic):
            dic[tipo] = 0
    #import pdb; pdb.set_trace()
    sorted_keys = sorted(dic)
    x = []
    y = []
    for key in sorted_keys:
        x += [key]
        y += [dic[key]]
    #print(dano)
    #print(x)
    #print(y)
    ax.append(plt.subplot2grid(gridsize, (i//3, i%3)))
    ax[i].bar(x, y)
    ax[i].set_title(dfaux.index.levels[0][i])
    plt.xticks(rotation='vertical')
import numpy as np
vc = df.groupby(pd.cut(df['aeronave_pmd'], np.arange(0, 100000, 2000), right=False)).size()
plt.plot(np.arange(0, 100000-2000, 2000), vc.values)
plt.xlabel('Peso')
plt.ylabel('Frequencia')
plt.show()
vc = df['aeronave_fase_operacao'].value_counts()
plt.figure(figsize=(20, 7))
plt.bar(vc.index, vc)
plt.xticks(rotation='vertical')
plt.show()
dfaux = df.groupby(['aeronave_nivel_dano', 'aeronave_fase_operacao']).size()
gridsize = (2, 3) # 3 rows, 2 columns
fig = plt.figure(figsize=(25, 10)) # this creates a figure without axes
ax = []
for i in range(len(dfaux.index.levels[0])):
    if(i<3):
        plt.xticks([])
    dano = dfaux.index.levels[0][i]
    
    dic = {fase:dfaux[dano][fase] for fase in dfaux[dano].index}
    for fase in dfaux.index.levels[1]:
        if(fase not in dic):
            dic[fase] = 0
    #import pdb; pdb.set_trace()
    sorted_keys = sorted(dic)
    x = []
    y = []
    for key in sorted_keys:
        x += [key]
        y += [dic[key]]
    #print(dano)
    #print(x)
    #print(y)
    ax.append(plt.subplot2grid(gridsize, (i//3, i%3)))
    ax[i].bar(x, y)
    ax[i].set_title(dfaux.index.levels[0][i])
    plt.xticks(rotation='vertical')
vc = df['aeronave_tipo_operacao'].value_counts()
plt.bar(vc.index, vc)
plt.xticks(rotation='vertical')
plt.show()
dfaux = df.groupby(['aeronave_nivel_dano', 'aeronave_tipo_operacao']).size()
gridsize = (2, 3) # 3 rows, 2 columns
fig = plt.figure(figsize=(25, 10)) # this creates a figure without axes
ax = []
for i in range(len(dfaux.index.levels[0])):
    if(i<3):
        plt.xticks([])
    dano = dfaux.index.levels[0][i]
    
    dic = {tipo:dfaux[dano][tipo] for tipo in dfaux[dano].index}
    for tipo in dfaux.index.levels[1]:
        if(tipo not in dic):
            dic[tipo] = 0
    #import pdb; pdb.set_trace()
    sorted_keys = sorted(dic)
    x = []
    y = []
    for key in sorted_keys:
        x += [key]
        y += [dic[key]]
    #print(dano)
    #print(x)
    #print(y)
    ax.append(plt.subplot2grid(gridsize, (i//3, i%3)))
    ax[i].bar(x, y)
    ax[i].set_title(dfaux.index.levels[0][i])
    plt.xticks(rotation='vertical')
vc = df['aeronave_nivel_dano'].value_counts()
plt.bar(vc.index, vc)
plt.xticks(rotation='vertical')
plt.show()