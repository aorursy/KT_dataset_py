import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
anp = pd.read_excel('../input/tabela_de_pocos_janeiro_2018.xlsx')
anp['POCO_TIPO'] = anp['POCO'].astype(str).str[0]
anp.info()
anp['BACIA'].value_counts()
anp['OPERADOR'].value_counts()
anp.groupby('TERRA_MAR').describe()
pd.Series(anp.PROFUNDIDADE_SONDADOR_M ).sum() 
pd.Series(anp.PROFUNDIDADE_SONDADOR_M ).groupby(anp['BACIA']).sum().nlargest(5)
pd.Series(anp.PROFUNDIDADE_SONDADOR_M ).groupby(anp['OPERADOR']).sum().nlargest(20)
anp_mar = anp[anp.TERRA_MAR=='M']  
anp_mar['LAMINA_D_AGUA_M'].describe()
anp_mar['POCO_TIPO'].value_counts()
pd.Series(anp_mar.PROFUNDIDADE_SONDADOR_M ).groupby(anp_mar['BACIA']).sum().nlargest(5)
pd.Series(anp_mar.PROFUNDIDADE_SONDADOR_M ).sum() 
anp_terra = anp[anp.TERRA_MAR=='T'] 
anp_terra.describe()
pd.Series(anp_terra.PROFUNDIDADE_SONDADOR_M ).sum() 
pd.Series(anp_terra.PROFUNDIDADE_SONDADOR_M ).groupby(anp_terra['BACIA']).sum().nlargest(5)
petrobras = anp[anp.OPERADOR=='Petrobras'] 
petrobras.describe()
petrobras.groupby('TERRA_MAR').describe()
quebra_monopolio = pd.to_datetime('06/08/1997')
anp_mod_mar = anp_mar.loc[anp_mar.INICIO >= quebra_monopolio, :]
anp_mod_terra = anp_terra.loc[anp_terra.INICIO >= quebra_monopolio, :]
op_terra = anp_mod_terra.OPERADOR.value_counts()
op_terra
op_mar = anp_mod_mar.OPERADOR.value_counts()
op_mar
fig1, ax = plt.subplots()

def autopct_more_than_1(pct):
    return ('%1.f%%' % pct) if pct > 1 else ''

p,t,a = ax.pie(op_mar.values, autopct=autopct_more_than_1)
ax.axis('equal') 
normsizes = op_mar/op_mar.sum()*100
h,l = zip(*[(h,lab) for h,lab,i in zip(p,op_mar.index.values,normsizes.values) if i > 1])

ax.legend(h, l,loc="best", bbox_to_anchor=(1,1))

plt.show()
fig1, ax = plt.subplots()

def autopct_more_than_1(pct):
    return ('%1.f%%' % pct) if pct > 1 else ''

p,t,a = ax.pie(op_terra.values, autopct=autopct_more_than_1)
ax.axis('equal') 
normsizes = op_terra/op_terra.sum()*100
h,l = zip(*[(h,lab) for h,lab,i in zip(p,op_terra.index.values,normsizes.values) if i > 1])

ax.legend(h, l,loc="best", bbox_to_anchor=(1,1))

plt.show()
petrobras_mod = petrobras.loc[petrobras.INICIO >= quebra_monopolio, :]
petrobras_mod.CATEGORIA.value_counts()
anp_exp_mod_mar = anp_mod_mar[anp_mod_mar.TIPO=='Explotatório']
anp_exp_mod_mar.info()
anp_exp_mod_mar.RECLASSIFICACAO.value_counts()
mar_suc = anp_exp_mod_mar[anp_exp_mod_mar.RECLASSIFICACAO=='PRODUTOR COMERCIAL DE PETRÓLEO']
pd.Series(anp_exp_mod_mar.OPERADOR).groupby(anp_exp_mod_mar['RECLASSIFICACAO']).value_counts().nlargest(50)
anp_exp_mod_mar.RECLASSIFICACAO.value_counts()
anp_exp_mod_mar.OPERADOR.value_counts().nlargest(20)
anp_mod_mar.NOM_SONDA.value_counts()
petrobras_mod_mar = petrobras_mod[petrobras_mod.TERRA_MAR=='M']
petrobras_mod_mar['POCO_TIPO'].value_counts().plot.pie(subplots=True)
petrobras_mod_terra = petrobras_mod[petrobras_mod.TERRA_MAR=='T']
pd.Series(petrobras_mod_mar.PROFUNDIDADE_SONDADOR_M ).sum() 
mar_suc.info()
anp_exp_mod_mar.SITUACAO.describe()
petrobras_time = petrobras.INICIO.dt.year.value_counts().sort_index()
petrobras_metro_ano = pd.Series(petrobras.PROFUNDIDADE_SONDADOR_M.dropna()).groupby(petrobras.INICIO.dt.year).sum()
petrobras_metro_ano.plot()
anp_time = anp.INICIO.dt.year.value_counts().sort_index()
anp_time.plot()
anp_metro_ano = pd.Series(anp.PROFUNDIDADE_SONDADOR_M.dropna()).groupby(anp.INICIO.dt.year).sum()
anp_metro_ano.plot()
anppio = anp[anp['POCO_TIPO'] == '1'] 
anppio_metro_ano = pd.Series(anppio.PROFUNDIDADE_SONDADOR_M.dropna()).groupby(anp.INICIO.dt.year).sum()
anppio_time = anppio.INICIO.dt.year.value_counts().sort_index()
anppio_time.plot()
anpdes = anp[anp['POCO_TIPO'] == '7'] 
anpdes_metro_ano = pd.Series(anpdes.PROFUNDIDADE_SONDADOR_M.dropna()).groupby(anp.INICIO.dt.year).sum()
anpdes_time = anpdes.INICIO.dt.year.value_counts().sort_index()
anpdes_metro_ano.plot()

