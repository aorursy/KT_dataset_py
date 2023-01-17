import pandas as pd
df_eleitorado = pd.read_csv('../input/BR_eleitorado_2016_municipio.csv', delimiter=',')
df_eleitorado.head(1)
l_regiao = [
["Norte","AM"],
["Norte","RR"],
["Norte","AP"],    
["Norte","PA"],
["Norte","TO"],
["Norte","RO"],
["Norte","AC"],
["Nordeste","MA"],
["Nordeste","PI"],
["Nordeste","CE"],
["Nordeste","RN"],
["Nordeste","PE"],
["Nordeste","PB"],
["Nordeste","SE"],
["Nordeste","AL"],
["Nordeste","BA"],
["Centro-Oeste","DF"],
["Centro-Oeste","MT"],
["Centro-Oeste","MS"],
["Centro-Oeste","GO"],
["Sudeste","SP"],
["Sudeste","RJ"],
["Sudeste","ES"],
["Sudeste","MG"],
["Sul","PR"],
["Sul","RS"],
["Sul","SC"]]
df_regiao = pd.DataFrame(l_regiao, columns=["Regiao", "uf"])
df_eleitorado = pd.merge(df_eleitorado, df_regiao, on='uf')

#calculando percentual do total de eleitores em relação ao estado em relação d
p = pd.DataFrame( df_eleitorado.groupby(by=['uf'])['total_eleitores'].sum().reset_index().rename(columns={'index':'uf'}) )
p['uf_percentual'] = p['total_eleitores']/ df_eleitorado.sum()['total_eleitores']*100
p.drop('total_eleitores',  axis=1, inplace=True)
df_eleitorado = pd.merge(df_eleitorado, p, on='uf')

#calculando percentual do total de eleitores em relação ao estado em relação d
p = pd.DataFrame( df_eleitorado.groupby(by=['uf'])['gen_feminino'].sum().reset_index().rename(columns={'index':'uf'}) )
p2 = pd.DataFrame( df_eleitorado.groupby(by=['uf'])['gen_masculino'].sum().reset_index().rename(columns={'index':'uf'}) )
p3 = pd.DataFrame( df_eleitorado.groupby(by=['uf'])['gen_nao_informado'].sum().reset_index().rename(columns={'index':'uf'}) )
p = pd.merge(p, p2, on='uf')
p = pd.merge(p, p3, on='uf')
p['percentual_gen_feminino_uf'] = p['gen_feminino']/(p['gen_feminino']+p['gen_masculino']+p['gen_nao_informado'])*100
p['percentual_gen_masculino_uf'] = p['gen_masculino']/(p['gen_feminino']+p['gen_masculino']+p['gen_nao_informado'])*100
p['percentual_gen_nao_informado_uf'] = p['gen_nao_informado']/(p['gen_feminino']+p['gen_masculino']+p['gen_nao_informado'])*100
p.drop(['gen_feminino','gen_masculino','gen_nao_informado'],  axis=1, inplace=True)
df_eleitorado = pd.merge(df_eleitorado,p , on = 'uf')
resposta = [["Regiao", "Categorica Nominal"],
            ["faixa_etaria","Categorica Ordinal"],
            ["total_eleitores","Quantitativa Discreta"],
            ["gen_feminino","Quantitativa Discreta"],
            ["gen_masculino","Quantitativa Discreta"],
            ["gen_nao_informado","Quantitativa Discreta"],
            ["uf_percentual","Quantitativa Continua"],
            ["percentual_gen_feminino_uf","Quantitativa Continua"]
           ] #variáveis relacionadas a tempo são contínuas, mas podem ser discretas pois não há perdas -- (discretização)
resposta = pd.DataFrame(resposta, columns=["Variavel", "Classificação"])
resposta
df_eleitorado.groupby(by='Regiao').count()['cod_municipio_tse']
dr = (df_eleitorado.groupby(by='Regiao').count()['cod_municipio_tse']).reset_index().rename(columns={'index':'Regiao', 'cod_municipio_tse' : 'Frequencia_absoluta'})
dr2 = pd.DataFrame({'Regiao':['Total'],'Frequencia_absoluta':dr['Frequencia_absoluta'].sum()})
df_frequencia = pd.concat([dr, dr2], ignore_index=True,sort=True)

df_frequencia['Frequencia_relativa'] = round((df_frequencia['Frequencia_absoluta']/float(df_frequencia[df_frequencia['Regiao'] == 'Total']['Frequencia_absoluta']))*100,2)
df_frequencia = df_frequencia[['Regiao','Frequencia_absoluta','Frequencia_relativa']]
df_frequencia
#df_eleitorado.head(1)
df_faixas = df_eleitorado.melt( id_vars=['cod_municipio_tse','uf','nome_municipio','total_eleitores','gen_feminino','gen_masculino','gen_nao_informado','Regiao','percentual_gen_feminino_uf','percentual_gen_masculino_uf','percentual_gen_nao_informado_uf'
], var_name='faixa_etaria', value_name='qtd_faixa')
df_faixas.groupby(by='faixa_etaria').sum()['qtd_faixa']
d = (df_faixas.groupby(by='faixa_etaria').sum()['qtd_faixa']).reset_index().rename(columns={'index':'faixa_etaria', 'qtd_faixa' : 'Frequencia_absoluta'})
d

d2 = pd.DataFrame({'faixa_etaria':['Total'],'Frequencia_absoluta':d['Frequencia_absoluta'].sum()})
d3 = pd.concat([d, d2], ignore_index=True,sort=True)

d3['Frequencia_relativa'] = round((d3['Frequencia_absoluta']/float(d3[d3['faixa_etaria'] == 'Total']['Frequencia_absoluta']))*100,2)
d3

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvas
from matplotlib.figure import Figure

# definindo o tamanho da figura
fig = Figure(figsize=(7,7))

# definindo qual canvas utilizar 
canvas = FigureCanvas(fig)
df_frequencia = (df_eleitorado.Regiao.value_counts()).reset_index().rename(columns={'index':'Categoria', 'Regiao' : 'Frequencia_absoluta'})
df_frequencia2 = pd.DataFrame({'Categoria':['Total'],'Frequencia_absoluta':df_frequencia['Frequencia_absoluta'].sum()})
df_frequencia = pd.concat([df_frequencia, df_frequencia2], ignore_index=True)
df_frequencia['Frequencia_relativa'] = (df_frequencia['Frequencia_absoluta']/float(df_frequencia[df_frequencia['Categoria'] == 'Total']['Frequencia_absoluta']))*100


#df_frequencia = (df_eleitorado.Regiao.value_counts()).reset_index().rename(columns={'index':'Categoria', 'Regiao' : 'Quantidade'})
x, y = df_frequencia[df_frequencia['Categoria'] != 'Total']['Categoria'],df_frequencia[df_frequencia['Categoria'] != 'Total']['Frequencia_absoluta']


#print(df_eleitorado.Regiao.value_counts())
total = df_frequencia.groupby(by='Categoria').get_group('Total')['Frequencia_relativa']
sudeste = df_frequencia.groupby(by='Categoria').get_group('Sudeste')['Frequencia_relativa']
norte = df_frequencia.groupby(by='Categoria').get_group('Norte')['Frequencia_relativa']
nordeste = df_frequencia.groupby(by='Categoria').get_group('Nordeste')['Frequencia_relativa']
sul = df_frequencia.groupby(by='Categoria').get_group('Sul')['Frequencia_relativa']
centro_oeste = df_frequencia.groupby(by='Categoria').get_group('Centro-Oeste')['Frequencia_relativa']



gridsize = (3, 2) # 4 rows, 2 columns
fig = plt.figure(figsize=(12, 12)) # this creates a figure without axes
regioes =  plt.subplot2grid(gridsize, (0, 0))
p_sudeste =plt.subplot2grid(gridsize, (0, 1)) # 2nd argument = origin of individual box
p_sul = plt.subplot2grid(gridsize, (1, 0))
p_norte = plt.subplot2grid(gridsize, (1, 1))
p_nordeste = plt.subplot2grid(gridsize, (2, 0))
p_centro_oeste = plt.subplot2grid(gridsize, (2, 1))


regioes.bar(x, y, color = 'b')
regioes.set_title('Diferença entre quantidade de municípios por Região')
#regioes.xticks(rotation='vertical')





rotulos = [ str(round( (float(total) - float(sudeste) ),2)) + '%' , str(round(float(sudeste),2)) + '%' ]
valores = [round( (float(total) - float(sudeste) ),2), round(float(sudeste),2)]
cores = ['Gainsboro','coral']
p_sudeste.pie(valores, labels = rotulos, colors = cores, shadow = True, explode=(0, 0.1), autopct='%1.1f%%')
p_sudeste.set_title('Percentual Sudeste')
p_sudeste.legend( ['Total','Sudeste'],loc="best")

rotulos = [ str(round( (float(total) - float(sul) ),2)) + '%' , str(round(float(sul),2)) + '%' ]
valores = [round( (float(total) - float(sul) ),2), round(float(sul),2)]
cores = ['Gainsboro','coral']
p_sul.pie(valores, labels = rotulos, colors = cores, shadow = True,  explode=(0, 0.1), autopct='%1.1f%%')
p_sul.set_title('Percentual Sul')
p_sul.legend( ['Total','Sul'], loc="best")

rotulos =[ str(round( (float(total) - float(norte) ),2)) + '%' , str(round(float(norte),2)) + '%' ]
valores = [round( (float(total) - float(norte) ),2), round(float(norte),2)]
cores = ['Gainsboro','coral']
p_norte.pie(valores, labels = rotulos, colors = cores, shadow = True)
p_norte.set_title('Percentual Norte')
p_norte.legend( ['Total','Norte'], loc="best")

rotulos =[str(round( (float(total) - float(nordeste) ),2)) + '%' , str(round(float(nordeste),2)) + '%' ]
valores = [round( (float(total) - float(nordeste) ),2), round(float(nordeste),2)]
cores = ['Gainsboro','coral']
p_nordeste.pie(valores, labels = rotulos, colors = cores, shadow = True,  explode=(0, 0.1), autopct='%1.1f%%')
p_nordeste.set_title('Percentual Nordeste')
p_nordeste.legend( ['Total','Nordeste'], loc="best")

rotulos = [str(round( (float(total) - float(centro_oeste) ),2)) + '%' , str(round(float(centro_oeste),2)) + '%' ]
valores = [round( (float(total) - float(centro_oeste) ),2), round(float(centro_oeste),2)]
cores = ['Gainsboro','coral']
p_centro_oeste.pie(valores, labels = rotulos, colors = cores, shadow = True,  explode=(0, 0.1), autopct='%1.1f%%')
p_centro_oeste.set_title('Percentual Centro-Oeste')
p_centro_oeste.legend(['Total','Centro-Oeste'], loc="best")


# super title of figure
fig.suptitle('Quantidade de Municípios por Região ', y = 1.05, fontsize=15)

# clean up whitespace padding
fig.tight_layout()

df_faixas = df_eleitorado.melt( id_vars=['cod_municipio_tse','uf','nome_municipio','total_eleitores','gen_feminino','gen_masculino','gen_nao_informado','Regiao','percentual_gen_feminino_uf','percentual_gen_masculino_uf','percentual_gen_nao_informado_uf'
], var_name='faixa_etaria', value_name='qtd_faixa')
df_faixas.groupby(by='faixa_etaria').sum()['qtd_faixa']
d = (df_faixas.groupby(by='faixa_etaria').sum()['qtd_faixa']).reset_index().rename(columns={'index':'faixa_etaria', 'qtd_faixa' : 'Frequencia_absoluta'})


d2 = pd.DataFrame({'faixa_etaria':['Total'],'Frequencia_absoluta':d['Frequencia_absoluta'].sum()})
d3 = pd.concat([d, d2], ignore_index=True,sort=True)

d3['Frequencia_relativa'] = round((d3['Frequencia_absoluta']/float(d3[d3['faixa_etaria'] == 'Total']['Frequencia_absoluta']))*100,2)


x,y = d3[d3['faixa_etaria'] != 'Total']['faixa_etaria'], d3[d3['faixa_etaria'] != 'Total']['Frequencia_absoluta']

#x, y = df_frequencia[df_frequencia['Categoria'] != 'Total']['Categoria'],df_frequencia[df_frequencia['Categoria'] != 'Total']['Frequencia_absoluta']

plt.bar(x, y,  color = 'b')
plt.xticks(rotation='vertical')


faixas = plt
faixas.title('Distribuição do eleitorado por faixas etárias')
faixas.show()
total = df_eleitorado.sum()['total_eleitores']

#df_regiao = (pd.DataFrame(df_eleitorado.groupby(by=['Regiao','uf']).sum()['total_eleitores']).groupby(by=['Regiao']).sum()['total_eleitores'])
#df_uf = (pd.DataFrame(df_eleitorado.groupby(by=['Regiao','uf']).sum()['total_eleitores']))

total = df_eleitorado.sum()['total_eleitores']
sudeste = df_eleitorado.groupby(by=['Regiao']).get_group('Sudeste')['total_eleitores'].sum()

sp = df_eleitorado.groupby(by=['Regiao','uf']).get_group(('Sudeste', 'SP')).sum()['total_eleitores']
mg = df_eleitorado.groupby(by=['Regiao','uf']).get_group(('Sudeste', 'MG')).sum()['total_eleitores']
es = df_eleitorado.groupby(by=['Regiao','uf']).get_group(('Sudeste', 'ES')).sum()['total_eleitores']
rj = df_eleitorado.groupby(by=['Regiao','uf']).get_group(('Sudeste', 'RJ')).sum()['total_eleitores']
#homens_sobr = tit.groupby(by=['Sobreviveu','Sexo'])
#print(homens_sobr.get_group(('Sim', 'Homem')).mean()['Idade'])


norte = df_eleitorado.groupby(by=['Regiao']).get_group('Norte')['total_eleitores'].sum()

am = df_eleitorado.groupby(by=['Regiao','uf']).get_group(('Norte', 'AM')).sum()['total_eleitores']
rr = df_eleitorado.groupby(by=['Regiao','uf']).get_group(('Norte', 'RR')).sum()['total_eleitores']
ap = df_eleitorado.groupby(by=['Regiao','uf']).get_group(('Norte', 'AP')).sum()['total_eleitores']
pa = df_eleitorado.groupby(by=['Regiao','uf']).get_group(('Norte', 'PA')).sum()['total_eleitores']
to = df_eleitorado.groupby(by=['Regiao','uf']).get_group(('Norte', 'TO')).sum()['total_eleitores']
ro = df_eleitorado.groupby(by=['Regiao','uf']).get_group(('Norte', 'RO')).sum()['total_eleitores']
ac = df_eleitorado.groupby(by=['Regiao','uf']).get_group(('Norte', 'AC')).sum()['total_eleitores']



nordeste = df_eleitorado.groupby(by=['Regiao']).get_group('Nordeste')['total_eleitores'].sum()

ma = df_eleitorado.groupby(by=['Regiao','uf']).get_group(('Nordeste', 'MA')).sum()['total_eleitores']
pi = df_eleitorado.groupby(by=['Regiao','uf']).get_group(('Nordeste', 'PI')).sum()['total_eleitores']
ce = df_eleitorado.groupby(by=['Regiao','uf']).get_group(('Nordeste', 'CE')).sum()['total_eleitores']
rn = df_eleitorado.groupby(by=['Regiao','uf']).get_group(('Nordeste', 'RN')).sum()['total_eleitores']
pe = df_eleitorado.groupby(by=['Regiao','uf']).get_group(('Nordeste', 'PE')).sum()['total_eleitores']
pb = df_eleitorado.groupby(by=['Regiao','uf']).get_group(('Nordeste', 'PB')).sum()['total_eleitores']
se = df_eleitorado.groupby(by=['Regiao','uf']).get_group(('Nordeste', 'SE')).sum()['total_eleitores']
al = df_eleitorado.groupby(by=['Regiao','uf']).get_group(('Nordeste', 'AL')).sum()['total_eleitores']
ba = df_eleitorado.groupby(by=['Regiao','uf']).get_group(('Nordeste', 'BA')).sum()['total_eleitores']


sul = df_eleitorado.groupby(by=['Regiao']).get_group('Sul')['total_eleitores'].sum()

pr = df_eleitorado.groupby(by=['Regiao','uf']).get_group(('Sul', 'PR')).sum()['total_eleitores']
rs = df_eleitorado.groupby(by=['Regiao','uf']).get_group(('Sul', 'RS')).sum()['total_eleitores']
sc = df_eleitorado.groupby(by=['Regiao','uf']).get_group(('Sul', 'SC')).sum()['total_eleitores']


centro_oeste = df_eleitorado.groupby(by=['Regiao']).get_group('Centro-Oeste')['total_eleitores'].sum()


mt = df_eleitorado.groupby(by=['Regiao','uf']).get_group(('Centro-Oeste', 'MT')).sum()['total_eleitores']
ms = df_eleitorado.groupby(by=['Regiao','uf']).get_group(('Centro-Oeste', 'MS')).sum()['total_eleitores']
go = df_eleitorado.groupby(by=['Regiao','uf']).get_group(('Centro-Oeste', 'GO')).sum()['total_eleitores']


gridsize = (5, 2) # 4 rows, 2 columns
fig = plt.figure(figsize=(12, 12)) # this creates a figure without axes

p_eleitores_sudeste =plt.subplot2grid(gridsize, (0, 0)) # 2nd argument = origin of individual box
p_eleitores_sudeste_uf =plt.subplot2grid(gridsize, (0, 1))

p_eleitores_sul = plt.subplot2grid(gridsize, (1, 0))
p_eleitores_sul_uf = plt.subplot2grid(gridsize, (1, 1))

p_eleitores_norte = plt.subplot2grid(gridsize,( 2,0))
p_eleitores_norte_uf = plt.subplot2grid(gridsize,( 2,1))

p_eleitores_nordeste = plt.subplot2grid(gridsize, (3, 0))
p_eleitores_nordeste_uf = plt.subplot2grid(gridsize, (3, 1))

p_eleitores_centro_oeste = plt.subplot2grid(gridsize, (4, 0))
p_eleitores_centro_oeste_uf = plt.subplot2grid(gridsize, (4, 1))

rotulos = ['Restante Brasil','Sudeste']
valores = [round( (float(total) - float(sudeste) ),2), round(float(sudeste),2)]
cores = ['Gainsboro','coral']
p_eleitores_sudeste.pie(valores, labels = rotulos, colors = cores, shadow = True, explode=(0, 0.1), autopct='%1.1f%%')
p_eleitores_sudeste.set_title('Sudeste')
p_eleitores_sudeste.axis('equal')
p_eleitores_sudeste.legend( ['Restante Brasil','Sudeste'], loc="best")


rotulos = [round(float(sp)/float(sudeste),2)*100 , round(float(rj)/float(sudeste),2)*100, round(float(mg)/float(sudeste)*100,2), round(float(es)/float(sudeste)*100,2)]
valores = [ sp , rj, mg, es ]
#cores = ['r','b','y','g']
cores = ['b','g','r','c']
p_eleitores_sudeste_uf.pie(valores, labels = rotulos, colors = cores, shadow = True)
p_eleitores_sudeste_uf.set_title('Região Sudeste')
p_eleitores_sudeste_uf.axis('equal')
p_eleitores_sudeste_uf.legend( ['SP','RJ','MG','ES'], loc="best")


#rotulos =[ str(round( (float(total) - float(norte) ),2)) + '%' , str(round(float(norte),2)) + '%' ]
rotulos = ['Restante Brasil','Nordeste']
valores = [round( (float(total) - float(nordeste) ),2), round(float(nordeste),2)]
cores = ['Gainsboro','coral']
p_eleitores_nordeste.pie(valores, labels = rotulos, colors = cores, shadow = True, explode=(0, 0.1), autopct='%1.1f%%')
p_eleitores_nordeste.set_title('Nordeste')
p_eleitores_nordeste.axis('equal')
p_eleitores_nordeste.legend( ['Restante Brasil','Nordeste'], loc="best")


rotulos = [round(float(ma)/float(nordeste),2)*100 , round(float(pi)/float(nordeste),2)*100, round(float(ce)/float(nordeste)*100,2), 
round(float(rn)/float(nordeste)*100,2),round(float(pe)/float(nordeste),2)*100 , round(float(pb)/float(nordeste),2)*100, 
round(float(se)/float(nordeste)*100,2), round(float(al)/float(nordeste)*100,2),round(float(ba)/float(nordeste)*100,2)]
valores = [ma,pi,ce,rn,pe,pb,se,al,ba]
#cores = ['r','b','y','g','r','b','y','g','r']
cores = ['b','g','r','c','m','y','k','w','Gainsboro']
p_eleitores_nordeste_uf.pie(valores, labels = rotulos, colors = cores, shadow = True)
p_eleitores_nordeste_uf.set_title('Região Nordeste')
p_eleitores_nordeste_uf.axis('equal')
p_eleitores_nordeste_uf.legend( ['MA','PI','CE','RN','PE','PB','SE','AL','BA'], loc="best")




#rotulos =[ str(round( (float(total) - float(norte) ),2)) + '%' , str(round(float(norte),2)) + '%' ]
rotulos = ['Restante Brasil','Sul']
valores = [round( (float(total) - float(sul) ),2), round(float(sul),2)]
cores = ['Gainsboro','coral']
p_eleitores_sul.pie(valores, labels = rotulos, colors = cores, shadow = True, explode=(0, 0.1), autopct='%1.1f%%')
p_eleitores_sul.set_title('Sul')
p_eleitores_sul.axis('equal')
p_eleitores_sul.legend( ['Restante Brasil','Sul'], loc="best")



rotulos = [round(float(pr)/float(sul),2)*100 , round(float(rs)/float(sul),2)*100, round(float(sc)/float(sul)*100,2)]
valores = [ pr,rs,sc]
#cores = ['r','b','y','g']
cores = ['b','g','r','c']
p_eleitores_sul_uf.pie(valores, labels = rotulos, colors = cores, shadow = True)
p_eleitores_sul_uf.set_title('Região Sul')
p_eleitores_sul_uf.axis('equal')
p_eleitores_sul_uf.legend( ['PR','RS','SC'], loc="best")




#rotulos =[ str(round( (float(total) - float(norte) ),2)) + '%' , str(round(float(norte),2)) + '%' ]
rotulos = ['Restante Brasil','Norte']
valores = [round( (float(total) - float(norte) ),2), round(float(norte),2)]
cores = ['Gainsboro','coral']
p_eleitores_norte.pie(valores, labels = rotulos, colors = cores, shadow = True, explode=(0, 0.1), autopct='%1.1f%%')
p_eleitores_norte.set_title('Região Norte')
p_eleitores_norte.axis('equal')
p_eleitores_norte.legend( ['Restante Brasil','Norte'], loc="best")



rotulos = [round(float(am)/float(norte),2)*100 , round(float(rr)/float(norte),2)*100, round(float(ap)/float(norte)*100,2), round(float(pa)/float(norte)*100,2),round(float(to)/float(norte),2)*100 , 
round(float(ro)/float(norte),2)*100, round(float(ac)/float(norte)*100,2)]
valores = [am,rr,ap,pa,to,ro,ac]
#cores = ['r','b','y','g','r','b','g']
cores = ['b','g','r','c','m','y','k']
p_eleitores_norte_uf.pie(valores, labels = rotulos, colors = cores, shadow = True)
p_eleitores_norte_uf.set_title('Região Norte')
p_eleitores_norte_uf.axis('equal')
p_eleitores_norte_uf.legend( ['AM','RR','AP','PA','TO','RO','AC'], loc="best")





#rotulos =[ str(round( (float(total) - float(norte) ),2)) + '%' , str(round(float(norte),2)) + '%' ]
rotulos = ['Restante Brasil','Centro-Oeste']
valores = [round( (float(total) - float(centro_oeste) ),2), round(float(centro_oeste),2)]
cores = ['Gainsboro','coral']
p_eleitores_centro_oeste.pie(valores, labels = rotulos, colors = cores, shadow = True, explode=(0, 0.1), autopct='%1.1f%%')
p_eleitores_centro_oeste.set_title('Região Centro-oeste')
p_eleitores_centro_oeste.axis('equal')
p_eleitores_centro_oeste.legend( ['Restante Brasil','Centro-oeste'], loc="best")


rotulos = [round(float(mt)/float(centro_oeste),2)*100 , round(float(ms)/float(centro_oeste),2)*100, round(float(go)/float(centro_oeste)*100,2)]
valores = [ mt,ms,go]
#cores = ['r','b','y']
cores = ['b','g','r']
p_eleitores_centro_oeste_uf.pie(valores, labels = rotulos, colors = cores, shadow = True)
p_eleitores_centro_oeste_uf.set_title('Regiaõ Centro-Oeste')
p_eleitores_centro_oeste_uf.axis('equal')
p_eleitores_centro_oeste_uf.legend( ['MT','MS','GO'], loc="best")




# super title of figure
fig.suptitle('Percentual de eleitores por região ', y = 1.05, fontsize=15)

# clean up whitespace padding
fig.tight_layout()

#inspirado em http://emptypipes.org/2013/11/09/matplotlib-multicategory-barchart/
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import operator as o

import numpy as np

d_sexo_f = pd.DataFrame( df_eleitorado.groupby(by=['Regiao'])['gen_feminino'].sum().reset_index().rename(columns={'index':'Regiao','gen_feminino':'Quantidade'}) )
d_sexo_f['Genero'] = 'Feminino'

d_sexo_m = pd.DataFrame( df_eleitorado.groupby(by=['Regiao'])['gen_masculino'].sum().reset_index().rename(columns={'index':'Regiao','gen_masculino':'Quantidade'}) )
d_sexo_m['Genero'] = 'Masculino'

d_sexo_i = pd.DataFrame( df_eleitorado.groupby(by=['Regiao'])['gen_nao_informado'].sum().reset_index().rename(columns={'index':'Regiao','gen_nao_informado':'Quantidade'}) )
d_sexo_i['Genero'] = 'Indefinido'
#p2 = pd.DataFrame( df_eleitorado.groupby(by=['uf'])['gen_masculino'].sum().reset_index().rename(columns={'index':'uf'}) )
#p3 = pd.DataFrame( df_eleitorado.groupby(by=['uf'])['gen_nao_informado'].sum().reset_index().rename(columns={'index':'uf'}) )

d_sexo = pd.concat([d_sexo_f, d_sexo_m,d_sexo_i], ignore_index=True)
d_sexo = d_sexo[['Genero','Regiao','Quantidade']]

dpoints = d_sexo.values
#dpoints = np.array([['rosetta', '1mfq', 9.97],
#           ['rosetta', '1gid', 27.31],
#           ['rosetta', '1y26', 5.77],
#           ['rnacomposer', '1mfq', 5.55],
#           ['rnacomposer', '1gid', 37.74],
#           ['rnacomposer', '1y26', 5.77],
#           ['random', '1mfq', 10.32],
#           ['random', '1gid', 31.46],
#           ['random', '1y26', 18.16]])

fig = plt.figure()
ax = fig.add_subplot(111)

def barplot(ax, dpoints):
    '''
    Create a barchart for data across different categories with
    multiple conditions for each category.
    
    @param ax: The plotting axes from matplotlib.
    @param dpoints: The data set as an (n, 3) numpy array
    '''
    
    # Aggregate the conditions and the categories according to their
    # mean values
    conditions = [(c, np.mean(dpoints[dpoints[:,0] == c][:,2].astype(float))) 
                  for c in np.unique(dpoints[:,0])]
    categories = [(c, np.mean(dpoints[dpoints[:,1] == c][:,2].astype(float))) 
                  for c in np.unique(dpoints[:,1])]
    
    # sort the conditions, categories and data so that the bars in
    # the plot will be ordered by category and condition
    conditions = [c[0] for c in sorted(conditions, key=o.itemgetter(1))]
    categories = [c[0] for c in sorted(categories, key=o.itemgetter(1))]
    
    dpoints = np.array(sorted(dpoints, key=lambda x: categories.index(x[1])))

    # the space between each set of bars
    space = 0.3
    n = len(conditions)
    width = (1 - space) / (len(conditions))
    
    # Create a set of bars at each position
    for i,cond in enumerate(conditions):
        indeces = range(1, len(categories)+1)
        vals = dpoints[dpoints[:,0] == cond][:,2].astype(np.float)
        pos = [j - (1 - space) / 2. + i * width for j in indeces]
        ax.bar(pos, vals, width=width, label=cond, 
               color=cm.Accent(float(i) / n))
    
    # Set the x-axis tick labels to be equal to the categories
    ax.set_xticks(indeces)
    ax.set_xticklabels(categories)
    plt.setp(plt.xticks()[1], rotation=90)
    
    # Add the axis labels
    ax.set_ylabel("Eleitores")
 
    plt.title('Agrupamento de genero por regiões do Brasil')
    
    # Add a legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='upper left')
        
barplot(ax, dpoints)
#savefig('barchart_3.png')
plt.show()

df = pd.read_csv('../input/anv.csv', delimiter=',')
df.head(1)