import sklearn
print(format(sklearn.__version__))
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime as dt
from sklearn import linear_model, svm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from plotly.offline import iplot, init_notebook_mode
from plotly.graph_objs import Scatter, Figure, Layout
init_notebook_mode()
import plotly.graph_objs as go
import numpy as np

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
abril = pd.read_csv(
    "../input/2290_01042018_30042018.csv",
    delimiter=",",
    dtype = {
        "numero_de_serie":int,
        "milissegundo":int,
        "faixa":int,
        "velocidade_entrada":int,
        "velocidade_saida":int,
        "classificacao":int,
        "tamanho": float,
        "placa":str,
        "tempo_ocupacao_laco":int
    },
    parse_dates=["data_hora"]
)
marco = pd.read_csv(
    "../input/2290_01032018_31032018.csv",
    delimiter=",",
    dtype = {
        "numero_de_serie":int,
        "milissegundo":int,
        "faixa":int,
        "velocidade_entrada":int,
        "velocidade_saida":int,
        "classificacao":int,
        "tamanho": float,
        "placa":str,
        "tempo_ocupacao_laco":int
    },
    parse_dates=["data_hora"]
)
maio = pd.read_csv(
    "../input/2290_01052016_31052016.csv",
    delimiter=",",
    dtype = {
        "numero_de_serie":int,
        "milissegundo":int,
        "faixa":int,
        "velocidade_entrada":int,
        "velocidade_saida":int,
        "classificacao":int,
        "tamanho": float,
        "placa":str,
        "tempo_ocupacao_laco":int
    },
    parse_dates=["data_hora"]
)
def SeparaFaixas(tabela, dia, faixa):
    tabela['datahora_mili'] = tabela['data_hora'] + pd.to_timedelta(tabela['milissegundo'], unit='ms')
    tabela['dia_semana'] = tabela["data_hora"].dt.dayofweek
    tabela_sep = tabela[tabela['dia_semana'] == dia].copy()
    tabela_fx = tabela_sep[tabela_sep['faixa']==faixa][['datahora_mili', 'velocidade_entrada', 'velocidade_saida']]
    
    tabela_fx['time_diff'] = tabela_fx['datahora_mili'].diff()
    tabela_fx['velocidade_m/s'] = np.round(tabela_fx['velocidade_entrada']/3.6, 2)
    tabela_fx['timediff_seconds'] = tabela_fx['time_diff'].dt.total_seconds()
    tabela_fx['espacamento_metros'] = tabela_fx['velocidade_m/s'] * tabela_fx['timediff_seconds']
    tabela_fx = tabela_fx[np.isfinite(tabela_fx['velocidade_m/s'])]
    tabela_fx.set_index('datahora_mili', inplace = True, drop = True)
    tabela_fx_grouped = tabela_fx.resample("10T", label="left", closed="right")
    return AgrupaDados(tabela_fx_grouped, dia)
def AgrupaDados(tab, dia):
    tab_10min = pd.DataFrame()
    tab_10min['espacamento_medio'] = tab.mean()['espacamento_metros']
    tab_10min['densidade_veic/km'] = (1/tab_10min["espacamento_medio"]) * 1000
    tab_10min = tab_10min[tab_10min.index.get_level_values('datahora_mili').dayofweek == dia]
    return tab_10min
def XYArraySplit(tab_treino, tab_teste):
    x = pd.DataFrame()
    X = pd.DataFrame()
    y = pd.DataFrame()
    Y = pd.DataFrame()
    
    y['Den. Atual'] = [row[0] for row in tab_treino]
    y['Den. 10mins'] = [row[1] for row in tab_treino]
    y['Den. 20mins'] = [row[2] for row in tab_treino]
    y['Den. 30mins'] = [row[3] for row in tab_treino]
    Y['Den. Atual'] = [row[0] for row in tab_teste]
    Y['Den. 10mins'] = [row[1] for row in tab_teste]
    Y['Den. 20mins'] = [row[2] for row in tab_teste]
    Y['Den. 30mins'] = [row[3] for row in tab_teste]
    x['den-3'] = [row[4] for row in tab_treino]
    x['den-4'] = [row[5] for row in tab_treino]
    x['den-5'] = [row[6] for row in tab_treino]
    x['semana_anterior'] = [row[7] for row in tab_treino]
    x['semana_anterior-1'] = [row[8] for row in tab_treino]
    X['den-3'] = [row[4] for row in tab_teste]
    X['den-4'] = [row[5] for row in tab_teste]
    X['den-5'] = [row[6] for row in tab_teste]
    X['semana_anterior'] = [row[7] for row in tab_teste]
    X['semana_anterior-1'] = [row[8] for row in tab_teste]
    
    return x, y, X, Y
def TabelaDePrevisao(tab, dia_semana):
    #tab = Filtrar(dados, dia_semana)
    tabela = pd.DataFrame()
    tabela['den'] = tab['densidade_veic/km']
    tabela['den 10mins'] = tab['densidade_veic/km'].shift(-1)
    tabela['den 20mins'] = tab['densidade_veic/km'].shift(-2)
    tabela['den 30mins'] = tab['densidade_veic/km'].shift(-3)
    #tabela = tabela[tabela['den_atual'] > 0] #retirar essa linha caso queira contar as médias 0
    tabela['den-3'] = tab['densidade_veic/km'].shift(3)
    tabela['den-4'] = tab['densidade_veic/km'].shift(4)
    tabela['den-5'] = tab['densidade_veic/km'].shift(5)
    tabela['semana_anterior'] = tab['densidade_veic/km'].shift(int((60 / 10) * 24))
    tabela['semana_anterior-1'] = tab['densidade_veic/km'].shift(int((60 / 10) * 24) + 1)
    tabela_filtrada = np.round(Filtrar(tabela, dia_semana), 2)
    #tabela_filtrada = tabela_filtrada[tabela_filtrada >= 0]
    return tabela_filtrada
def Filtrar(dados, dia_semana):
    dados_filtrado = dados[np.isfinite(dados)]
    dados_filtrado['dia_semana'] = dados_filtrado.index.get_level_values('datahora_mili').dayofweek
    dados_filtrado = dados_filtrado[dados_filtrado.index.dayofweek == dia_semana]
    dados_filtrado = dados_filtrado.drop('dia_semana', axis=1)
    return dados_filtrado
def Impute(tab_treino, tab_teste):
    imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    iTreino = imputer.fit_transform(tab_treino)
    iTeste = imputer.transform(tab_teste)
    return iTreino, iTeste
def MontarGrafico(tabela) :
    i = 0
    grafico = []
    dias = np.unique(tabela.index.get_level_values('datahora_mili').day)
    for len in dias :
        tabela_grafico = tabela[tabela.index.get_level_values('datahora_mili').day == dias[i]]
        trace = go.Scatter(
            x = tabela_grafico.index.get_level_values('datahora_mili').time,
            y = tabela_grafico['densidade_veic/km'],
            mode = "lines",
            name = "Densidade - dia %s" % dias[i]
        )
        grafico.append(trace)
        i += 1
    return grafico
def GraficoPredissao(tabela, predicao,indice) :
    grafico = []
    dias = np.unique(tabela.index.get_level_values('datahora_mili').day)
    tabela_grafico = tabela[tabela.index.get_level_values('datahora_mili').day == dias[indice]]
    trace = go.Scatter(
        x = tabela_grafico.index.get_level_values('datahora_mili').time,
        y = tabela_grafico['den'],
        mode = "lines",
        name = "Densidade - dia %s" % dias[indice]
    )
    grafico.append(trace)
    predicao_grafico = predicao[predicao.index.get_level_values('datahora_mili').day == dias[indice]]
    trace = go.Scatter(
        x = predicao_grafico.index.get_level_values('datahora_mili').time,
        y =predicao_grafico['Den. 10mins'],
        mode = "lines",
        name = "Previsão 10mins - dia %s" % dias[indice]
    )
    grafico.append(trace)
    trace = go.Scatter(
        x = predicao_grafico.index.get_level_values('datahora_mili').time,
        y =predicao_grafico['Den. 20mins'],
        mode = "lines",
        name = "Previsão 20mins - dia %s" % dias[indice]
    )
    grafico.append(trace)
    trace = go.Scatter(
        x = predicao_grafico.index.get_level_values('datahora_mili').time,
        y =predicao_grafico['Den. 30mins'],
        mode = "lines",
        name = "Previsão 30mins - dia %s" % dias[indice]
    )
    grafico.append(trace)
    layout = Layout(
        title = "Previsão de Densidade",
        xaxis=dict(title="Tempo"),
        yaxis=dict(title="Densidade [Carros/km]")
    )
    graph = Figure(data = grafico, layout = layout)
    return graph
def PRegressao(x_treino, x_teste, y_treino, y_teste, periodo, grau):
    reg = make_pipeline(PolynomialFeatures(grau), linear_model.LinearRegression())
    #reg = linear_model.LinearRegression()
    reg.fit(x_treino, y_treino)
    y_predissao = pd.DataFrame()
    y_predissao['valor'] = reg.predict(x_teste)
    #y_predissao['erro %d0mins' % periodo] = mean_squared_error(y_teste, y_predissao['valor'])
    #y_predissao['variação %d0mins' % periodo] = r2_score(y_teste, y_predissao['valor'])
    print(r2_score(y_teste, y_predissao['valor']))
    return y_predissao
def Ridge(x_treino, x_teste, y_treino, y_teste, grau) :  
    #ridge = svm.SVR(gamma = 'auto', kernel='linear')
    ridge = make_pipeline(PolynomialFeatures(grau), linear_model.Ridge(alpha = 0.01, fit_intercept = False))
    ridge.fit(x_treino, y_treino)
    y_predissao = pd.DataFrame()
    y_predissao['valor'] = ridge.predict(x_teste)
    #y_predissao['erro'] = mean_squared_error(y_teste, y_predissao['valor'])
    #y_predissao['variação'] = r2_score(y_teste, y_predissao['valor'])
    print(r2_score(y_teste, y_predissao['valor']))
    
    return y_predissao
def SVR(x_treino, x_teste, y_treino, y_teste, grau) :  
    svr = svm.SVR(gamma = 'auto', kernel='linear')
    #svr = make_pipeline(PolynomialFeatures(grau), svm.SVR(gamma='auto', kernel='linear'))
    svr.fit(x_treino, y_treino)
    y_predissao = pd.DataFrame()
    y_predissao['valor'] = svr.predict(x_teste)
    #y_predissao['erro'] = mean_squared_error(y_teste, y_predissao['valor'])
    #y_predissao['variação'] = r2_score(y_teste, y_predissao['valor'])
    print(r2_score(y_teste, y_predissao['valor']))
    
    return y_predissao
def LSVR(x_treino, x_teste, y_treino, y_teste) :    
    lsvr = svm.LinearSVR(epsilon = 0, max_iter=90000)
    lsvr.fit(x_treino, y_treino)
    y_predissao = pd.DataFrame()
    y_predissao['valor'] = lsvr.predict(x_teste)
    #y_predissao['erro'] = mean_squared_error(y_teste, y_predissao['valor'])
    #y_predissao['variação'] = r2_score(y_teste, y_predissao['valor'])
    print(r2_score(y_teste, y_predissao['valor']))
    
    return y_predissao
def PreverPRegressao(tabela_treino, tabela_teste, grau) :
    tab_treino, tab_teste = Impute(tabela_treino, tabela_teste)
    #tab_treino = tabela_treino.fillna(np.mean(tabela_treino['den']))
    #tab_teste = tabela_teste.fillna(np.mean(tabela_teste['den']))
    predissao = pd.DataFrame()
    x_treino, y_treino, x_teste, y_teste = XYArraySplit(tab_treino, tab_teste)
    predissao["Den. Atual"] = y_teste['Den. Atual']
    predissao["Den. 10mins"] = np.round(PRegressao(x_treino, x_teste, y_treino["Den. 10mins"], y_teste["Den. 10mins"], 1, grau), 2)
    predissao["Den. 20mins"] = np.round(PRegressao(x_treino, x_teste, y_treino["Den. 20mins"], y_teste["Den. 20mins"], 2, grau), 2)
    predissao["Den. 30mins"] = np.round(PRegressao(x_treino, x_teste, y_treino["Den. 30mins"], y_teste["Den. 30mins"], 3, grau), 2)
    predissao.set_index(tabela_teste.index, inplace = True, drop = False)
    return predissao
def PreverSVR(tabela_treino, tabela_teste, grau) :
    tab_treino, tab_teste = Impute(tabela_treino, tabela_teste)
    predissao = pd.DataFrame()
    x_treino, y_treino, x_teste, y_teste = XYArraySplit(tab_treino, tab_teste)
    predissao['Den. Atual'] = y_teste['Den. Atual']
    predissao['Den. 10mins'] = np.round(SVR(x_treino, x_teste, y_treino['Den. 10mins'], y_teste['Den. 10mins'], grau), 2)
    predissao['Den. 20mins'] = np.round(SVR(x_treino, x_teste, y_treino['Den. 20mins'], y_teste['Den. 20mins'], grau), 2)
    predissao['Den. 30mins'] = np.round(SVR(x_treino, x_teste, y_treino['Den. 30mins'], y_teste['Den. 30mins'], grau), 2)
    predissao.set_index(tabela_teste.index, inplace = True, drop = False)
    return predissao
def PreverLSVR(tabela_treino, tabela_teste, grau) :
    tab_treino, tab_teste = Impute(tabela_treino, tabela_teste)
    predissao = pd.DataFrame()
    x_treino, y_treino, x_teste, y_teste = XYArraySplit(tab_treino, tab_teste)
    predissao['Den. Atual'] = y_teste['Den. Atual']
    predissao['Den. 10mins'] = np.round(LSVR(x_treino, x_teste, y_treino['Den. 10mins'], y_teste['Den. 10mins']), 2)
    predissao['Den. 20mins'] = np.round(LSVR(x_treino, x_teste, y_treino['Den. 20mins'], y_teste['Den. 20mins']), 2)
    predissao['Den. 30mins'] = np.round(LSVR(x_treino, x_teste, y_treino['Den. 30mins'], y_teste['Den. 30mins']), 2)
    predissao.set_index(tabela_teste.index, inplace = True, drop = False)
    return predissao
def PreverRidge(tabela_treino, tabela_teste, grau) :
    tab_treino, tab_teste = Impute(tabela_treino, tabela_teste)
    predissao = pd.DataFrame()
    x_treino, y_treino, x_teste, y_teste = XYArraySplit(tab_treino, tab_teste)
    predissao['Den. Atual'] = y_teste['Den. Atual']
    predissao['Den. 10mins'] = np.round(Ridge(x_treino, x_teste, y_treino['Den. 10mins'], y_teste['Den. 10mins'], grau), 2)
    predissao['Den. 20mins'] = np.round(Ridge(x_treino, x_teste, y_treino['Den. 20mins'], y_teste['Den. 20mins'], grau), 2)
    predissao['Den. 30mins'] = np.round(Ridge(x_treino, x_teste, y_treino['Den. 30mins'], y_teste['Den. 30mins'], grau), 2)
    predissao.set_index(tabela_teste.index, inplace = True, drop = False)
    return predissao
dia = 1
faixa = 2
grau = 2

teste = TabelaDePrevisao(SeparaFaixas(maio, dia, faixa), dia)
treino = TabelaDePrevisao(SeparaFaixas(marco, dia, faixa), dia)
regr = PreverPRegressao(treino, teste, grau)

iplot(GraficoPredissao(teste, regr, 2))

dia = 0
faixa = 1
grau = 1

teste = TabelaDePrevisao(SeparaFaixas(maio, dia, faixa), dia)
treino = TabelaDePrevisao(SeparaFaixas(marco, dia, faixa), dia)
regr = PreverSVR(treino, teste, grau)

iplot(GraficoPredissao(teste, regr, 1))
dia = 0
faixa = 1
grau = 1

teste = TabelaDePrevisao(SeparaFaixas(maio, dia, faixa), dia)
treino = TabelaDePrevisao(SeparaFaixas(marco, dia, faixa), dia)
regr = PreverLSVR(treino, teste, grau)

iplot(GraficoPredissao(teste, regr, 1))
dia = 0
faixa = 1
grau = 2

teste = TabelaDePrevisao(SeparaFaixas(maio, dia, faixa), dia)
treino = TabelaDePrevisao(SeparaFaixas(marco, dia, faixa), dia)
regr = PreverRidge(treino, teste, grau)

iplot(GraficoPredissao(teste, regr, 3))
dia = 0
faixa = 2

iplot(MontarGrafico(SeparaFaixas(abril, dia, faixa)))
dia = 1
faixa = 2

iplot(MontarGrafico(SeparaFaixas(abril, dia, faixa)))
dia = 3
faixa = 2

iplot(MontarGrafico(SeparaFaixas(abril, dia, faixa)))