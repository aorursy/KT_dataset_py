import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime as dt
from sklearn import linear_model, svm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode()
import plotly.graph_objs as go
from plotly.graph_objs import Scatter, Figure, Layout

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
dados_marco = pd.read_csv(
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
dados_abril = pd.read_csv(
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
dados_geral = pd.DataFrame()
dados_geral = dados_geral.append(dados_marco)
dados_geral = dados_geral.append(dados_abril)
dados_prova = pd.read_csv(
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
def AvgSpeed(serie):
    if len(serie) == 0:
        return 0
    else:
        return np.round(len(serie)/(sum(1/serie)),2)
def VelocidadeFaixa(base_dados, dia, faixa, intervalo):
    if dia > -1 and dia < 7 and faixa > 0:
        base_dados["dia_semana"] = base_dados["data_hora"].dt.dayofweek
        dados_calc = base_dados[base_dados['faixa'] == faixa].copy()
        dados_calc = dados_calc[dados_calc['dia_semana'] == dia]
        dados_calc = dados_calc[np.isfinite(dados_calc['velocidade_entrada'])]
        dados_calc['velocidade_km'] = np.round(dados_calc['velocidade_entrada'],2)
        dados_calc = dados_calc.query('velocidade_km > 0 & velocidade_km < 150')
        tabela = dados_calc.set_index("data_hora", inplace = True, drop = True)
        tabela = dados_calc.resample("%dT" % intervalo, label = "left", closed = "right")
        tabela_agregada = tabela.agg(
            {
                "velocidade_km" : AvgSpeed
            }
        ).rename(
            columns={
                "velocidade_km" : "velocidade_media_km"
            }
        )
        return TabelaDePrevisao(tabela_agregada, intervalo, dia)
    else:
        return None
def TabelaDePrevisao(dados, intervalo, dia_semana):
    dados = Filtrar(dados, dia_semana)
    tabela = pd.DataFrame()
    tabela['vel_atual'] = dados['velocidade_media_km']
    tabela['vel+1'] = dados['velocidade_media_km'].shift(-1)
    tabela['vel+2'] = dados['velocidade_media_km'].shift(-2)
    tabela['vel+3'] = dados['velocidade_media_km'].shift(-3)
    tabela['vel-3'] = dados['velocidade_media_km'].shift(3)
    tabela['vel-4'] = dados['velocidade_media_km'].shift(4)
    tabela['vel-5'] = dados['velocidade_media_km'].shift(5)
    tabela['semana_anterior'] = dados['velocidade_media_km'].shift(int((60 / intervalo) * 24))
    tabela['semana_anterior-1'] = dados['velocidade_media_km'].shift(int((60 / intervalo) * 24) + 1)
    tabela_filtrada = Filtrar(tabela, dia_semana)
    return tabela_filtrada
def Filtrar(dados, dia_semana):
    dados_filtrado = dados[np.isfinite(dados)]
    dados_filtrado['dia_semana'] = dados_filtrado.index.get_level_values('data_hora').dayofweek
    dados_filtrado = dados_filtrado[dados_filtrado.index.dayofweek == dia_semana]
    dados_filtrado = dados_filtrado.drop('dia_semana', axis=1)
    return dados_filtrado
def GraficoCompara(tabela, indice) :
    grafico = []
    dias = np.unique(tabela.index.get_level_values('dataHoraConsulta').day)
    tabela_grafico = tabela[tabela.index.get_level_values('dataHoraConsulta').day == dias[indice]]
    trace = go.Scatter(
        x = tabela_grafico.index.get_level_values('dataHoraConsulta').time,
        y = tabela_grafico['ValorAtual'],
        mode = "lines",
        name = "Velocidade - dia %s" % dias[indice]
    )
    trace1 = go.Scatter(
        x = tabela_grafico.index.get_level_values('dataHoraConsulta').time,
        y = tabela_grafico['PrevisaoDez'],
        mode = "lines",
        name = "Previsão - 10 min"
    )
    trace2 = go.Scatter(
        x = tabela_grafico.index.get_level_values('dataHoraConsulta').time,
        y = tabela_grafico['PrevisaoVinte'],
        mode = "lines",
        name = "Previsão - 20 min"
    )
    trace3 = go.Scatter(
        x = tabela_grafico.index.get_level_values('dataHoraConsulta').time,
        y = tabela_grafico['PrevisaoTrinta'],
        mode = "lines",
        name = "Previsão - 30 min"
    )
    grafico = [trace, trace1, trace2, trace3]
    layout = Layout(
        title = "Previsão de Velocidade Média",
        xaxis=dict(title="Tempo"),
        yaxis=dict(title="Velocidade Média [km/h]")
    )
    graph = Figure(data = grafico, layout = layout)
    return graph
def Grafico(predicao,indice) :
    grafico = []
    dias = np.unique(predicao.index.get_level_values('data_hora').day)
    tabela_grafico = predicao[predicao.index.get_level_values('data_hora').day == dias[indice]]
    trace = go.Scatter(
        x = tabela_grafico.index.get_level_values('data_hora').time,
        y = tabela_grafico['velAtual'],
        mode = "lines",
        name = "Velocidade - dia %s" % dias[indice]
    )
    grafico.append(trace)
    predicao_grafico = predicao[predicao.index.get_level_values('data_hora').day == dias[indice]]
    trace = go.Scatter(
        x = predicao_grafico.index.get_level_values('data_hora').time,
        y =predicao_grafico['Vel+1'],
        mode = "lines",
        name = "Predição - 10min" % dias[indice]
    )
    grafico.append(trace)
    trace = go.Scatter(
        x = predicao_grafico.index.get_level_values('data_hora').time,
        y =predicao_grafico['Vel+2'],
        mode = "lines",
        name = "Predição - 20min" % dias[indice]
    )
    grafico.append(trace)
    trace = go.Scatter(
        x = predicao_grafico.index.get_level_values('data_hora').time,
        y =predicao_grafico['Vel+3'],
        mode = "lines",
        name = "Predição - 30min" % dias[indice]
    )
    grafico.append(trace)
    layout = Layout(
        title = "Previsão de Velocidade Média",
        xaxis=dict(title="Tempo"),
        yaxis=dict(title="Velocidade Média [km/h]")
    )
    graph = Figure(data = grafico, layout = layout)
    return graph
def Impute(tab_treino, tab_teste):
    imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    iTreino = imputer.fit_transform(tab_treino)
    iTeste = imputer.transform(tab_teste)
    return iTreino, iTeste
def XYArraySplit(tab_treino, tab_teste):
    x = pd.DataFrame()
    X = pd.DataFrame()
    y = pd.DataFrame()
    Y = pd.DataFrame()
    
    y['velAtual'] = [row[0] for row in tab_treino]
    y['vel+1'] = [row[1] for row in tab_treino]
    y['vel+2'] = [row[2] for row in tab_treino]
    y['vel+3'] = [row[3] for row in tab_treino]
    Y['velAtual'] = [row[0] for row in tab_teste]
    Y['vel+1'] = [row[1] for row in tab_teste]
    Y['vel+2'] = [row[2] for row in tab_teste]
    Y['vel+3'] = [row[3] for row in tab_teste]
    x['vel-3'] = [row[4] for row in tab_treino]
    x['vel-4'] = [row[5] for row in tab_treino]
    x['vel-5'] = [row[6] for row in tab_treino]
    x['semana_anterior'] = [row[7] for row in tab_treino]
    x['semana_anterior-1'] = [row[8] for row in tab_treino]
    X['vel-3'] = [row[4] for row in tab_teste]
    X['vel-4'] = [row[5] for row in tab_teste]
    X['vel-5'] = [row[6] for row in tab_teste]
    X['semana_anterior'] = [row[7] for row in tab_teste]
    X['semana_anterior-1'] = [row[8] for row in tab_teste]
    
    return x, y, X, Y
def RegressaoMes(x_treino, x_teste, y_treino, y_teste) :    
    regr = linear_model.LinearRegression()
    regr.fit(x_treino, y_treino)
    y_predissao = pd.DataFrame()
    y_predissao['valor'] = regr.predict(x_teste)
    print(r2_score(y_teste, y_predissao['valor']))
    return y_predissao
def RegressaoRidge(x_treino, x_teste, y_treino, y_teste) :    
    regr = linear_model.Ridge(alpha = 0.01, fit_intercept = False)
    regr.fit(x_treino, y_treino)
    y_predissao = pd.DataFrame()
    y_predissao['valor'] = regr.predict(x_teste)
    print(r2_score(y_teste, y_predissao['valor']))
    return y_predissao
def Lasso(x_treino, x_teste, y_treino, y_teste) :    
    a = 0.1
    lasso = linear_model.Lasso(alpha = 0.01)
    lasso.fit(x_treino, y_treino)
    y_predissao = pd.DataFrame()
    y_predissao['valor'] = lasso.predict(x_teste)
    print(r2_score(y_teste, y_predissao['valor']))
    
    return y_predissao
def ENet(x_treino, x_teste, y_treino, y_teste) :    
    a = 0.1
    enet = linear_model.ElasticNet(alpha = 0.01, l1_ratio = 0.01)
    enet.fit(x_treino, y_treino)
    y_predissao = pd.DataFrame()
    y_predissao['valor'] = enet.predict(x_teste)
    print(r2_score(y_teste, y_predissao['valor']))
    
    return y_predissao
def LARS(x_treino, x_teste, y_treino, y_teste) :    
    lars = linear_model.Lars(precompute = 'auto')
    lars.fit(x_treino, y_treino)
    y_predissao = pd.DataFrame()
    y_predissao['valor'] = lars.predict(x_teste)
    print(r2_score(y_teste, y_predissao['valor']))
    
    return y_predissao
def SVR(x_treino, x_teste, y_treino, y_teste) :    
    svr = svm.SVR(gamma = 'scale')
    svr.fit(x_treino, y_treino)
    y_predissao = pd.DataFrame()
    y_predissao['valor'] = svr.predict(x_teste)
    print(r2_score(y_teste, y_predissao['valor']))
    
    return y_predissao
def LSVR(x_treino, x_teste, y_treino, y_teste) :    
    lsvr = svm.LinearSVR(max_iter = 10000)
    lsvr.fit(x_treino, y_treino)
    y_predissao = pd.DataFrame()
    y_predissao['valor'] = lsvr.predict(x_teste)
    print(r2_score(y_teste, y_predissao['valor']))
    
    return y_predissao
def PRidge(x_treino, x_teste, y_treino, y_teste) :    
    ridge = make_pipeline(PolynomialFeatures(1), linear_model.Ridge())
    ridge.fit(x_treino, y_treino)
    y_predissao = pd.DataFrame()
    y_predissao['valor'] = ridge.predict(x_teste)
    print(r2_score(y_teste, y_predissao['valor']))
    
    return y_predissao
def PSVR(x_treino, x_teste, y_treino, y_teste, grau) :  
    svr = make_pipeline(PolynomialFeatures(grau), svm.SVR(gamma = 'auto'))
    svr.fit(x_treino, y_treino)
    y_predissao = pd.DataFrame()
    y_predissao['valor'] = svr.predict(x_teste)
    print(r2_score(y_teste, y_predissao['valor']))
    
    return y_predissao
def PRegressao(x_treino, x_teste, y_treino, y_teste, grau):
    reg = make_pipeline(PolynomialFeatures(grau), linear_model.LinearRegression())
    reg.fit(x_treino, y_treino)
    y_predissao = pd.DataFrame()
    y_predissao['valor'] = reg.predict(x_teste)
    print(r2_score(y_teste, y_predissao['valor']))
    return y_predissao
def PreverRegressao(tabela_treino, tabela_teste) :
    tabela_treino_filtrada, tabela_teste_filtrada = Impute(tabela_treino, tabela_teste)
    predissao = pd.DataFrame()
    x_treino, y_treino, x_teste, y_teste = XYArraySplit(tabela_treino_filtrada, tabela_teste_filtrada)
    predissao['velAtual'] = y_teste['velAtual']
    predissao["Vel+1"] = np.round(RegressaoMes(x_treino, x_teste, y_treino["vel+1"], y_teste["vel+1"]), 2)
    predissao["Vel+2"] = np.round(RegressaoMes(x_treino, x_teste, y_treino["vel+2"], y_teste["vel+2"]), 2)
    predissao["Vel+3"] = np.round(RegressaoMes(x_treino, x_teste, y_treino["vel+3"], y_teste["vel+3"]), 2)
    predissao.set_index(tabela_teste.index, inplace = True, drop = False)
    return predissao
def PreverRidge(tabela_treino, tabela_teste) :
    tabela_treino_filtrada, tabela_teste_filtrada = Impute(tabela_treino, tabela_teste)
    predissao = pd.DataFrame()
    x_treino, y_treino, x_teste, y_teste = XYArraySplit(tabela_treino_filtrada, tabela_teste_filtrada)
    predissao['velAtual'] = y_teste['velAtual']
    predissao["Vel+1"] = np.round(PRidge(x_treino, x_teste, y_treino["vel+1"], y_teste["vel+1"]), 2)
    predissao["Vel+2"] = np.round(PRidge(x_treino, x_teste, y_treino["vel+2"], y_teste["vel+2"]), 2)
    predissao["Vel+3"] = np.round(PRidge(x_treino, x_teste, y_treino["vel+3"], y_teste["vel+3"]), 2)
    predissao.set_index(tabela_teste.index, inplace = True, drop = False)
    return predissao
def PreverLasso(tabela_treino, tabela_teste) :
    tabela_treino_filtrada, tabela_teste_filtrada = Impute(tabela_treino, tabela_teste)
    predissao = pd.DataFrame()
    x_treino, y_treino, x_teste, y_teste = XYArraySplit(tabela_treino_filtrada, tabela_teste_filtrada)
    predissao['velAtual'] = y_teste['velAtual']
    predissao["Vel+1"] = np.round(Lasso(x_treino, x_teste, y_treino["vel+1"], y_teste["vel+1"]), 2)
    predissao["Vel+2"] = np.round(Lasso(x_treino, x_teste, y_treino["vel+2"], y_teste["vel+2"]), 2)
    predissao["Vel+3"] = np.round(Lasso(x_treino, x_teste, y_treino["vel+3"], y_teste["vel+3"]), 2)
    predissao.set_index(tabela_teste.index, inplace = True, drop = False)
    return predissao
def PreverENet(tabela_treino, tabela_teste) :
    tabela_treino_filtrada, tabela_teste_filtrada = Impute(tabela_treino, tabela_teste)
    predissao = pd.DataFrame()
    x_treino, y_treino, x_teste, y_teste = XYArraySplit(tabela_treino_filtrada, tabela_teste_filtrada)
    predissao['velAtual'] = y_teste['velAtual']
    predissao["Vel+1"] = np.round(ENet(x_treino, x_teste, y_treino["vel+1"], y_teste["vel+1"]), 2)
    predissao["Vel+2"] = np.round(ENet(x_treino, x_teste, y_treino["vel+2"], y_teste["vel+2"]), 2)
    predissao["Vel+3"] = np.round(ENet(x_treino, x_teste, y_treino["vel+3"], y_teste["vel+3"]), 2)
    predissao.set_index(tabela_teste.index, inplace = True, drop = False)
    return predissao
def PreverLARS(tabela_treino, tabela_teste) :
    tabela_treino_filtrada, tabela_teste_filtrada = Impute(tabela_treino, tabela_teste)
    predissao = pd.DataFrame()
    x_treino, y_treino, x_teste, y_teste = XYArraySplit(tabela_treino_filtrada, tabela_teste_filtrada)
    predissao['velAtual'] = y_teste['velAtual']
    predissao["Vel+1"] = np.round(LARS(x_treino, x_teste, y_treino["vel+1"], y_teste["vel+1"]), 2)
    predissao["Vel+2"] = np.round(LARS(x_treino, x_teste, y_treino["vel+2"], y_teste["vel+2"]), 2)
    predissao["Vel+3"] = np.round(LARS(x_treino, x_teste, y_treino["vel+3"], y_teste["vel+3"]), 2)
    predissao.set_index(tabela_teste.index, inplace = True, drop = False)
    return predissao
def PreverSVR(tabela_treino, tabela_teste) :
    tabela_treino_filtrada, tabela_teste_filtrada = Impute(tabela_treino, tabela_teste)
    predissao = pd.DataFrame()
    x_treino, y_treino, x_teste, y_teste = XYArraySplit(tabela_treino_filtrada, tabela_teste_filtrada)
    predissao['velAtual'] = y_teste['velAtual']
    predissao["Vel+1"] = np.round(SVR(x_treino, x_teste, y_treino["vel+1"], y_teste["vel+1"]), 2)
    predissao["Vel+2"] = np.round(SVR(x_treino, x_teste, y_treino["vel+2"], y_teste["vel+2"]), 2)
    predissao["Vel+3"] = np.round(SVR(x_treino, x_teste, y_treino["vel+3"], y_teste["vel+3"]), 2)
    predissao.set_index(tabela_teste.index, inplace = True, drop = False)
    return predissao
def PreverLSVR(tabela_treino, tabela_teste) :
    tabela_treino_filtrada, tabela_teste_filtrada = Impute(tabela_treino, tabela_teste)
    predissao = pd.DataFrame()
    x_treino, y_treino, x_teste, y_teste = XYArraySplit(tabela_treino_filtrada, tabela_teste_filtrada)
    predissao['velAtual'] = y_teste['velAtual']
    predissao["Vel+1"] = np.round(LSVR(x_treino, x_teste, y_treino["vel+1"], y_teste["vel+1"]), 2)
    predissao["Vel+2"] = np.round(LSVR(x_treino, x_teste, y_treino["vel+2"], y_teste["vel+2"]), 2)
    predissao["Vel+3"] = np.round(LSVR(x_treino, x_teste, y_treino["vel+3"], y_teste["vel+3"]), 2)
    predissao.set_index(tabela_teste.index, inplace = True, drop = False)
    return predissao
def PreverPSVR(tabela_treino, tabela_teste, grau) :
    tabela_treino_filtrada, tabela_teste_filtrada = Impute(tabela_treino, tabela_teste)
    predissao = pd.DataFrame()
    x_treino, y_treino, x_teste, y_teste = XYArraySplit(tabela_treino_filtrada, tabela_teste_filtrada)
    predissao['velAtual'] = y_teste['velAtual']
    predissao["Vel+1"] = np.round(PSVR(x_treino, x_teste, y_treino["vel+1"], y_teste["vel+1"], grau), 2)
    predissao["Vel+2"] = np.round(PSVR(x_treino, x_teste, y_treino["vel+2"], y_teste["vel+2"], grau), 2)
    predissao["Vel+3"] = np.round(PSVR(x_treino, x_teste, y_treino["vel+3"], y_teste["vel+3"], grau), 2)
    predissao.set_index(tabela_teste.index, inplace = True, drop = False)
    return predissao
def PreverPRegressao(tabela_treino, tabela_teste, grau) :
    tabela_treino_filtrada, tabela_teste_filtrada = Impute(tabela_treino, tabela_teste)
    predissao = pd.DataFrame()
    x_treino, y_treino, x_teste, y_teste = XYArraySplit(tabela_treino_filtrada, tabela_teste_filtrada)
    predissao['velAtual'] = y_teste['velAtual']
    predissao["Vel+1"] = np.round(PRegressao(x_treino, x_teste, y_treino["vel+1"], y_teste["vel+1"], grau), 2)
    predissao["Vel+2"] = np.round(PRegressao(x_treino, x_teste, y_treino["vel+2"], y_teste["vel+2"], grau), 2)
    predissao["Vel+3"] = np.round(PRegressao(x_treino, x_teste, y_treino["vel+3"], y_teste["vel+3"], grau), 2)
    predissao.set_index(tabela_teste.index, inplace = True, drop = False)
    return predissao
faixa = 3
dia = 6
intervalo = 10

tabela_treino = VelocidadeFaixa(dados_marco, dia, faixa, intervalo)
tabela_teste = VelocidadeFaixa(dados_abril, dia, faixa, intervalo)
grau = 1
predissao_abril = PreverPRegressao(tabela_treino, tabela_teste, grau)
iplot(Grafico(predissao_abril, 4))
predissao_abril.head()
predissao_abril_ridge = PreverRidge(tabela_treino, tabela_teste)
iplot(Grafico(predissao_abril_ridge, 1))
predissao_abril_ridge.head()
predissao_abril_lasso = PreverLasso(tabela_treino, tabela_teste)
iplot(Grafico(predissao_abril_lasso, 1))
predissao_abril_lasso.head()
predissao_abril_enet = PreverENet(tabela_treino, tabela_teste)
iplot(Grafico(predissao_abril_enet, 1))
predissao_abril_enet.head()
predissao_abril_lars = PreverLARS(tabela_treino, tabela_teste)
iplot(Grafico(predissao_abril_lars, 1))
predissao_abril_lars.head()
grau = 1
predissao_svr = PreverPSVR(tabela_treino, tabela_teste, grau)
iplot(Grafico(predissao_svr, 1))
predissao_svr.head()
predissao_lsvr = PreverLSVR(tabela_treino, tabela_teste)
iplot(Grafico(predissao_lsvr, 1))
predissao_lsvr.head()
dados_compara = pd.read_csv(
    "../input/2290_01042018_30042018_SPEED_FORECASTS.csv",
    delimiter=",",
        dtype = {
            "intervalo":float,
            "ValorHistorico":float,
            "ValorAtual":float,
            "PrevisaoDez":float,
            "PrevisaoVinte":float,
            "PrevisaoTrinta":float
        },
    parse_dates=["dataHoraConsulta"])
dados_compara.set_index("dataHoraConsulta", inplace=True, drop=True)
tab_10 = pd.DataFrame()
tab_10['ValorAtual'] = dados_compara['ValorAtual']
tab_10['PrevisaoDez'] = dados_compara['PrevisaoDez']
tab_10['PrevisaoVinte'] = dados_compara['PrevisaoVinte']
tab_10['PrevisaoTrinta'] = dados_compara['PrevisaoTrinta']
tab_10 = tab_10[tab_10['PrevisaoDez'] > 0]
tab_10 = tab_10[tab_10['PrevisaoVinte'] > 0]
tab_10 = tab_10[tab_10['PrevisaoTrinta'] > 0]
print(r2_score(tab_10['ValorAtual'].shift(-1).fillna(np.mean(tab_10["ValorAtual"])), tab_10['PrevisaoDez']))
print(r2_score(tab_10['ValorAtual'].shift(-2).fillna(np.mean(tab_10["ValorAtual"])), tab_10['PrevisaoVinte']))
print(r2_score(tab_10['ValorAtual'].shift(-3).fillna(np.mean(tab_10["ValorAtual"])), tab_10['PrevisaoTrinta']))
iplot(GraficoCompara(dados_compara, 6))