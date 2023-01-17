import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
yf.pdr_override() # ajutes do fix_yahoo_finance para poder usar o yahoo finance novamente no pandas_datareader
import numpy as np

from datetime import date
from dateutil.relativedelta import relativedelta

import matplotlib.pyplot as plt
import matplotlib.cm as cm
%matplotlib inline
plt.style.use('seaborn')

# Conjunto de criptomoedas da carteira
moedas = ["BTC-USD", "ETH-USD", "XRP-USD", "BCH-USD", "LTC-USD", "IOT-USD", 
          "XMR-USD", "DASH-USD"]

fim = date(2018, 11, 28) # Data final
inicio = fim + relativedelta(months = -2) # Data inicial (2 meses antes da final)

dados = pdr.get_data_yahoo(moedas, start=inicio, end=fim)
fechamento = dados['Adj Close']
# Normalizar dados caso exista algum dado da série temporal faltando
fechamento = fechamento[moedas].fillna(method='ffill').fillna(method='bfill')

fechamento.head(20)
mudanca_proporcional = fechamento / fechamento.iloc[0,:]
cores = cm.rainbow(np.linspace(0, 1, len(moedas)))
mudanca_proporcional.fillna(1).plot.line(figsize=(24, 8), grid=True, color=cores)

mudanca_proporcional.head()
carteira_inicial = np.array([1.0/len(moedas) for i in range(len(moedas))])
evolucao = pd.DataFrame(dict(inicial = np.sum(mudanca_proporcional * carteira_inicial, axis=1)))

evolucao.plot.line(figsize=(24, 8), grid=True, color=cores)
carteira_inicial
retorno_percentual = fechamento.pct_change().dropna()
retorno_percentual.plot.line(figsize=(24, 8), grid=True, color=cores)
retorno_percentual.tail(10)
retorno_inicial = np.dot(retorno_percentual.mean(), carteira_inicial)
retorno_inicial
carteira_cov = retorno_percentual.cov()
carteira_cov
risco_inicial = np.sqrt(np.dot(carteira_inicial.T, np.dot(carteira_cov, carteira_inicial)))
risco_inicial
sharpe_inicial = retorno_inicial / risco_inicial
sharpe_inicial
def gerar_distribuicao_carteira(ativos, retorno_percentual, cov, num_simulacoes):
    retornos = []
    volatilidades = []
    sharpes = []
    pesos_portfolio = []

    num_ativos = len(ativos)
    media_variacao = retorno_percentual.mean()
    
    np.random.seed(4765)
    for s in range(0, num_simulacoes):
        pesos = np.random.random(num_ativos)
        pesos /= pesos.sum()
        retorno = np.dot(pesos, media_variacao)
        volatilidade = np.sqrt(np.dot(pesos.T, np.dot(cov, pesos)))
        sharpe = retorno / volatilidade

        retornos.append(retorno)
        volatilidades.append(volatilidade)
        sharpes.append(sharpe)
        pesos_portfolio.append(pesos)

    # um dicionario de Retorno e Risco para cada carteira
    portfolio = {'Retornos': retornos, 'Riscos': volatilidades, 'Sharpes': sharpes}

    # acrescentando a composicao de cada moeda da carteira ao dicionario
    for counter, at in enumerate(ativos):
        portfolio[at + ' comp'] = [peso[counter] for peso in pesos_portfolio]

    # transformando o dicionario gerado em um Dataframe
    df = pd.DataFrame(portfolio)
    column_order = ['Retornos', 'Riscos', 'Sharpes'] + [at + ' comp' for at in ativos]    
    return df[column_order].sort_values(['Retornos']).reset_index().set_index('index')
  
df = gerar_distribuicao_carteira(moedas, retorno_percentual, carteira_cov, 10000)
df.head(20)
df.plot.scatter(x='Riscos', y='Retornos', c='Sharpes', cmap='RdYlBu', edgecolors='black', 
                figsize=(24, 8), grid=True, sharex=False)
plt.xlabel('Risco')
plt.ylabel('Retorno Esperado')
plt.title(u'Simulação Carteiras')
plt.show()
import cvxpy as cp
 
def minimizar_riscos(medias, cov, epv):
    pesos = cp.Variable(len(medias))
    
    risco_esperado = cp.quad_form(pesos, cov)   
    retorno_esperado = pesos.T * medias

    obj = cp.Minimize(risco_esperado)
    cons = [
        epv == retorno_esperado,
        cp.sum(pesos) == 1,
        pesos >= 0
    ]

    prob = cp.Problem(obj, cons)
    prob.solve(solver='ECOS')
    norm_pesos = pesos.value.round(4)
    
    return (norm_pesos, retorno_esperado.value.round(4), 
            cp.sqrt(risco_esperado).value.round(4))

def fronteira_risco_retorno(amostra_retornos, variacao_amostra, amostra_cov):
    media = variacao_amostra.mean()
    riscos = [minimizar_riscos(media, amostra_cov, rtr)[2] for rtr in amostra_retornos]
    return pd.DataFrame({'Retornos': amostra_retornos, 'Riscos': np.array(riscos)})
xs = np.linspace(-0.01025, -0.00430, num = 50)
fronteira = fronteira_risco_retorno(xs, retorno_percentual, carteira_cov)

ax = df.plot.scatter(x='Riscos', y='Retornos', c='Sharpes', cmap='RdYlBu', edgecolors='black', 
                     figsize=(24, 8), grid=True, sharex=False)
fronteira.plot.line(x='Riscos', y='Retornos', color='DarkGreen', ax=ax,
                    label='Fronteira Eficiente')
plt.xlabel('Risco')
plt.ylabel('Retorno Esperado')
plt.title(u'Simulação Carteiras')
plt.show()

fronteira.head()
def maximizar_retornos(medias, cov, risco_maximo):
    top_risco = risco_maximo**2
    pesos = cp.Variable(len(medias))
    
    risco_esperado = cp.quad_form(pesos, cov)
    retorno_esperado = pesos.T * medias
    
    obj = cp.Maximize(retorno_esperado)
    cons = [
        risco_esperado <= top_risco,
        cp.sum(pesos) == 1,
        pesos >= 0
    ]
    
    prob = cp.Problem(obj, cons)
    prob.solve(solver='ECOS')
    
    return (pesos.value.round(4), retorno_esperado.value, 
            np.sqrt(risco_esperado.value))
    
def fronteira_retorno_risco(amostra_riscos, variacao_amostra, amostra_cov):
    media = variacao_amostra.mean()
    retornos = [maximizar_retornos(media, amostra_cov, rsc)[1] for rsc in amostra_riscos]
    return pd.DataFrame({'Retornos': np.array(retornos), 'Riscos': amostra_riscos})

xs = np.linspace(0.03562, 0.04670, num = 50)
fronteira = fronteira_retorno_risco(xs, retorno_percentual, carteira_cov)
fronteira.head()

ax = df.plot.scatter(x='Riscos', y='Retornos', c='Sharpes', cmap='RdYlBu', edgecolors='black',
                     figsize=(26, 8), grid=True, sharex=False)
fronteira.plot.line(x='Riscos', y='Retornos', color='DarkGreen', ax = ax,
                    label='Fronteira Eficiente')
plt.xlabel('Risco')
plt.ylabel('Retorno Esperado')
plt.title(u'Simulação Carteiras')
plt.show()
carteira_otimizada, _, _ = maximizar_retornos(retorno_percentual.mean(), carteira_cov,
                                              risco_inicial)
carteira_arriscada, _, _ = maximizar_retornos(retorno_percentual.mean(), carteira_cov,
                                              risco_inicial + 0.05)
carteira_segura, _, _ = maximizar_retornos(retorno_percentual.mean(), carteira_cov,
                                           risco_inicial - 0.008)

distribuicao = pd.DataFrame(dict(inicial = carteira_inicial,
                                 otimizada = carteira_otimizada,
                                 arriscada = carteira_arriscada,
                                 segura = carteira_segura),
                            index = moedas)
distribuicao.plot.pie(subplots=True, figsize=(20, 4), autopct='%1.0f%%', 
                      pctdistance=1.1, labeldistance=1.2, legend=False,
                      colors=cores)

evolucao = pd.DataFrame(dict(
    inicial = np.sum(mudanca_proporcional * carteira_inicial, axis=1),
    otimizada = np.sum(mudanca_proporcional * carteira_otimizada, axis=1),
    arriscada = np.sum(mudanca_proporcional * carteira_arriscada, axis=1),
    segura = np.sum(mudanca_proporcional * carteira_segura, axis=1)))

evolucao.plot.line(figsize=(24, 8), grid=True)
evolucao.tail()