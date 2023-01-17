from IPython.display import Image
Image("../input/figuras/Fro_Eff.png")
Image("../input/figuras/Perfis_Port.png")
import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sco
import os as os
# from matplotlib.pylab import rcParams
# import seaborn as sns
plt.style.use('fivethirtyeight')
# rcParams['figure.figsize'] = 20, 8
np.random.seed(777)
%matplotlib inline
# path = "X:/Historico_Cotacoes/teste/alpha_data/"
nomeArq = '../input/portfolio-de-ativos-da-b3-bovespa/alphacart12018.csv'
#pwd = os.getcwd() # guarda o path corrente
#os.chdir(os.path.dirname(path)) # muda para o path dos arquivos

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
data = pd.read_csv(nomeArq, parse_dates=["datetime"], index_col="datetime",date_parser=dateparse)
data.info()
## Definindo a data de inicio das observacoes, quando todos os ativos possuem dados
start_date = "2013-01-02"
end_date = "2018-12-31"
s = data.loc[start_date:end_date] 
data = s.copy()
df = data.reset_index()
df = df[['datetime', 'CODNEG', 'adj_close']]
df.head()
df.info()
table = df.pivot_table(index = 'datetime', columns = 'CODNEG', values = 'adj_close')
table.columns.name=None
table.head()
plt.figure(figsize=(20, 10))
for c in table.columns.values:
    plt.plot(table.index, table[c], lw=3, alpha=0.8, label=c)
    
legend = plt.legend(loc='upper left', fontsize=14)
plt.setp(legend.get_texts(), color='k')   
# plt.legend(loc='upper left', fontsize=14)
plt.title('Evolução Dária dos Preços de cada Ativo do Portfólio')
plt.ylabel('precos em BRL');
returns = table.pct_change() ## retornos

plt.figure(figsize=(18, 10))
plt.setp(legend.get_texts(), color='k');

for c in returns.columns.values:
    plt.plot(returns.index, returns[c], lw=3, alpha=0.8,label=c)
    
legend = plt.legend(loc='best', fontsize=10)

#plt.legend(loc='lower right', fontsize=10)
plt.title('Série de Retornos Percentuais do Portfólio')
plt.ylabel('retornos diarios')
plt.show();
def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns*weights ) *252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std, returns

def random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate):
    dim_res = len(table.columns) ## cuidado para não renomear table
    results = np.zeros((dim_res,num_portfolios)) ## 
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(dim_res)
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_std_dev, portfolio_return = portfolio_annualised_performance(weights, \
                                              mean_returns, cov_matrix)
        results[0,i] = portfolio_std_dev
        results[1,i] = portfolio_return
        results[2,i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
    return results, weights_record
returns = table.pct_change()
mean_returns = returns.mean()
cov_matrix = returns.cov()
num_portfolios = 25000
risk_free_rate = 0.0345
def display_simulated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate):
    results, weights = random_portfolios(num_portfolios,mean_returns, cov_matrix, risk_free_rate)
    
    max_sharpe_idx = np.argmax(results[2])
    sdp, rp = results[0,max_sharpe_idx], results[1,max_sharpe_idx]
    max_sharpe_allocation = pd.DataFrame(weights[max_sharpe_idx],index=table.columns,columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i*100,2)for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T
    
    min_vol_idx = np.argmin(results[0])
    sdp_min, rp_min = results[0,min_vol_idx], results[1,min_vol_idx]
    min_vol_allocation = pd.DataFrame(weights[min_vol_idx],index=table.columns,columns=['allocation'])
    min_vol_allocation.allocation = [round(i*100,2)for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T
    
    print ("-"*80)
    print ("Alocacao da Carteira pelo Indice de Sharpe Maximo\n")
    print ("Retorno Anualizado:", round(rp,2))
    print ("Volatilidade Anualizada:", round(sdp,2))
    print ("\n")
    print (max_sharpe_allocation)
    print ("-"*80)
    print ("Alocacao da Carteira pela Volatilidade Minima\n")
    print ("Retorno Anualizado:", round(rp_min,2))
    print ("Volatilidade Anualizada:", round(sdp_min,2))
    print ("\n")
    print (min_vol_allocation)

    plt.figure(figsize=(12, 7))
    plt.scatter(results[0,:],results[1,:],c=results[2,:],cmap='YlGnBu', marker='o', s=10, alpha=0.3)
    plt.colorbar()
    plt.scatter(sdp,rp,marker='*',color='r',s=500, label='Indice de Sharpe Maximo')
    plt.scatter(sdp_min,rp_min,marker='*',color='g',s=500, label='Volatilidade Minima')
    
    plt.title('Simulacao de Portfolio Optimizado com base na Fronteira de Eficiencia')
    plt.xlabel('volatilidade anualizada')
    plt.ylabel('retornos anualizados')
    plt.legend(labelspacing=0.8)
display_simulated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate)
constraints = ({'type': 'eq', 'fun': lambda x:np.sum(x) -1})
def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_var, p_ret = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_var

def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = sco.minimize(neg_sharpe_ratio, num_assets * [1./num_assets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result
def portfolio_volatility(weights, mean_returns, cov_matrix):
    return portfolio_annualised_performance(weights, mean_returns, cov_matrix)[0]

def min_variance(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = sco.minimize(portfolio_volatility, num_assets*[1./num_assets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result
def efficient_return(mean_returns, cov_matrix, target):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)

    def portfolio_return(weights):
        return portfolio_annualised_performance(weights, mean_returns, cov_matrix)[1]
    
    constraints = ({'type': 'eq', 'fun': lambda x: portfolio_return(x) - target}, {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0,1) for asset in range(num_assets))
    result = sco.minimize(portfolio_volatility, num_assets*[1./num_assets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def efficient_frontier(mean_returns, cov_matrix, returns_range):
    efficients = []
    for ret in returns_range:
        efficients.append(efficient_return(mean_returns, cov_matrix, ret))
    return efficients
def display_calculated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate):
    results, _ = random_portfolios(num_portfolios,mean_returns, cov_matrix, risk_free_rate)
    max_sharpe = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
    sdp, rp = portfolio_annualised_performance(max_sharpe['x'], mean_returns, cov_matrix)
    max_sharpe_allocation = pd.DataFrame(max_sharpe.x,index=table.columns,columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i*100,2)for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T

    min_vol = min_variance(mean_returns, cov_matrix)
    sdp_min, rp_min = portfolio_annualised_performance(min_vol['x'], mean_returns, cov_matrix)
    min_vol_allocation = pd.DataFrame(min_vol.x,index=table.columns,columns=['allocation'])
    min_vol_allocation.allocation = [round(i*100,2)for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T

    print ("-"*80)
    print ("Alocacao da Carteira pelo Indice de Sharpe Maximo\n")
    print ("Retorno Anualizado:", round(rp,2))
    print ("Volatilidade Anualizada:", round(sdp,2))
    print ("\n")
    print (max_sharpe_allocation)
    print ("-"*80)
    print ("Alocacao da Carteira pela Volatilidade Minima\n")
    print ("Retorno Anualizado:", round(rp_min,2))
    print ("Volatilidade Anualizada:", round(sdp_min,2))
    print ("\n")
    print (min_vol_allocation)
    
    plt.figure(figsize=(12, 7))
    plt.scatter(results[0,:],results[1,:],c=results[2,:],cmap='YlGnBu', marker='o', s=10, alpha=0.3)
    plt.colorbar()
    plt.scatter(sdp,rp,marker='*',color='r',s=500, label='Maximum Sharpe ratio')
    plt.scatter(sdp_min,rp_min,marker='*',color='g',s=500, label='Minimum volatility')
    target = np.linspace(rp_min, 0.28, 50)
    
    efficient_portfolios = efficient_frontier(mean_returns, cov_matrix, target)

    plt.plot([p['fun'] for p in efficient_portfolios], target, linestyle='-.', color='black', label='efficient frontier')
    plt.title('Otimização do Portfolio pelo Método da Fronteira Eficiente (Efficient Frontier)')
    plt.xlabel('volatilidade anualizada')
    plt.ylabel('retorno anualizado')
    plt.legend(labelspacing=0.8)
display_calculated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate)
def display_ef_with_selected(mean_returns, cov_matrix, risk_free_rate):
    max_sharpe = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
    sdp, rp = portfolio_annualised_performance(max_sharpe['x'], mean_returns, cov_matrix)
    max_sharpe_allocation = pd.DataFrame(max_sharpe.x,index=table.columns,columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i*100,2)for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T

    min_vol = min_variance(mean_returns, cov_matrix)
    sdp_min, rp_min = portfolio_annualised_performance(min_vol['x'], mean_returns, cov_matrix)
    min_vol_allocation = pd.DataFrame(min_vol.x,index=table.columns,columns=['allocation'])
    min_vol_allocation.allocation = [round(i*100,2)for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T  

    an_vol = np.std(returns) * np.sqrt(252)
    an_rt = mean_returns * 252  

    print ("-"*80)
    print ("Alocacao da Carteira pelo Indice de Sharpe Maximo\n")
    print ("Retorno Anualizado:", round(rp,2))
    print ("Volatilidade Anualizada (risco):", round(sdp,2))
    print ("\n")
    print ("\n")
    print (max_sharpe_allocation)
    print ("-"*80)
    print ("Alocacao da Carteira pela Volatilidade Minima\n")
    print ("Retorno Anualizado:", round(rp_min,2))
    print ("Volatilidade Anualizada (risco):", round(sdp_min,2))
    print ("\n")
    print (min_vol_allocation)
    print ("-"*80)
    print ("Retornos e Volatilidade das Açoes Individualmente Referenciadas\n")
    
    for i, txt in enumerate(table.columns):
        print (txt,":","retorno anualizado",round(an_rt[i],2),", volatilidade anualizada (risco):",round(an_vol[i],2))
    print ("-"*80)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.scatter(an_vol,an_rt,marker='o',s=200)
    
    for i, txt in enumerate(table.columns):
        ax.annotate(txt, (an_vol[i],an_rt[i]), xytext=(10,0), textcoords='offset points')
    ax.scatter(sdp,rp,marker='*',color='r',s=500, label='Indice de Sharpe Maximo')
    ax.scatter(sdp_min,rp_min,marker='*',color='g',s=500, label='Volatilidade Minima (risco)')

    target = np.linspace(rp_min, 0.30, 50)
    efficient_portfolios = efficient_frontier(mean_returns, cov_matrix, target)

    ax.plot([p['fun'] for p in efficient_portfolios], target, linestyle='-.', color='black', label='efficient frontier')
    ax.set_title('Otimizacao de Carteira com Acoes Individualmente Referenciadas')
    ax.set_xlabel('volatilidade anualizada (risco)')
    ax.set_ylabel('retorno anualizado')
    ax.legend(labelspacing=0.8)
display_ef_with_selected(mean_returns, cov_matrix, risk_free_rate)
