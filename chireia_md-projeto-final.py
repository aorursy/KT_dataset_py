# Bibliotecas Utilizadas



# Python Data Analysis Library - Utilizada para trabalhar com a estrutura dos dados e todo processo de ETL

import pandas as pd



# Matplotlib - Utilizada para plot dos gráficos 2D

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (11,7) # Define o tamanho e estilo dos gráficos

plt.style.use('seaborn')



# SKLearning - Utilizada para treinar e testar a regressão linear executada no projeto

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

from sklearn.linear_model import LinearRegression
original_gas_prices_df = pd.read_csv('/kaggle/input/gas-prices-in-brazil/2004-2019.tsv', sep='\t')

original_gas_prices_df.drop(columns='Unnamed: 0', inplace=True)

original_gas_prices_df.sample(5)
# Adicionando tipagem às colunas que vamos utilizar posteriormente



# Algumas colunas possuem '-' como valor, causando erro ao converter o tipo, por isso errors='coerse' seta como NaN

original_gas_prices_df['DATA FINAL'] = pd.to_datetime(original_gas_prices_df['DATA FINAL'])

original_gas_prices_df['PREÇO MÉDIO DISTRIBUIÇÃO'] = pd.to_numeric(original_gas_prices_df['PREÇO MÉDIO DISTRIBUIÇÃO'], errors='coerce')

original_gas_prices_df['PREÇO MÉDIO REVENDA'] = pd.to_numeric(original_gas_prices_df['PREÇO MÉDIO REVENDA'], errors='coerce')



# Seta Data final da pesquisa como indíce

original_gas_prices_df.set_index(['DATA FINAL'], inplace=True)



# Definindo quais colunas e tuplas vamos utilizar

used_columns = ['DATA','PRODUTO','ESTADO','REGIÃO','PREÇO MÉDIO DISTRIBUIÇÃO','PREÇO MÉDIO REVENDA']

used_products = ['ETANOL HIDRATADO','GASOLINA COMUM','ÓLEO DIESEL']



# GroupBy por PRODUTO, ESTADO e REGIÃO

# Agrupa também as datas pelo mês, fazendo a média do valor

monthly_df = original_gas_prices_df.groupby(['PRODUTO', 'ESTADO', 'REGIÃO']).resample('M').mean().reset_index()

# Cria a coluna DATA removendo os dias

monthly_df['DATA'] = monthly_df['DATA FINAL'].dt.to_period('M')



# Cria o dataframe final de preço de combustível, já filtrando apenas com as colunas e tuplas relevantes

gas_prices_df = monthly_df[used_columns]

gas_prices_df = gas_prices_df[gas_prices_df.PRODUTO.isin(used_products)]

# Remove tuplas com dados nulos

gas_prices_df = gas_prices_df.dropna(how='any')



# Cria colunas de variações entre a distribuição e revenda, valor e porcentagem

gas_prices_df['VARIAÇÃO DIST X REV'] = gas_prices_df['PREÇO MÉDIO REVENDA'] - gas_prices_df['PREÇO MÉDIO DISTRIBUIÇÃO']

gas_prices_df['VARIAÇÃO PERCENTUAL DIST X REV'] = gas_prices_df['VARIAÇÃO DIST X REV'] /  gas_prices_df['PREÇO MÉDIO DISTRIBUIÇÃO'] * 100



gas_prices_df.set_index(['DATA', 'PRODUTO', 'ESTADO', 'REGIÃO'], inplace=True)

gas_prices_df.sample(n=5)
original_inflation_rate_df = pd.read_csv('/kaggle/input/inflation-rate-in-brazil-1984-2019/inflation-rate-brazil.csv')

original_inflation_rate_df.sample(n=5)
# Tipa corretamente a coluna de data e ignora os dias

original_inflation_rate_df['date'] = pd.to_datetime(original_inflation_rate_df['date']).dt.to_period('M')

# Define data como index da coluna para o merge posterior

inflation_rate_df = original_inflation_rate_df.set_index('date')['2004':'2019']

inflation_rate_df.sample(n=5)
# Nossa DW é o dataframe main_df



# Copia os valores do dataframe resultante do ETL de preço

main_df = gas_prices_df.copy()

# Reseta o index para um valor sequencial e depois seta a coluna DATA como novo index

main_df.reset_index(inplace=True)

main_df.set_index('DATA', inplace=True)





main_df['INFLAÇÃO ANUAL'] = inflation_rate_df['annual_accumulation']  # INFLAÇÃO ANUAL ACUMULADA NOS ÚLTIMOS 12 MESES

main_df['INFLAÇÃO ABSOLUTA'] = inflation_rate_df['absolute_index']  # INFLAÇÃO ANUAL ACUMULADA NOS ÚLTIMOS 12 MESES



main_df = main_df.to_timestamp()

main_df = main_df['2006':]

main_df.sample(5)
# Função que agrupa os dados, define os eixos e plota a variação entre o preço de revenda e distribuição dos combustíveis ao longo do tempo

def plot_prices_correlation(produto):

    grp = main_df[main_df.PRODUTO == produto].groupby(['DATA']).mean()

    fig, ax = plt.subplots()



    grp.plot(y='PREÇO MÉDIO REVENDA', label='Revenda', ax=ax)

    grp.plot(y='PREÇO MÉDIO DISTRIBUIÇÃO', label='Distribuição', ax=ax)

    fig.suptitle('Preços de Distribuição e Revenda - ' + produto)

    ax.set_xlabel('Data')

    ax.set_ylabel('Preço - R$/l')

    ax.grid(True)

    

    ax_var = ax.twinx()



    grp.plot(y='VARIAÇÃO PERCENTUAL DIST X REV', label='Variação - Distribuição', ax=ax_var, c='#c44e52')

    ax_var.set_ylabel('% - Variação entre Distribuição e Revenda', color='#c44e52')

    ax_var.get_legend().remove()

    ax_var.grid(False)

    plt.show()



plot_prices_correlation('GASOLINA COMUM')

plot_prices_correlation('ETANOL HIDRATADO')

plot_prices_correlation('ÓLEO DIESEL')

# Função que agrupa os dados, define os eixos e plota a variação do preço por região ao longo do tempo

def plot_avg_region(produto):

    avg_per_region_df = main_df['2009':].groupby(['PRODUTO', 'REGIÃO', 'DATA']).mean()

    avg_per_region_for_product_df = avg_per_region_df.iloc[avg_per_region_df.index.get_level_values('PRODUTO') == produto]

    

    fig, ax = plt.subplots()

    fig.suptitle('Preço por Região - ' + produto)

    

    for key, grp in avg_per_region_for_product_df.groupby('REGIÃO'):

        grp_as_timeseries = grp.reset_index().set_index('DATA')

        grp_as_timeseries.plot(y='PREÇO MÉDIO REVENDA', label=key, ax=ax)



        ax.set_xlabel('Data')

        ax.set_ylabel('Preço - R$/l')



    plt.grid(True)

    plt.show()



plot_avg_region('GASOLINA COMUM')

plot_avg_region('ETANOL HIDRATADO')

plot_avg_region('ÓLEO DIESEL')
# Função que agrupa os dados, define os eixos e plota a variação da inflação e preços a nível mensal ao longo do tempo

def plot_inflation_correlation(produto):

    annual_price_change_df = main_df[main_df.PRODUTO == produto].groupby(['DATA']).mean()

    annual_price_change_df['VARIAÇÃO'] = annual_price_change_df['PREÇO MÉDIO REVENDA'].pct_change()

    annual_price_change_df['VARIAÇÃO 12 MESES'] = annual_price_change_df['VARIAÇÃO'].rolling(min_periods=12, window=12).sum() * 100

    annual_price_change_df.tail()

    

    # ploting

    fig, ax_gas = plt.subplots()

    annual_price_change_df.plot(y='VARIAÇÃO 12 MESES', c='#4c72b0', ax=ax_gas)

    fig.suptitle('Variação da Inflação e Preço Acumulado - ' + produto)

    ax_gas.set_xlabel('Data de pesquisa')

    ax_gas.set_ylabel('% - Variação Acumulativa - 12 meses', color='#4c72b0')

    ax_gas.get_legend().remove()

    ax_gas.grid(True)

    

    # plot_inflation

    ax_inflation = ax_gas.twinx()

    

    annual_price_change_df.plot(y='INFLAÇÃO ANUAL', ax=ax_inflation, c='#55a868')

    ax_inflation.set_ylabel('% - Inflação Acumulada - 12 meses', color='#55a868')

    ax_inflation.get_legend().remove()

    ax_inflation.grid(False)

    plt.show()



plot_inflation_correlation('GASOLINA COMUM')

plot_inflation_correlation('ETANOL HIDRATADO')

plot_inflation_correlation('ÓLEO DIESEL')

# Através do índice de inflação absoluta, podemos realizar uma regressão linear.

models = {}

def regression_for_product(produto):

    # Cria Dataframe necessário para regressão

    reg_df = main_df['2009':].groupby(['PRODUTO', 'DATA']).mean()

    reg_df = reg_df.iloc[reg_df.index.get_level_values('PRODUTO') == produto].groupby('DATA').mean()[['PREÇO MÉDIO REVENDA', 'INFLAÇÃO ABSOLUTA']]



    # Separa eixos da regressão

    X = X = reg_df[['INFLAÇÃO ABSOLUTA']]

    Y = reg_df['PREÇO MÉDIO REVENDA'].values



    # Divide dados entre treinamento e teste

    x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.20)



    # Cria model de regressão e faz o treinamento

    lr=LinearRegression()

    lr.fit(x_train,y_train)

    models[produto] = lr

    predicted_values = []

    for i in range(0, len(y_test)):

        predicted_values.append(lr.predict(x_test.iloc[[i],:])[0])



    predicted_df = pd.DataFrame({'Inflação Absoluta':x_test['INFLAÇÃO ABSOLUTA'].values , 'Valor Real':y_test, 'Valor Predito':predicted_values})

    fig, ax = plt.subplots()

    fig.suptitle('Regressão - ' + produto)

    ax.set_ylabel('Preço Revenda - R$/l')

    predicted_df.sort_values(by=['Inflação Absoluta']).set_index('Inflação Absoluta').plot(ax=ax)

    plt.show()

    

regression_for_product('GASOLINA COMUM')

regression_for_product('ETANOL HIDRATADO')

regression_for_product('ÓLEO DIESEL')
# Com um modelo de regressão linear podemos fazer predizer quais serão os preços futuros.

def predict_price(produto):

    rates = {}

    rates['2019-08-01'] = inflation_rate_df['absolute_index'].iloc[-1]

    for i in range(2020, 2029):

        last_year = str(i - 1)+'-08-01'

        curr_year = str(i)+'-08-01'

        rates[curr_year]  = (rates[last_year] * 0.04) + rates[last_year]



    rates_df = pd.DataFrame(list(rates.items()), columns = ['DATA', 'INFLAÇÂO ABSOLUTA'])



    rates_df['DATA'] = pd.to_datetime(rates_df['DATA'])

    rates_df.set_index('DATA', inplace=True)

    predictions = {}



    rates_df['PREVISÃO'] = models[produto].predict(rates_df)



    return rates_df
# Previsão para Gasolina

predict_price('GASOLINA COMUM')
# Previsão para Etanol

predict_price('ETANOL HIDRATADO')
# Previsão para Diesel

predict_price('ÓLEO DIESEL')










































gas = main_df[main_df.PRODUTO == 'GASOLINA COMUM']

eta = main_df[main_df.PRODUTO == 'ETANOL HIDRATADO']

gas = gas.groupby(['DATA']).mean()[['PREÇO MÉDIO REVENDA']]

eta = eta.groupby(['DATA']).mean()[['PREÇO MÉDIO REVENDA']]

group = gas.copy()

group['PREÇO MÉDIO REVENDA ETA'] = eta[['PREÇO MÉDIO REVENDA']]

group['RAZAO'] = group['PREÇO MÉDIO REVENDA ETA'] / group['PREÇO MÉDIO REVENDA'] * 100

group.sample(30)



fig, ax = plt.subplots()

ax.set_ylim(60,90)

group.plot(y='RAZAO', label='Razao', ax=ax)

ax.set_xlabel('Data')

ax.set_ylabel('Preço - R$/l')

ax.grid(True)



plt.show()