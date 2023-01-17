# Pacotes utilizados:

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import datetime

import warnings

warnings.filterwarnings('ignore')



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error
# Leitura dos arquivos:

covid_br = pd.read_csv('../input/covid19-br/brazil_covid19_cities.csv', parse_dates = ['date'])

cities_pop = pd.read_csv('../input/covid19-br/brazil_population_2019.csv')
# Verificando o formato das variáveis:

covid_br.dtypes
# Observando as primeiras linhas dos datasets:

covid_br.head()
# Alterando a nomeclatura de cada estado para o seu respectivo 'UF':

estado_uf = {'Acre': 'AC', 'Alagoas': 'AL', 'Amapá': 'AP', 'Amazonas': 'AM', 'Bahia': 'BA', 'Ceará': 'CE',

              'Distrito Federal': 'DF', 'Espírito Santo': 'ES', 'Goiás': 'GO', 'Maranhão': 'MA', 'Minas Gerais': 'MG',

              'Mato Grosso do Sul': 'MS', 'Mato Grosso': 'MT', 'Pará': 'PA', 'Paraíba': 'PB', 'Pernambuco': 'PE',

              'Piauí': 'PI', 'Paraná': 'PR', 'Rio de Janeiro': 'RJ', 'Rio Grande do Norte': 'RN', 'Rondônia': 'RO',

              'Roraima': 'RR', 'Rio Grande do Sul': 'RS', 'Santa Catarina': 'SC', 'Sergipe': 'SE', 'São Paulo': 'SP',

              'Tocantins': 'TO'}



covid_br = covid_br.replace({'state': estado_uf})



# Criando a variável 'city', concatenando a cidade e o 'UF' do seu respectivo estado de referência:

covid_br['city'] = covid_br['name'] + ' - ' + covid_br['state']



# Selecionando apenas as capitais de cada estado:

capitais = ['Aracaju - SE', 'Belém - PA', 'Belo Horizonte - MG', 'Boa Vista - RR', 'Brasília - DF', 'Campo Grande - MS',

            'Cuiabá - MT', 'Curitiba - PR', 'Florianópolis - SC', 'Fortaleza - CE', 'Goiânia - GO', 'João Pessoa - PB',

            'Macapá - AP', 'Maceió - AL', 'Manaus - AM', 'Natal - RN', 'Palmas - TO', 'Porto Alegre - RS',

            'Porto Velho - RO', 'Recife - PE', 'Rio Branco - AC', 'Rio de Janeiro - RJ', 'Salvador - BA', 'São Luís - MA',

            'São Paulo - SP', 'Teresina - PI', 'Vitória - ES']



covid_cap_br = covid_br[covid_br.city.isin(capitais)]



# Deletando as colunas 'state' e 'name' do dataset:

covid_cap_br = covid_cap_br.drop(columns = ['state', 'name'])



covid_cap_br.head()
# Incorporando o número populacional ao dataset com informações sobre a Covid-19:

covid_cap_br = covid_cap_br.rename(columns = {'code' : 'city_code'})

cities_pop = cities_pop[['city_code', 'population']]

covid_cap_br = pd.merge(covid_cap_br, cities_pop, on = 'city_code')

covid_cap_br.head()
covid_atual = covid_cap_br[covid_cap_br['date'] == max(covid_cap_br['date'])]



covid_atual['cases_rate'] = round(((covid_atual['cases'] / covid_atual['population'])*100000), 1)



covid_atual['deaths_rate'] = round(((covid_atual['deaths'] / covid_atual['population'])*100000), 1)



covid_atual['deaths_prop'] = round(((covid_atual['deaths'] / covid_atual['cases'])*100), 1)



covid_atual.head()
# Ordenando o dataset por número de mortes:

covid_deaths = covid_atual.sort_values(by = ['deaths'], ascending = False)



# Ordenando o dataset por número de mortes por 100 mil habitantes:

covid_deaths_rate = covid_atual.sort_values(by = ['deaths_rate'], ascending = False)



# Ordenando o dataset por número de casos confirmados por 100 mil habitantes:

covid_cases_rate = covid_atual.sort_values(by = ['cases_rate'], ascending = False)



# Criando variáveis que indicam a primeira e última datas apresentadas no dataset:

ult_date = max(covid_cap_br['date']).strftime('%d/%m/%Y')

min_date = min(covid_cap_br['date']).strftime('%d/%m/%Y')



# Criando o plot:

fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (16, 5))



sns.barplot(x = 'city', y = 'deaths', data = covid_deaths[0:5], ci = None,

            palette = 'Blues', ax = ax[0]).set_title('MORTES TOTAIS ATÉ '+ ult_date)

ax[0].set_xlabel('Cidade')

ax[0].set_ylabel('Número de mortos')



sns.barplot(x = 'city', y = 'deaths_rate', data = covid_deaths_rate[0:5], ci = None,

            palette = 'Blues', ax = ax[1]).set_title('MORTES POR 100 MIL HABITANTES ATÉ '+ ult_date)

ax[1].set_xlabel('Cidade')

ax[1].set_ylabel('Número de mortos por 100 mil hab.')



fig.show()
# Ordenando o dataset por percentagem de mortos entre os casos confirmados:

covid_deaths_prop = covid_atual.sort_values(by = ['deaths_prop'], ascending = False)



# Criando o plot:

fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (16, 5))



sns.barplot(x = 'city', y = 'deaths_rate', data = covid_deaths_rate[0:5], ci = None,

            palette = 'Blues', ax = ax[0]).set_title('MORTES POR 100 MIL HABITANTES ATÉ '+ ult_date)

ax[0].set_xlabel('Cidade')

ax[0].set_ylabel('Número de mortos por 100 mil hab.')



sns.barplot(x = 'city', y = 'deaths_prop', data = covid_deaths_prop[0:5], ci = None,

            palette = 'Blues', ax = ax[1]).set_title('PERCENTAGEM DE MORTOS ENTRE CASOS CONFIRMADOS ATÉ '+ ult_date)

ax[1].set_xlabel('Cidade')

ax[1].set_ylabel('Percentagem de mortos entre confirmados')



fig.show()
# Criando o plot:

fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (17, 10))



sns.barplot(x = 'city', y = 'deaths', data = covid_deaths[22:], ci = None,

            palette = 'Blues', ax = ax[0,0]).set_title('MORTES TOTAIS ATÉ '+ ult_date)

ax[0,0].set_xlabel('Cidade')

ax[0,0].set_ylabel('Número de mortos')



sns.barplot(x = 'city', y = 'deaths_rate', data = covid_deaths_rate[22:], ci = None,

            palette = 'Blues', ax = ax[0,1]).set_title('MORTES POR 100 MIL HABITANTES ATÉ '+ ult_date)

ax[0,1].set_xlabel('Cidade')

ax[0,1].set_ylabel('Número de mortos por 100 mil hab.')



sns.barplot(x = 'city', y = 'deaths_prop', data = covid_deaths_prop[22:], ci = None,

            palette = 'Blues', ax = ax[1,0]).set_title('% DE MORTOS ATÉ '+ ult_date)

ax[1,0].set_xlabel('Cidade')

ax[1,0].set_ylabel('Percentagem de mortos entre confirmados')



sns.barplot(x = 'city', y = 'cases_rate', data = covid_cases_rate[22:], ci = None,

            palette = 'Blues', ax = ax[1,1]).set_title('CASOS CONFIRMADOS POR 100 MIL HABITANTES ATÉ '+ ult_date)

ax[1,1].set_xlabel('Cidade')

ax[1,1].set_ylabel('Número de casos confirmados por 100 mil hab.')



fig.show()
# Criando uma variável numérica que represente a contagem dos dias a partir da primeira data de referência do dataset:

covid_cap_br['day_count'] = (covid_cap_br['date'] - min(covid_cap_br['date']))

covid_cap_br['day_count'] = pd.to_numeric(covid_cap_br['day_count'].dt.days, downcast='integer')
# Criando uma função que gera um plot que apresenta a evolução do número de mortes dia a dia:

def deaths_per_day(city_name):

    """Cria um gráfico apresentando as mortes dia a dia.

    

    Args:

    city_name: string indicando a cidade a ser filtrada. A convenção no dataset é 'Cidade - UF'.

    """

    filt_city = covid_cap_br[covid_cap_br['city'] == city_name]

    

    filt_city = filt_city.reset_index(drop = True)

    

    filt_city['deaths_per_day'] = ''

    for i in range(0, len(filt_city.index)):

        if i == 0:

            filt_city['deaths_per_day'][i] = filt_city['deaths'][i]

        else:

            filt_city['deaths_per_day'][i] = filt_city['deaths'][i] - filt_city['deaths'][(i-1)]

    

    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (16, 5))

    

    sns.barplot(x = 'day_count', y = 'deaths_per_day', data = filt_city, palette = "autumn_r").set(

        xlabel = 'Contagem dos dias desde ' + min_date,

        ylabel = 'Mortes diárias',

        title = 'Evolução diária do número de mortes em ' + city_name)

    fig.show()
deaths_per_day('São Paulo - SP')
deaths_per_day('Rio de Janeiro - RJ')
deaths_per_day('Belém - PA')
deaths_per_day('Porto Alegre - RS')
# Criando função para filtrar cidade e colunas desejadas:

def city_filter(city_name):

    """Filtra o dataset já alterado - neste ponto, covid_cap_br - para a capital de interesse.

    

    Args:

    city_name: string indicando a cidade a ser filtrada. A convenção no dataset é 'Cidade - UF'.

    

    Returns:

    filt_data: dataset contendo os dados referentes à cidade de interesse.

    """

    filt_data = covid_cap_br[covid_cap_br['city'] == city_name]

    filt_data = filt_data[['day_count', 'deaths']]

    return filt_data



# Criando função para plotar gráfico com a evolução do número acumulado de mortes semana a semana:

def deaths_evolution(city_data, city_name):

    """Cria um gráfico de barras com a contagem acumulada das mortes, espaçadas de 7 em 7 dias.

    

    Args:

    city_data: dataset filtrado pela função city_filter();

    city_name: string com o nome da cidade a ser apresentado no gráfico.

    """

    sns.barplot(x = 'day_count', y = 'deaths', data = city_data[city_data['day_count'] % 7 == 0], palette = "Blues").set(

        xlabel = 'Contagem de dias a partir de ' + min_date,

        ylabel = 'Número de mortes',

        title = 'Evolução do número de mortes na cidade de ' + city_name)
# Criando função para desenvolver, treinar e avaliar o modelo:

def covid_poly_model(city_data, n_degree, seed):

    """Transforma os dados para uma regressão polinomial, cria o modelo de regressão linear e o avalia, juntamente com a

    apresentação de um gráfico que ilustra os dados previstos pelo modelo em relação aos dados originais.

    

    Args:

    city_data: dataset filtrado pela função city_filter();

    n_degree: número que representa o grau da transformação linear a ser aplicada;

    seed: número que permite a reprodução da separação dos dados de treino e de teste para posterior comparação.

    

    Returns:

    poly_regr: modelo de transformação polinomial a ser aplicado posteriormente, permitindo ao modelo fazer previsões;

    model: modelo de regressão linear para fazer as previsões.

    """

    # Separando as variáveis:

    X = city_data[['day_count']]

    Y = city_data[['deaths']]

    

    # Aplicando a transformação polinomial aos dados:

    poly_regr = PolynomialFeatures(degree = n_degree)

    X_poly = poly_regr.fit_transform(X)

    

    # Criando dados de treinos e de teste:

    X_poly_train, X_poly_test, Y_train, Y_test = train_test_split(X_poly, Y,

                                                                 test_size = 0.2,

                                                                 random_state = seed)

    

    # Criando o modelo:

    model = LinearRegression()

    

    # Treinando o modelo:

    model.fit(X_poly_train, Y_train)

    

    # Fazendo as previsões para os dados de teste:

    Y_pred = model.predict(X_poly_test)

    

    # Avaliando o modelo pela métrica MAE (mean absolut error):

    mae = round(mean_absolute_error(Y_test, Y_pred))

    

    sns.scatterplot(x = 'day_count', y = 'deaths', data = city_data).set(

        xlabel = 'Contagem dos dias a partir de ' + min_date,

        ylabel = 'Número de mortos',

        title = 'Evolução Mortes x Regressão Polinomial de Grau ' + str(n_degree))

    plt.plot(X, model.predict(X_poly), color = 'r')

    plt.show()

    

    print('O erro médio absoluto da previsão do número de mortos para a regressão polinomial de grau ' +

          str(n_degree) + ' é de: ' + str(mae) + '.')

 

    return poly_regr, model
# Filtrando os dados para a cidade de São Paulo:

covid_sp = city_filter('São Paulo - SP')



# Avaliando o crescimento no número de mortes na capital paulista:

deaths_evolution(covid_sp, 'São Paulo - SP')
# Criação e avaliação do modelo de regressão para a cidade de São Paulo:

sp_poly_regr_5, sp_model_5 = covid_poly_model(covid_sp, n_degree = 5, seed = 200)



sp_poly_regr_6, sp_model_6 = covid_poly_model(covid_sp, n_degree = 6, seed = 200)



sp_poly_regr_7, sp_model_7 = covid_poly_model(covid_sp, n_degree = 7, seed = 200)
sp_poly_regr = sp_poly_regr_5

sp_model = sp_model_5
# Filtrando os dados para a cidade do Rio de Janeiro:

covid_rj = city_filter('Rio de Janeiro - RJ')



# Avaliando o crescimento no número de mortes na capital carioca:

deaths_evolution(covid_rj, 'Rio de Janeiro - RJ')
# Criação e avaliação do modelo de regressão para a cidade do Rio de Janeiro:

rj_poly_regr_4, rj_model_4 = covid_poly_model(covid_rj, n_degree = 4, seed = 201)



rj_poly_regr_5, rj_model_5 = covid_poly_model(covid_rj, n_degree = 5, seed = 201)



rj_poly_regr_6, rj_model_6 = covid_poly_model(covid_rj, n_degree = 6, seed = 201)
rj_poly_regr = rj_poly_regr_5

rj_model = rj_model_5
# Filtrando os dados para a cidade de Belém:

covid_bl = city_filter('Belém - PA')



# Avaliando o crescimento no número de mortes na capital paraense:

deaths_evolution(covid_bl, 'Belém - PA')
# Criação e avaliação do modelo de regressão para a cidade de Belém:

bl_poly_regr_4, bl_model_4 = covid_poly_model(covid_bl, n_degree = 4, seed = 202)



bl_poly_regr_5, bl_model_5 = covid_poly_model(covid_bl, n_degree = 5, seed = 202)



bl_poly_regr_6, bl_model_6 = covid_poly_model(covid_bl, n_degree = 6, seed = 202)
bl_poly_regr = bl_poly_regr_4

bl_model = bl_model_4
# Filtrando os dados para a cidade de Porto Alegre:

covid_pa = city_filter('Porto Alegre - RS')



# Avaliando o crescimento no número de mortes na capital gaúcha:

deaths_evolution(covid_pa, 'Porto Alegre - RS')
# Criação e avaliação do modelo de regressão para a cidade de Porto Alegre:

pa_poly_regr_3, pa_model_3 = covid_poly_model(covid_pa, n_degree = 3, seed = 203)



pa_poly_regr_4, pa_model_4 = covid_poly_model(covid_pa, n_degree = 4, seed = 203)



pa_poly_regr_5, pa_model_5 = covid_poly_model(covid_pa, n_degree = 5, seed = 203)
pa_poly_regr = pa_poly_regr_4

pa_model = pa_model_4
# Carregando o arquivo com as datas futuras de referência e aplicando as transformações necessárias:

pred_df = pd.read_csv('../input/covid19-br/covid_predict.csv', parse_dates = ['date'])

pred_df['day_count'] = (pred_df['date'] - min(covid_cap_br['date']))

pred_df['day_count']
# Transformando a contagem de dias em números inteiros:

pred_df['day_count'] = pd.to_numeric(pred_df['day_count'].dt.days, downcast='integer')
# Criando uma função para facilitar a aplicação dos modelos criados anteriormente:

def make_prediction(model, poly_regr, cidade):

    """Utiliza os modelos de transformação polinomial e regressão linear criados anteriormente para fazer previsões.

    

    Args:

    model: modelo de regressão linear criado pela função covid_poly_model();

    poly_regr: modelo de transformação polinomial criado pela função covid_poly_model();

    cidade: string da cidade que é objeto da previsão.

    

    Returns:

    retorna duas strings com a previsão do número de mortos total até as duas datas contidas no objeto 'pred_df'.

    """

    Y_predicted = model.predict(poly_regr.fit_transform([[pred_df['day_count'][0]], [pred_df['day_count'][1]]]))

    

    print('Até ' + pred_df['date'][0].strftime('%d/%m/%Y') + ', na cidade de ' + cidade + ', a previsão é de ' +

      str(round(int(Y_predicted[0]))) + ' mortos por Covid-19.')

    

    print('Até ' + pred_df['date'][1].strftime('%d/%m/%Y') + ', na cidade de ' + cidade + ', a previsão é de ' +

      str(round(int(Y_predicted[1]))) + ' mortos por Covid-19.')
# Fazendo as previsões para a cidade de São Paulo:

make_prediction(sp_model, sp_poly_regr, 'São Paulo - SP')
# Fazendo as previsões para a cidade do Rio de Janeiro:

make_prediction(rj_model, rj_poly_regr, 'Rio de Janeiro - RJ')
# Fazendo as previsões para a cidade de Belém:

make_prediction(bl_model, bl_poly_regr, 'Belém - PA')
# Fazendo as previsões para a cidade de Porto Alegre:

make_prediction(pa_model, pa_poly_regr, 'Porto Alegre - RS')