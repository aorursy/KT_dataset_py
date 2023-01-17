import requests

from bs4 import BeautifulSoup

import pandas as pd
# Os dados ultilizados serão extraídos do site worldmeters.

worldometer_url = 'https://www.worldometers.info/coronavirus/'
#Usaremos o comando requests.get para fazer o download da página, e assim ultilizar seus dados.

worldometer_html = requests.get(worldometer_url)
#Status code retornando 200 nos mostra que o download foi concluído com sucesso.

worldometer_html.status_code
#O conteúdo do site é exibido através do comando content.

worldometer_html.content
# Criamos um objeto Soup do código HTML da página.

worldometer_soup = BeautifulSoup(worldometer_html.content, 'html.parser')

worldometer_soup
# Definimos a função e o argumento obrigatório (tabela em HTML)

def worldometertable_to_df(table):

    

    # Separamos o corpo da tabela

    table_tbody =table.find('tbody')

    

    # Separamos na variavel table_data uma lista com todas as linhas da tabela

    table_data = table_tbody.find_all('tr')

    

    #Criamos diversas listas para adicionar posteriormente os dados encontrados.

    countries = []

    total_cases = []

    new_cases = []

    total_deaths = []

    new_deaths = []

    total_recovered = []

    new_recovered = []

    active_cases = []

    serious_critical = []

    total_tests = []

    population = []

    continent = []

    

    # Itineramos cada linha da tabela

    for table_row in table_data:

        

        # A tabela apresentada no site contém dados compilados, como de determinados continentes e soma geral.

        # Como não precisamos desses dados - e poderiam atrapalhar a análise posterior -, vamos continuar a manipulação

        # sem adiconar esses dados nas listas.

        if 'total_row_world' in (table_row['class'] if table_row.has_attr('class') else []):

            continue

            

        row_data = table_row.find_all('td')

        

        # Adicionamos os dados nas listas

        countries.append(row_data[1].get_text().strip())

        total_cases.append(row_data[2].get_text().strip())

        new_cases.append(row_data[3].get_text().strip())

        total_deaths.append(row_data[4].get_text().strip())

        new_deaths.append(row_data[5].get_text().strip())

        total_recovered.append(row_data[6].get_text().strip())

        new_recovered.append(row_data[7].get_text().strip())

        active_cases.append(row_data[8].get_text().strip())

        serious_critical.append(row_data[9].get_text().strip())

        total_tests.append(row_data[12].get_text().strip())

        population.append(row_data[14].get_text().strip())

        continent.append(row_data[15].get_text().strip())

        

    # Compilamos todas as listas num DataFrame

    table_df = pd.DataFrame({

        'country': countries,

        'total_cases': total_cases,

        'total_deaths': total_deaths,

        'total_recovered': total_recovered,

        'active_cases': active_cases,

        'serious_critical': serious_critical,

        'total_tests': total_tests,

        'population': population,

        'continent': continent

    })

    

    # Separamos somente os dígitos dos dados númericos, excluindo pontos ou vírgulas.

    table_df['total_cases'] = pd.to_numeric(table_df['total_cases'].str.findall(pat = '(\-?\.?\d)').apply(''.join), errors='coerce')

    table_df['total_deaths'] = pd.to_numeric(table_df['total_deaths'].str.findall(pat = '(\-?\.?\d)').apply(''.join), errors='coerce')

    table_df['total_recovered'] = pd.to_numeric(table_df['total_recovered'].str.findall(pat = '(\-?\.?\d)').apply(''.join), errors='coerce')

    table_df['active_cases'] = pd.to_numeric(table_df['active_cases'].str.findall(pat = '(\-?\.?\d)').apply(''.join), errors='coerce')

    table_df['serious_critical'] = pd.to_numeric(table_df['serious_critical'].str.findall(pat = '(\-?\.?\d)').apply(''.join), errors='coerce')

    table_df['total_tests'] = pd.to_numeric(table_df['total_tests'].str.findall(pat = '(\-?\.?\d)').apply(''.join), errors='coerce')

    table_df['population'] = pd.to_numeric(table_df['population'].str.findall(pat = '(\-?\.?\d)').apply(''.join), errors='coerce')

    return table_df
# Hoje

today_table = worldometer_soup.find(id='main_table_countries_today')

today_df = worldometertable_to_df(today_table)

today_df.rename(columns = {

    'total_cases': 'total_cases_today',

    'total_deaths': 'total_deaths_today',

    'total_recovered': 'total_recovered_today',

    'active_cases': 'active_cases_today',

    'serious_critical': 'serious_critical_today',

    'total_tests': 'total_tests_today'

}, inplace = True)

today_df
# Ontem

yesterday_table = worldometer_soup.find(id='main_table_countries_yesterday')

yesterday_df = worldometertable_to_df(yesterday_table)

yesterday_df = yesterday_df.drop(['population','continent'], axis = 1)

yesterday_df.rename(columns = {

    'total_cases': 'total_cases_yesterday',

    'total_deaths': 'total_deaths_yesterday',

    'total_recovered': 'total_recovered_yesterday',

    'active_cases': 'active_cases_yesterday',

    'serious_critical': 'serious_critical_yesterday',

    'total_tests': 'total_tests_yesterday'

}, inplace = True)

yesterday_df
# Anteontem

before_yesterday_table = worldometer_soup.find(id='main_table_countries_yesterday2')

before_yesterday_df = worldometertable_to_df(before_yesterday_table)

before_yesterday_df = before_yesterday_df.drop(['population','continent'], axis = 1)

before_yesterday_df.rename(columns = {

    'total_cases': 'total_cases_bef_yesterday',

    'total_deaths': 'total_deaths_bef_yesterday',

    'total_recovered': 'total_recovered_bef_yesterday',

    'active_cases': 'active_cases_bef_yesterday',

    'serious_critical': 'serious_critical_bef_yesterday',

    'total_tests': 'total_tests_bef_yesterday'

}, inplace = True)

before_yesterday_df
corona_df = pd.merge(today_df, yesterday_df, on="country")

corona_df = pd.merge(corona_df, before_yesterday_df, on="country")

corona_df
corona_df = corona_df.reindex(columns=['continent','country','total_cases_today','total_cases_yesterday','total_cases_bef_yesterday','total_deaths_today','total_deaths_yesterday','total_deaths_bef_yesterday','total_recovered_today','total_recovered_yesterday','total_recovered_bef_yesterday','active_cases_today','active_cases_yesterday','active_cases_bef_yesterday','serious_critical_today','serious_critical_yesterday','serious_critical_bef_yesterday','total_tests_today','total_tests_yesterday','total_tests_bef_yesterday','population'])

corona_df
#Nossa tabela está pronta. Iremos gravá-la em um arquivo csv.

corona_df.to_csv('coronavirus.csv', index=False)
#Criaremos novas colunas no dataframe.

#Faremos colunas indicando a taxa de letalidade (mortos/recuperados) de cada país em cada um dos três dias. Iremos multiplicar o resultado por 100 para ser mostrado o número percentual (0 a 100%).



corona_df["mortality_rate_today"] = (corona_df["total_deaths_today"]/corona_df["total_cases_today"])*100

corona_df["mortality_rate_yesterday"] = (corona_df["total_deaths_yesterday"]/corona_df["total_cases_today"])*100

corona_df["mortality_rate_bef_yesterday"] = (corona_df["total_deaths_bef_yesterday"]/corona_df["total_cases_today"])*100

corona_df
#Em seguida faremos colunas indicando a taxa de recuperados (recuperados/contaminados) de cada país em cada dia.

corona_df["recovery_rate_today"] = (corona_df["total_recovered_today"]/corona_df["total_cases_today"])*100

corona_df["recovery_rate_yesterday"] = (corona_df["total_recovered_yesterday"]/corona_df["total_cases_yesterday"])*100

corona_df["recovery_rate_bef_yesterday"] = (corona_df["total_recovered_bef_yesterday"]/corona_df["total_cases_bef_yesterday"])*100

corona_df
#Por fim faremos novas colunas indicando a taxa de contaminados em tratamento (casos ainda ativos) de cada país em cada dia.

corona_df["active_cases_rate_today"] = (corona_df["active_cases_today"]/corona_df["total_cases_today"])*100

corona_df["active_cases_rate_yesterday"] = (corona_df["active_cases_yesterday"]/corona_df["total_cases_yesterday"])*100

corona_df["active_cases_rate_bef_yesterday"] = (corona_df["active_cases_bef_yesterday"]/corona_df["total_cases_bef_yesterday"])*100

corona_df
#Faremos algumas operações estatísticas (soma e média) com nossos dados.



#Soma do total de casos dos países em cada dia (do dia mais recente ao dia mais antigo):

print(corona_df["total_cases_today"].sum())             

print(corona_df["total_cases_yesterday"].sum())     

print(corona_df["total_cases_bef_yesterday"].sum()) 
#Média do total de casos dos países em cada dia:

print((corona_df["total_cases_today"].mean()))

print(corona_df["total_cases_yesterday"].mean())

print(corona_df["total_cases_bef_yesterday"].mean())
#Soma do total de mortes dos países em cada dia:

print(corona_df["total_deaths_today"].sum())

print(corona_df["total_deaths_yesterday"].sum())

print(corona_df["total_deaths_bef_yesterday"].sum())
#Média do total de mortes dos países em cada dia:

print(corona_df["total_deaths_today"].mean())

print(corona_df["total_deaths_yesterday"].mean())

print(corona_df["total_deaths_bef_yesterday"].mean())
#Soma do total de recuperados dos países em cada dia:

print(corona_df["total_recovered_today"].sum())

print(corona_df["total_recovered_yesterday"].sum())

print(corona_df["total_recovered_bef_yesterday"].sum())
#Média do total de recuperados dos países em cada dia:

print(corona_df["total_recovered_today"].mean())

print(corona_df["total_recovered_yesterday"].mean())

print(corona_df["total_recovered_bef_yesterday"].mean())
#Podemos pegar o total de mortos dos países e dividir pelo total de casos (depois multiplicar por 100) para sabermos o percentual de letalidade juntando todos os países. Faremos isso para os 3 dias.

#Guardaremos os resultados em variáveis a, b e c

a = ((corona_df["total_deaths_today"].mean()/corona_df["total_cases_today"].mean())*100)

b = ((corona_df["total_deaths_yesterday"].mean()/corona_df["total_cases_yesterday"].mean())*100)

c = ((corona_df["total_deaths_bef_yesterday"].mean()/corona_df["total_cases_bef_yesterday"].mean())*100)



print(a)

print(b)

print(c)
#Faremos o mesmo esquema para sabermos o percentual de recuperados em cada dia, juntando os países.

#Guardaremos os resultados em variáveis x, y e z

x = ((corona_df["total_recovered_today"].mean()/corona_df["total_cases_today"].mean())*100)

y = ((corona_df["total_recovered_yesterday"].mean()/corona_df["total_cases_yesterday"].mean())*100)

z = ((corona_df["total_recovered_bef_yesterday"].mean()/corona_df["total_cases_bef_yesterday"].mean())*100)

print(x)

print(y)

print(z)
#Para descobrirmos o percentual de casos ativos nesse esquema, fazemos o seguintes cálculos:

print(100 - a - x)

print(100 - b - y)

print(100 - c - z)
#importaremos algumas bibliotecas que possibilitam a criação de visualizações dos dados.

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
#criaremos gráficos de barras ranqueando os 50 primeiros países do dataframe de acordo com seus percentuais de letalidade, recuperados e casos ativos no último dia da tabela.

ranking_letalidade_hoje = corona_df.head(50).sort_values(["mortality_rate_today"], ascending = False)

ranking_recuperados_hoje = corona_df.head(50).sort_values(["recovery_rate_today"], ascending = False)

ranking_casos_ativos_hoje = corona_df.head(50).sort_values(["active_cases_rate_today"], ascending = False)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1,figsize=(20,30))

sns.barplot(x = ranking_letalidade_hoje["mortality_rate_today"], y = ranking_letalidade_hoje["country"], ax = ax1)

ax1.set_title("ranking dos países pela taxa de letalidade")

sns.barplot(x = ranking_recuperados_hoje["recovery_rate_today"], y = ranking_recuperados_hoje["country"], ax = ax2)

ax2.set_title("ranking dos países pela taxa de recuperados")

sns.barplot(x = ranking_casos_ativos_hoje["active_cases_rate_today"], y = ranking_casos_ativos_hoje["country"], ax = ax3)

ax3.set_title("ranking dos países pela taxa de casos ativos")
#Criaremos também um gráfico de dispersão relacionando o total de mortos no último dia dos 50 primeiros países da tabela (eixo y) com suas respectivas populações (eixo x). 

plt.figure(figsize = (10,5))

sns.scatterplot(x = corona_df["population"].head(50), y = corona_df["total_deaths_today"].head(50))

plt.axvline((corona_df["population"]).mean(), color = "red", linestyle = "--", label = "média da população dos países do gráfico")

plt.axhline((corona_df["total_deaths_today"]).mean(), color = "black", linestyle = "--", label = "média de mortos dos países do gráfico")

plt.legend()
#E por fim criaremos um gráfico de multibarras relacionando o número total de casos dos 10 primeiros países da tabela em cada um dos 3 dias

ax = plt.gca()

corona_df.head(10).plot(kind = "bar", x = "country", y = "total_cases_today", color = "blue", ax = ax)

corona_df.head(10).plot(kind = "bar", x = "country", y = "total_cases_yesterday", color = "red", ax = ax)

corona_df.head(10).plot(kind = "bar", x = "country", y = "total_cases_bef_yesterday", color = "yellow", ax = ax)



plt.show()