from bs4 import BeautifulSoup
from urllib.request import urlopen
url_sobre = 'https://ocean-web-scraping.herokuapp.com/sobre.html'
url_busca = 'https://ocean-web-scraping.herokuapp.com/'
url_corretoras = 'https://ocean-web-scraping.herokuapp.com/corretoras.html'
response_sobre = urlopen(url_sobre)
html_sobre = response_sobre.read()
html_sobre
soup_sobre = BeautifulSoup(html_sobre, 'html.parser')
soup_sobre
response_busca = urlopen(url_busca)
html_busca = response_busca.read()
html_busca
soup_busca = BeautifulSoup(html_busca, 'html.parser')
soup_busca
response_corretoras = urlopen(url_corretoras)
html_corretoras = response_corretoras.read()
html_corretoras
soup_corretoras = BeautifulSoup(html_corretoras, 'html.parser')
soup_corretoras
# Insira o seu código aqui
# Insira o seu código aqui
# Insira o seu código aqui
# Insira o seu código aqui
import pandas as pd
soup_busca.find('div', {'class': 'ad-card'})
soup_busca.find('ul', {'class': 'features'})
soup_busca.find('ul', {'class': 'features'}).find_all('li')
for caracteristica in soup_busca.find('ul', {'class': 'features'}).find_all('li'):
    print(caracteristica.get_text())
soup_busca.find('ul', {'class': 'resources'})
soup_busca.find('ul', {'class': 'resources'}).find_all('li')
for recurso in soup_busca.find('ul', {'class': 'resources'}).find_all('li'):
    print(recurso.get_text())
soup_busca.find('div', {'class': 'ad-card-price'})
soup_busca.find('div', {'class': 'ad-card-price'}).find_all('p')
soup_busca.find('div', {'class': 'ad-card-price'}).find_all('p')[:-1]
for precos in soup_busca.find('div', {'class': 'ad-card-price'}).find_all('p')[:-1]:
    print(precos.get_text())
soup_busca.find('div', {'class': 'ad-card-price'})
soup_busca.find('div', {'class': 'ad-card-price'}).find_all('p')
soup_busca.find('div', {'class': 'ad-card-price'}).find_all('p')[-1]
soup_busca.find('div', {'class': 'ad-card-price'}).find_all('p')[-1].get_text()
soup_busca.find('div', {'class': 'ad-card-info'})
soup_busca.find('div', {'class': 'ad-card-info'}).find('p')
soup_busca.find('div', {'class': 'ad-card-info'}).find('p').get_text()
soup_busca.find('div', {'class': 'ad-card-list'})
soup_busca.find('div', {'class': 'ad-card-list'}).find_all('div', {'class': 'ad-card'})
for resultado in soup_busca.find('div', {'class': 'ad-card-list'}).find_all('div', {'class': 'ad-card'}):
    print(resultado)
    print('-------------')
for resultado in soup_busca.find('div', {'class': 'ad-card-list'}).find_all('div', {'class': 'ad-card'}):
    caracteristicas = resultado.find('ul', {'class': 'features'})
    print(caracteristicas)
    print('-------------')
for resultado in soup_busca.find('div', {'class': 'ad-card-list'}).find_all('div', {'class': 'ad-card'}):
    caracteristicas = resultado.find('ul', {'class': 'features'})
    
    for caract in caracteristicas.find_all('li'):
        print(caract.get_text())
    
    print('-------------')
for resultado in soup_busca.find('div', {'class': 'ad-card-list'}).find_all('div', {'class': 'ad-card'}):
    caracteristicas = resultado.find('ul', {'class': 'features'})
    recursos = resultado.find('ul', {'class': 'resources'})
    
    for caract in caracteristicas.find_all('li'):
        print(caract.get_text())
        
    for recur in recursos.find_all('li'):
        print(recur.get_text())
        
    print('-------------')
for resultado in soup_busca.find('div', {'class': 'ad-card-list'}).find_all('div', {'class': 'ad-card'}):
    caracteristicas = resultado.find('ul', {'class': 'features'})
    recursos = resultado.find('ul', {'class': 'resources'})
    precos_corretora = resultado.find('div', {'class': 'ad-card-price'})
    
    for caract in caracteristicas.find_all('li'):
        print(caract.get_text())
        
    for recur in recursos.find_all('li'):
        print(recur.get_text())
        
    for prec_corr in precos_corretora.find_all('p'):
        print(prec_corr.get_text())
        
    print('-------------')
for resultado in soup_busca.find('div', {'class': 'ad-card-list'}).find_all('div', {'class': 'ad-card'}):
    caracteristicas = resultado.find('ul', {'class': 'features'})
    recursos = resultado.find('ul', {'class': 'resources'})
    precos_corretora = resultado.find('div', {'class': 'ad-card-price'})
    localizacao = resultado.find('div', {'class': 'ad-card-info'}).find('p')
    
    for caract in caracteristicas.find_all('li'):
        print(caract.get_text())
        
    for recur in recursos.find_all('li'):
        print(recur.get_text())
        
    for prec_corr in precos_corretora.find_all('p'):
        print(prec_corr.get_text())
    
    print(localizacao.get_text())
        
    print('-------------')
resutados_lista = []
for resultado in soup_busca.find('div', {'class': 'ad-card-list'}).find_all('div', {'class': 'ad-card'}):
    caracteristicas = resultado.find('ul', {'class': 'features'})
    recursos = resultado.find('ul', {'class': 'resources'})
    precos_corretora = resultado.find('div', {'class': 'ad-card-price'})
    localizacao = resultado.find('div', {'class': 'ad-card-info'}).find('p')
    
    resutado_dict = {}
    
    resutado_dict['área'] = caracteristicas.find_all('li')[0].get_text()
    resutado_dict['quartos'] = caracteristicas.find_all('li')[1].get_text()
    resutado_dict['banheiros'] = caracteristicas.find_all('li')[2].get_text()
    resutado_dict['vagas garagem'] = caracteristicas.find_all('li')[3].get_text()
    
    for recur in recursos.find_all('li'):
        resutado_dict[recur.get_text().lower()] = 'sim'
        
    resutado_dict['aluguel'] = precos_corretora.find_all('p')[0].get_text()
    resutado_dict['condomínio'] = precos_corretora.find_all('p')[1].get_text()
    resutado_dict['iptu'] = precos_corretora.find_all('p')[2].get_text()
    resutado_dict['corretora'] = precos_corretora.find_all('p')[3].get_text()
    
    resutado_dict['localizacao'] = localizacao.get_text()
    
    resutados_lista.append(resutado_dict)
resutados_lista
resultados_df = pd.DataFrame(resutados_lista)
resultados_df
resultados_df.dtypes
resutados_lista = []
for resultado in soup_busca.find('div', {'class': 'ad-card-list'}).find_all('div', {'class': 'ad-card'}):
    caracteristicas = resultado.find('ul', {'class': 'features'})
    recursos = resultado.find('ul', {'class': 'resources'})
    precos_corretora = resultado.find('div', {'class': 'ad-card-price'})
    localizacao = resultado.find('div', {'class': 'ad-card-info'}).find('p')
    
    resutado_dict = {}
    
    resutado_dict['área'] = int(caracteristicas.find_all('li')[0].get_text().replace(' ','').replace('m²',''))
    resutado_dict['quartos'] = int(caracteristicas.find_all('li')[1].get_text().replace(' ','').replace('quarto(s)',''))
    resutado_dict['banheiros'] = int(caracteristicas.find_all('li')[2].get_text().replace(' ','').replace('banheiro(s)',''))
    resutado_dict['vagas garagem'] = int(caracteristicas.find_all('li')[3].get_text().replace(' ','').replace('vaga(s)',''))
    
    for recur in recursos.find_all('li'):
        resutado_dict[recur.get_text().lower()] = 'sim'
        
    resutado_dict['aluguel'] = float(precos_corretora.find_all('p')[0].get_text().replace('Aluguel: R$ ','').replace('.','').replace(',','.'))
    resutado_dict['condomínio'] = float(precos_corretora.find_all('p')[1].get_text().replace('Condomínio: ','').replace('.','').replace(',','.'))
    resutado_dict['iptu'] = float(precos_corretora.find_all('p')[2].get_text().replace('IPTU: R$ ','').replace('.','').replace(',','.'))
    resutado_dict['corretora'] = precos_corretora.find_all('p')[3].get_text().replace('Corretora: ','')
    
    resutado_dict['localizacao'] = localizacao.get_text().replace('Localização: ','')
    
    resutados_lista.append(resutado_dict)
resultados_df = pd.DataFrame(resutados_lista)
resultados_df
resultados_df = resultados_df.fillna(value='não')
resultados_df
resultados_df.dtypes
import os
os.makedirs('imagens')
soup_busca.find('div', {'class': 'ad-card-list'})
soup_busca.find('div', {'class': 'ad-card-list'}).find_all('div', {'class': 'ad-card'})
for resultado in soup_busca.find('div', {'class': 'ad-card-list'}).find_all('div', {'class': 'ad-card'}):
    print(resultado.img)
for resultado in soup_busca.find('div', {'class': 'ad-card-list'}).find_all('div', {'class': 'ad-card'}):
    print(resultado.img.get('src'))
for resultado in soup_busca.find('div', {'class': 'ad-card-list'}).find_all('div', {'class': 'ad-card'}):
    print(resultado.img.get('src').replace('imagens/',''))
endereco = 'https://ocean-web-scraping.herokuapp.com/'

for resultado in soup_busca.find('div', {'class': 'ad-card-list'}).find_all('div', {'class': 'ad-card'}):
    print(endereco + resultado.img.get('src'))
from urllib.request import urlretrieve
for resultado in soup_busca.find('div', {'class': 'ad-card-list'}).find_all('div', {'class': 'ad-card'}):
    endereco_imagem = endereco + resultado.img.get('src')
    nome_imagem = resultado.img.get('src').replace('imagens/','')
    
    urlretrieve(endereco_imagem, '/kaggle/working/imagens/' + nome_imagem)
from zipfile import ZipFile
with ZipFile('imagens.zip', 'w') as zf:
    for pasta_principal, sub_pastas, arquivos in os.walk('/kaggle/working/imagens/'):
        for arquivo in arquivos:
            pasta = os.path.join(pasta_principal, arquivo)
            zf.write(pasta)
resutados_lista = []
for i in range(1,6):
    url_busca_todos = 'https://ocean-web-scraping.herokuapp.com/results&page=' + str(i) + '.html'
    print(url_busca_todos)
    response_busca_todos = urlopen(url_busca_todos)
    html_busca_todos = response_busca_todos.read()
    soup_busca_todos = BeautifulSoup(html_busca_todos, 'html.parser')
    
    for resultado in soup_busca_todos.find('div', {'class': 'ad-card-list'}).find_all('div', {'class': 'ad-card'}):
        caracteristicas = resultado.find('ul', {'class': 'features'})
        recursos = resultado.find('ul', {'class': 'resources'})
        precos_corretora = resultado.find('div', {'class': 'ad-card-price'})
        localizacao = resultado.find('div', {'class': 'ad-card-info'}).find('p')

        resutado_dict = {}

        resutado_dict['área'] = int(caracteristicas.find_all('li')[0].get_text().replace(' ','').replace('m²',''))
        resutado_dict['quartos'] = int(caracteristicas.find_all('li')[1].get_text().replace(' ','').replace('quarto(s)',''))
        resutado_dict['banheiros'] = int(caracteristicas.find_all('li')[2].get_text().replace(' ','').replace('banheiro(s)',''))
        resutado_dict['vagas garagem'] = int(caracteristicas.find_all('li')[3].get_text().replace(' ','').replace('vaga(s)',''))

        for recur in recursos.find_all('li'):
            resutado_dict[recur.get_text().lower()] = 'sim'

        resutado_dict['aluguel'] = float(precos_corretora.find_all('p')[0].get_text().replace('Aluguel: R$ ','').replace('.','').replace(',','.'))
        resutado_dict['condomínio'] = float(precos_corretora.find_all('p')[1].get_text().replace('Condomínio: ','').replace('.','').replace(',','.'))
        resutado_dict['iptu'] = float(precos_corretora.find_all('p')[2].get_text().replace('IPTU: R$ ','').replace('.','').replace(',','.'))
        resutado_dict['corretora'] = precos_corretora.find_all('p')[3].get_text().replace('Corretora: ','')

        resutado_dict['localizacao'] = localizacao.get_text().replace('Localização: ','')

        resutados_lista.append(resutado_dict)
resultados_df = pd.DataFrame(resutados_lista)
resultados_df = resultados_df.fillna(value='não')
resultados_df
resultados_df.to_csv('./resultados.csv', sep = ';', index = False, encoding = 'utf-8')
for i in range(1,6):
    url_busca_todos = 'https://ocean-web-scraping.herokuapp.com/results&page=' + str(i) + '.html'
    print(url_busca_todos)
    response_busca_todos = urlopen(url_busca_todos)
    html_busca_todos = response_busca_todos.read()
    soup_busca_todos = BeautifulSoup(html_busca_todos, 'html.parser')
    
    endereco = 'https://ocean-web-scraping.herokuapp.com/'
    
    for resultado in soup_busca_todos.find('div', {'class': 'ad-card-list'}).find_all('div', {'class': 'ad-card'}):
        endereco_imagem = endereco + resultado.img.get('src')
        nome_imagem = resultado.img.get('src').replace('imagens/','')
    
        urlretrieve(endereco_imagem, '/kaggle/working/imagens/' + nome_imagem)
with ZipFile('imagens.zip', 'w') as zf:
    for pasta_principal, sub_pastas, arquivos in os.walk('/kaggle/working/imagens/'):
        for arquivo in arquivos:
            pasta = os.path.join(pasta_principal, arquivo)
            zf.write(pasta)
lista_corretoras = []
lista_nomes = []
lista_contatos = []
contatos_corretoras_dict = {}
soup_corretoras.find('tbody')
soup_corretoras.find('tbody').find_all('tr')
soup_corretoras.find('tbody').find_all('tr')[0]
soup_corretoras.find('tbody').find_all('tr')[0].find_all('td')
print('Corretora: ', soup_corretoras.find('tbody').find_all('tr')[0].find_all('td')[0].get_text())
print('Contatos: ', soup_corretoras.find('tbody').find_all('tr')[0].find_all('td')[1].get_text())
soup_corretoras.find('tbody').find_all('tr')[0].find_all('td')[1].get_text().split(' - Fone:')
print('Corretora: ', soup_corretoras.find('tbody').find_all('tr')[0].find_all('td')[0].get_text())

contato = soup_corretoras.find('tbody').find_all('tr')[0].find_all('td')[1].get_text().split(' - Fone:')
print('Nome: ', contato[0])
print('Telefone: ', contato[1])
lista_corretoras.append(soup_corretoras.find('tbody').find_all('tr')[0].find_all('td')[0].get_text())
lista_nomes.append(contato[0])
lista_contatos.append(contato[1])
lista_corretoras, lista_nomes, lista_contatos
soup_corretoras.find('tbody').find_all('tr')[1]
soup_corretoras.find('tbody').find_all('tr')[1].find_all('td')
soup_corretoras.find('tbody').find_all('tr')[1].find_all('td')[1].get_text().split(' - Fone:')
soup_corretoras.find('tbody').find_all('tr')[1].find_all('td')[1].get_text().split('; ')
for item in soup_corretoras.find('tbody').find_all('tr')[1].find_all('td')[1].get_text().split('; '):
    print(item.split(' - Fone:'))
print('Corretora: ', soup_corretoras.find('tbody').find_all('tr')[1].find_all('td')[0].get_text())

contatos = soup_corretoras.find('tbody').find_all('tr')[1].find_all('td')[1].get_text().split('; ')

for contato in contatos:
    contato = contato.split(' - Fone:')
    print('Nome: ', contato[0])
    print('Telefone: ', contato[1])
contatos = soup_corretoras.find('tbody').find_all('tr')[1].find_all('td')[1].get_text().split('; ')

for contato in contatos:
    lista_corretoras.append(soup_corretoras.find('tbody').find_all('tr')[1].find_all('td')[0].get_text())
    
    contato = contato.split(' - Fone:')
    
    lista_nomes.append(contato[0])
    lista_contatos.append(contato[1])
lista_corretoras, lista_nomes, lista_contatos
soup_corretoras.find('tbody').find_all('tr')[2]
soup_corretoras.find('tbody').find_all('tr')[2].find_all('td')
soup_corretoras.find('tbody').find_all('tr')[2].find_all('td')[1].find_all('li')
print('Corretora: ', soup_corretoras.find('tbody').find_all('tr')[2].find_all('td')[0].get_text())

contatos = soup_corretoras.find('tbody').find_all('tr')[2].find_all('td')[1].find_all('li')

for contato in contatos:
    contato = contato.get_text().split(' - Fone:')
    print('Nome: ', contato[0])
    print('Telefone: ', contato[1])
contatos = soup_corretoras.find('tbody').find_all('tr')[2].find_all('td')[1].find_all('li')

for contato in contatos:
    lista_corretoras.append(soup_corretoras.find('tbody').find_all('tr')[2].find_all('td')[0].get_text())
    
    contato = contato.get_text().split(' - Fone:')
    
    lista_nomes.append(contato[0])
    lista_contatos.append(contato[1])
lista_corretoras, lista_nomes, lista_contatos
!pip install tabula-py
import tabula
soup_corretoras.find('tbody').find_all('tr')[3].find_all('td')
soup_corretoras.find('tbody').find_all('tr')[3].find_all('td')[1]
soup_corretoras.find('tbody').find_all('tr')[3].find_all('td')[1].find('a')
soup_corretoras.find('tbody').find_all('tr')[3].find_all('td')[1].find('a').get('href')
nome_pdf = soup_corretoras.find('tbody').find_all('tr')[3].find_all('td')[1].find('a').get('href')
arquivo = 'https://ocean-web-scraping.herokuapp.com/' + nome_pdf
arquivo
corretora_enguia_df = tabula.read_pdf(arquivo, pages='all', multiple_tables=False)[0]
corretora_enguia_df
print('Corretora: ', soup_corretoras.find('tbody').find_all('tr')[3].find_all('td')[0].get_text())

for nome,contato in zip(corretora_enguia_df['Corretor(a)'].to_list(),
                        corretora_enguia_df['Contato'].to_list()):
    print(nome)
    print(contato.replace('Fone:',''))
for nome,contato in zip(corretora_enguia_df['Corretor(a)'].to_list(),
                        corretora_enguia_df['Contato'].to_list()):
    lista_corretoras.append(soup_corretoras.find('tbody').find_all('tr')[3].find_all('td')[0].get_text())
    lista_nomes.append(nome)
    lista_contatos.append(contato.replace('Fone:',''))
lista_corretoras, lista_nomes, lista_contatos
contatos_corretoras_dict.update({'Corretoras': lista_corretoras,
                                 'Nomes': lista_nomes,
                                 'Contatos': lista_contatos})

contatos_corretoras_dict
contatos_corretoras_df = pd.DataFrame.from_dict(contatos_corretoras_dict)

contatos_corretoras_df
contatos_corretoras_df.to_csv('./contatos_corretoras.csv', sep = ';', index = False, encoding = 'utf-8')
resultados_df
busca = resultados_df['aluguel'] <= 1800
busca
len(resultados_df.loc[busca])
resultados_df.loc[busca]
busca = ((resultados_df['aluguel'] <= 1800) & (resultados_df['quartos'] >= 2))
len(resultados_df.loc[busca])
resultados_df.loc[busca]
busca = ((resultados_df['aluguel'] <= 1800) & 
         (resultados_df['quartos'] >= 2) & 
         (resultados_df['vagas garagem'] >= 1))
len(resultados_df.loc[busca])
resultados_df.loc[busca]
busca = ((resultados_df['aluguel'] <= 1800) & 
         (resultados_df['quartos'] >= 2) & 
         (resultados_df['vagas garagem'] >= 1) &
         (resultados_df['localizacao'] == 'Oceano Atlântico'))
len(resultados_df.loc[busca])
resultados_df.loc[busca]
contatos_corretoras_df
busca_corretora = contatos_corretoras_df['Corretoras'] == 'Carangueijo'
contatos_corretoras_df.loc[busca_corretora]