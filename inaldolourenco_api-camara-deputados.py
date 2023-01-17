#API Camara dos deputados

#Trabalho desenvolvido com dados abertos da Camara dos Deputados através de API.

#1. Api 
#https://dadosabertos.camara.leg.br/swagger/api.html

#2. Jupyter Notebbok

#partidos dos deputados
import requests
url = 'https://dadosabertos.camara.leg.br/api/v2/blocos?ordem=ASC&ordenarPor=nome'
resp = requests.get(url).json()
for d in resp['dados']:
    print (d['nome'], d['id'])


#lista dos deputados + ID
import requests
url = 'https://dadosabertos.camara.leg.br/api/v2/deputados?ordem=ASC&ordenarPor=nome'
resp = requests.get(url).json()
for d in resp['dados']:
    print (d['nome'], d['id'])

# Salva as fotos dos deputados no pc
import requests
url = 'https://dadosabertos.camara.leg.br/api/v2/deputados?siglaUf=&siglaUf=&ordem=ASC&ordenarPor=nome'
resp = requests.get(url).json()
for d in resp['dados']:
    if d['siglaPartido'] == 'PT' and d['siglaUf'] == '': AL  #
    nome = d['nome'].lower()   
    print (f'Gravando: {nome}')
    f = open(nome + '.jpg', 'wb')
    foto = requests.get(d['urlFoto']).content # pega o conteudo
    f.write(foto) # grava
    f.close() #fecha