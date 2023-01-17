import requests 
from bs4 import BeautifulSoup 
import pandas as pd 
import re
# Denzel's movie list...
url_fonte = 'http://www.imdb.com/name/nm0000243/?ref_=nv_sr_1'


# Connecting on the page
con = requests.get(url_fonte)
con.status_code
# Creating BeautifulSoup object 
soup = BeautifulSoup(con.content, "html.parser")
#print(soup.prettify())
# Getting everything about Denzel's participations in productions
nomeFilmeTudo = [tag.find('a') for tag in soup.findAll('div', class_='filmo-row')]
#nomeFilmeTudo
#Using re to clean data  

anoFilmeTudo = [tag.find('span') for tag in soup.findAll('div', class_='filmo-row')]

anosFilmes = []
for num in anoFilmeTudo:
    id = anoFilmeTudo.index(num)
    c = re.findall('\d+', str(anoFilmeTudo[id]))
    c = re.sub("\['", '', str(c))
    c = re.sub("\']", '', str(c))
    c = re.sub("\'\,", ' -', str(c))
    c = re.sub("\ '", ' ', str(c))
    anosFilmes.append(c)
    
#anosFilmes
#Using re to create the complete link of the production
#Example: http://www.imdb.com/title/tt0168786/?ref_=nm_flmg_prd_8 

linkPart = "http://www.imdb.com"
linksFilmes = []
for num in nomeFilmeTudo:
    id = nomeFilmeTudo.index(num)
    a = re.findall(r'"(.*?)"', str(nomeFilmeTudo[id]))
    a = re.sub("\['", '', str(a))
    a = re.sub("\']", '', str(a))
    a = linkPart + str(a)
    linksFilmes.append(a) 
    #print(linkCompl)
    
#len(linksFilmes)
#print (linksFilmes)
#Data Cleaning using re to get movie titles 
titulosFilmes =[]
for num in nomeFilmeTudo:
    id = nomeFilmeTudo.index(num)
    b = re.findall(r'>(.*?)<', str(nomeFilmeTudo[id]))
    b = re.sub("\['", '', str(b))
    b = re.sub("\']", '', str(b))
    b = re.sub('\["', '', str(b))
    b = re.sub('\"]', '', str(b))
    #a = linkPart + str(a)
    titulosFilmes.append(b) 
    #print(linkCompl)

#print (titulosFilmes)
#TESTES ANTES DAS FUNÇÕES
#Para testar Funções...
#con2 = requests.get('http://www.imdb.com/title/tt6000478/?ref_=nm_flmg_act_2') 
#soup2 = BeautifulSoup(con2.content, "html.parser")

#Título do Filme
#titFilme = [tag.find('h1', itemprop='name') for tag in soup2.findAll('div', class_='title_wrapper')]
#titFilme = re.findall(r'>(.*?)<', str(titFilme))
#print(titFilme[0])
#print("===============")

#Ano do Filme
#anoFilme = [tag.find('a') for tag in soup2.findAll('span', id_='titleYear')]
#anoFilme = [tag.find('a') for tag in soup2.findAll('h1', itemprop='name')]
#anoFilme = re.findall(r'>(.*?)<', str(anoFilme))
#print(anoFilme[0])
#print("++++++++++++")

#Genero
#genero = [tag.find('span', itemprop='genre') for tag in soup2.findAll('div', class_='subtext')]
#genero = re.findall(r'>(.*?)<', str(genero))
#print (genero[0])
#print("|||||||||||||")
# colocando os gêneros(se tiver + de 1) na mesma linha???

#Nota IMDB
#rating = [tag.find('span', itemprop='ratingValue') for tag in soup2.findAll('div', class_='ratingValue')]
#rating = re.findall(r'>(.*?)<', str(rating))
#print (rating[0])
#print("----------------")

#Reviews
#reviews = [tag.find('span', itemprop='ratingCount') for tag in soup2.findAll('div', class_='imdbRating')]
#reviews = re.findall(r'>(.*?)<', str(reviews))
#reviews = str.replace(',','',str(reviews))
#print (reviews[0])
#print("----------------")

#Duracao
#duracao = [tag.find('time', itemprop='duration') for tag in soup2.findAll('div', class_='subtext')]
#duracao = re.findall(r'\d+', str(duracao))
#print(duracao[0])
#print("----------------")

#Nota Metascore
#metascore = [tag.find('span') for tag in soup2.findAll('div', class_='metacriticScore score_mixed titleReviewBarSubItem')]
#metascore = re.findall(r'>(.*?)<', str(metascore))
#print(metascore[0])
#print("----------------")

#Reviews Metascore - Usuarios
#revMetascore = [tag.find('span') for tag in soup2.findAll('div', class_='titleReviewBarItem titleReviewbarItemBorder')]
#revMetascore = re.findall(r'>(.*?)<', str(revMetascore))
#revMetascore = re.findall(r'\d+', str(revMetascore))
#print(revMetascore[0])
#print(revMetascore[1])
#print("------++++++++-----")

#Diretor
#diretor = [tag.find('span', itemprop='name') for tag in soup2.findAll('span', itemprop='director')]
#diretor = re.findall(r'>(.*?)<', str(diretor))
#print(diretor[0])
#print("_______________")

#Roteirista
#roteirista = [tag.find('span', itemprop='name') for tag in soup2.findAll('span', itemprop='creator')]
#roteirista = re.findall(r'>(.*?)<', str(roteirista))
#print(roteirista[0])
#print("*****************")
#Year
def pegaAno(soup):
    #ano = soup.select("a[href*=/year/]")
    anoFilme = [tag.find('a') for tag in soup2.findAll('h1', itemprop='name')]
    anoFilme = re.findall(r'>(.*?)<', str(anoFilme))
    if len(anoFilme):
        return int(anoFilme[0])
    else:
        return None

#ano = pegaAno(soup2)
#print(ano)
#Genre
#con2 = requests.get('https://www.imdb.com/title/tt3766354/?ref_=nm_flmg_act_1') 
#soup2 = BeautifulSoup(con2.content, "html.parser")

def pegaGenero(soup):
    #genFilme = [tag.find('span', itemprop='genre') for tag in soup2.findAll('div', class_='subtext')]
    genFilme = [tag.find('a') for tag in soup2.findAll('div', itemprop='genre')]
    genFilme = re.findall(r'>(.*?)<', str(genFilme))
    if len(genFilme):
        return str(genFilme[0])
    else:
        return None
    
#genero = pegaGenero(soup2)
#print(genero)
#IMDb Score
def pegaNotaIMDB(soup):
    ratFilme = [tag.find('span', itemprop='ratingValue') for tag in soup2.findAll('div', class_='ratingValue')]
    ratFilme = re.findall(r'>(.*?)<', str(ratFilme))
    if len(ratFilme):
        return str(ratFilme[0])
    else:
        return None
    
#rating = pegaNotaIMDB(soup2)
#print(rating)
#Reviews
#con2 = requests.get('http://www.imdb.com/title/tt6000478/?ref_=nm_flmg_act_2') 
#soup2 = BeautifulSoup(con2.content, "html.parser")
def pegaReviewIMDB(soup):
    revFilme = [tag.find('span', itemprop='ratingCount') for tag in soup2.findAll('div', class_='imdbRating')]
    revFilme = re.findall(r'>(.*?)<', str(revFilme))
    if len(revFilme):
        revFilme[0] = re.sub(',', '', str(revFilme[0]))
        return int(revFilme[0])
    else:
        return None
    
#review = pegaReviewIMDB(soup2)
#print(review)
#Duration
def pegaDuracao(soup):
    durFilme = [tag.find('time', itemprop='duration') for tag in soup2.findAll('div', class_='subtext')]
    durFilme = re.findall(r'\d+', str(durFilme))
    if len(durFilme):
        return int(durFilme[0])
    else:
        return None
    
#duracao = pegaDuracao(soup2)
#print(duracao)
#Metacritic Score (Metascore)

#Conexão com filmes indicando o link de cada um no loop...
#con3 = requests.get('http://www.imdb.com/title/tt0454848/?ref_=nm_flmg_act_15')
#con3 = requests.get('http://www.imdb.com/title/tt0119099/?ref_=nm_flmg_act_27')
#soup3 = BeautifulSoup(con3.content, "html.parser")
#con3.close()

def pegaNotaMetaScore(soup):
    notaFilme = [tag.find('span') for tag in soup.findAll('div', class_='metacriticScore score_mixed titleReviewBarSubItem')]
    notaFilme = re.findall(r'>(.*?)<', str(notaFilme))
    notaFilme2 = [tag.find('span') for tag in soup.findAll('div', class_='metacriticScore score_favorable titleReviewBarSubItem')]
    notaFilme2 = re.findall(r'>(.*?)<', str(notaFilme2))
    notaFilme3 = [tag.find('span') for tag in soup.findAll('div', class_='metacriticScore score_unfavorable titleReviewBarSubItem')]
    notaFilme3 = re.findall(r'>(.*?)<', str(notaFilme3))
    if len(notaFilme):
        return int(notaFilme[0])
    elif len(notaFilme2):
        return int(notaFilme2[0])
    elif len(notaFilme3):
        return int(notaFilme3[0])
    else:
        return None
        
#notaMeta = pegaNotaMetaScore(soup3)
#print(notaMeta)
#Synopsis
#Conexão com filmes indicando o link de cada um no loop...
#con3 = requests.get('http://www.imdb.com/title/tt0454848/?ref_=nm_flmg_act_15')  
#soup3 = BeautifulSoup(con3.content, "html.parser")
#con3.close()

def pegaSinopse(soup):
    #sinFilme = [tag.find('div', class_='summary_text') for tag in soup.findAll('div', class_='plot_summary')]
    #sinFilme = re.findall(r'>(.*?)<', str(sinFilme))
    sinFilme = soup.find('div', class_='summary_text')
    if len(sinFilme):
        return str(sinFilme.text)
    else:
        return None
        
#sinopse = pegaSinopse(soup3)
#print(sinopse)
#Metascore Reviews - Users
def pegaRevMetaScoreUser(soup):
    revScore = [tag.find('span') for tag in soup2.findAll('div', class_='titleReviewBarItem titleReviewbarItemBorder')]
    revScore = re.findall(r'>(.*?)<', str(revScore))
    revScore = re.findall(r'\d+', str(revScore))
    if len(revScore):
        return int(revScore[0])
    else:
        return None
    
#notaMeta = pegaRevMetaScoreUser(soup2)
#print(notaMeta)
#Metascore Reviews - Critics
def pegaRevMetaScoreCrit(soup):
    revCrit = [tag.find('span') for tag in soup2.findAll('div', class_='titleReviewBarItem titleReviewbarItemBorder')]
    revCrit = re.findall(r'>(.*?)<', str(revCrit))
    revCrit = re.findall(r'\d+', str(revCrit))
    if len(revCrit):
        return int(revCrit[1])
    else:
        return None
    
#crit = pegaRevMetaScoreCrit(soup2)
#print(crit)
#Director
def pegaDiretor(soup):
    diretorFilme = [tag.find('span', itemprop='name') for tag in soup2.findAll('span', itemprop='director')]
    diretorFilme = re.findall(r'>(.*?)<', str(diretorFilme))
    if len(diretorFilme):
        return str(diretorFilme[0])
    else:
        return None
    
#diretor = pegaDiretor(soup2)
#print(diretor)
#Writer
def pegaRoteirista(soup):
    roteiristaFilme = [tag.find('span', itemprop='name') for tag in soup2.findAll('span', itemprop='creator')]
    roteiristaFilme = re.findall(r'>(.*?)<', str(roteiristaFilme))
    if len(roteiristaFilme):
        return str(roteiristaFilme[0])
    else:
        return None
    
#roteirista = pegaRoteirista(soup2)
#print(roteirista)
#Locations
#con3 = requests.get('https://www.imdb.com/title/tt0455944/?ref_=nm_flmg_act_5')  
#soup3 = BeautifulSoup(con3.content, "html.parser")
#con3.close()
def pegaLocacao(soup):
    #locacaoFilme = [tag.find('h4', class_='inline') for tag in soup3.findAll('div', class_='txt-block')]
    locacaoFilme = soup3.findAll('div', class_='txt-block')
    locacaoFilme = locacaoFilme[8].get_text()
    #locacaoFilme = re.findall(r'(\:\).*?)\n', str(locacaoFilme))
    if len(locacaoFilme):
        return str(locacaoFilme)
    else:
        return None
    
#locacao = pegaLocacao(soup3)
#print(locacao)
#BUDGET
#Opening Weekend USA
#Gross USA
#Cumulative Gross Worldwide

#con3 = requests.get('https://www.imdb.com/title/tt0455944/?ref_=nm_flmg_act_5')  
#con3 = requests.get('https://www.imdb.com/title/tt0088146/?ref_=nm_flmg_act_51')
con3 = requests.get('https://www.imdb.com/title/tt0119099/?ref_=nm_flmg_act_27')
#con3 = requests.get('https://www.imdb.com/title/tt0111996/?ref_=nm_flmg_act_28')
soup3 = BeautifulSoup(con3.content, "html.parser")
con3.close()

def pegaBudgetGross(soup):
    #budgetFilme = [tag.find('div', class_='article') for tag in soup3.findAll('div', class_='txt-block')]
    #budgetFilme = [tag.find('h4', class_='inline') for tag in soup3.findAll('div', class_='txt-block')]    
    valoresFilme = soup.findAll('div', class_='txt-block') 
    r = len(valoresFilme)
    print(r)
    
    
    budgetFilme = valoresFilme[9].get_text() if len(valoresFilme)>9 else ''
    openWeekFilme = valoresFilme[10].get_text() if len(valoresFilme)>10 else ''
    grossFilme = valoresFilme[11].get_text() if len(valoresFilme)>11 else ''
    cumWorldGross = valoresFilme[12].get_text() if len(valoresFilme)>12 else ''

    def limpaStringVoltaInt(string):
        #tratando valores com re
        string = re.findall('(\d+),(\d+),(\d+)', str(string))
        string = re.sub(',', '', str(string))
        string = re.sub('\'', '', str(string))
        string = re.sub("\s", "", str(string))
        string = re.sub("\[\(", "", str(string))
        string = re.sub("\)\]", "", str(string))
        return string
     
   
    budgetFilme = limpaStringVoltaInt(budgetFilme)
    openWeekFilme = limpaStringVoltaInt(openWeekFilme)
    grossFilme = limpaStringVoltaInt(grossFilme)
    cumWorldGross = limpaStringVoltaInt(cumWorldGross)
    
    if len(valoresFilme):
        return (budgetFilme,openWeekFilme,grossFilme,cumWorldGross)
    else:
        return None
    
#a,b,c,d = pegaBudgetGross(soup3)
#print(a,'*',b,'*',c,'*',d)

#Function to get data from productions
#con2 = requests.get('http://www.imdb.com/title/tt6000478/?ref_=nm_flmg_act_2') 
#soup2 = BeautifulSoup(con2.content, "html.parser")

qtdFilmAct = 56 #Qtt of productions Denzel act... parameter ref_=nm_flmg_act_x indicates that on link 
film = {}
listaFilmes = []

for i in range(qtdFilmAct):
    
    #Loop to connect and get data from each production...
    con2 = requests.get(linksFilmes[i])  #http://www.imdb.com/title/tt2671706/?ref_=nm_flmg_act_3
    soup2 = BeautifulSoup(con2.content, "html.parser")
    con2.close()
    
    film['Titulo'] = titulosFilmes[i]
    film['Link'] = linksFilmes[i]
    film['Ano'] = pegaAno(soup2)
    film['Genero'] = pegaGenero(soup2)
    film['Duracao'] = pegaDuracao(soup2)
    film['Diretor'] = pegaDiretor(soup2)
    film['Roteirista'] = pegaRoteirista(soup2)
    film['Nota IMDB'] = pegaNotaIMDB(soup2)
    film['Qtd Rev IMDB'] = pegaReviewIMDB(soup2)
    film['Nota Metascore'] = pegaNotaMetaScore(soup2)
    film['Qtd Rev Metascore'] = pegaRevMetaScoreUser(soup2)
    #film['Budget'],film['Opening Weekend USA'],film['Gross USA'],film['Cumulative Worldwide Gross'] = pegaBudgetGross(soup2) #========> not ready yet  
    film['Sinopse'] = pegaSinopse(soup2)
    ##film['Qtd Rev Metascore Criticos'] = pegaRevMetaScoreCrit(soup2)  #========> not ready yet
    
    listaFilmes.append(film.copy()) #Adding dict on a list
    #print(titulosFilmes[i] + ', ' + linksFilmes[i])
#Creating new Dataframe
dfFilme = pd.DataFrame(listaFilmes, columns=[
    'Titulo','Ano','Genero','Duracao','Diretor','Roteirista','Nota IMDB','Qtd Rev IMDB','Nota Metascore','Qtd Rev Metascore',
    'Sinopse','Link'
])

#In Progress...
#dfFilme = pd.DataFrame(listaFilmes, columns=[
#    'Titulo','Ano','Genero','Duracao','Diretor','Roteirista','Nota IMDB','Qtd Rev IMDB','Nota Metascore','Qtd Rev Metascore',
#    'Budget','Opening Weekend USA','Gross USA','Cumulative Worldwide Gross','Sinopse','Link'
#])


#dfFilme
dfFilme.dtypes
#filling NaN values with zero
dfFilme['Ano'].fillna(value=0, inplace=True)
dfFilme['Duracao'].fillna(value=0, inplace=True)
dfFilme['Qtd Rev IMDB'].fillna(value=0, inplace=True)
dfFilme['Nota Metascore'].fillna(value=0, inplace=True)
dfFilme['Qtd Rev Metascore'].fillna(value=0, inplace=True)
#Treating dataframe values 
dfFilme['Ano'] = dfFilme['Ano'].astype(int)
dfFilme['Duracao'] = dfFilme['Duracao'].astype(int)
dfFilme['Qtd Rev IMDB'] = dfFilme['Qtd Rev IMDB'].astype(int)
dfFilme['Qtd Rev IMDB'] = dfFilme['Qtd Rev IMDB'].astype(int)
dfFilme['Nota Metascore'] = dfFilme['Nota Metascore'].astype(int)
dfFilme['Qtd Rev Metascore'] = dfFilme['Qtd Rev Metascore'].astype(int)

dfFilme.dtypes
dfFilme
#Saving Dataframe to a .csv file
dfFilme.to_csv('filmes_denzel.csv', sep=';', index=False)
