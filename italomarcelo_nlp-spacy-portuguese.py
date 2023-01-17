# to update spacy. 

# I'll use portuguese spacy than I need  update it 

!pip install -U spacy

import spacy
# install Portuguese model language

import spacy.cli

spacy.cli.download("pt_core_news_sm")
# from spacy.lang.pt import Portuguese

# nlp = Portuguese()

# or below

nlp = spacy.load('pt_core_news_sm')
doc = nlp(u'Meu nome é Marcelo.')
doc
print([d for d in doc])
# tokenizing doc object

[token.text for token in doc]
# attribute, example: text GOING, lemma: numbers, lemma: GO

for token in doc:

  print(token.text, token.lemma, token.lemma_)
# viewing sentece's structure

from spacy import displacy

from IPython.display import SVG, display

def showSVG(s):

  display(SVG(s))



graph01 = displacy.render(doc)

showSVG(graph01)
# IS ALPHA, IS PUNCT, IS LOWER.. etc

for token in doc:

  print('TXT:',token.text, ' -- é caracter?:',token.is_alpha, 

        'é pontuacao?:', token.is_punct, 'está em caixa baixa?:', token.is_lower)
# similarity between two words

# I did metrics with all token's position

d2 = [token for token in nlp('O doce de batata doce')]

for a in d2:

  for b in d2:

    print('A similaridade entre "',a, '" e "',b, '" : ', a.similarity(b))
'Doc: ', doc
# structure POS (Part-of-Speech)

[(token.text, token.pos, token.pos_) for token in doc]
# return tokens that pos_ is VERB

[(token.text, token.lemma_, token.pos_) for token in doc if token.pos_ == 'VERB']
[(token.text, token.lemma_, token.pos_) for token in nlp('Calça a bota.') if token.pos_ == 'VERB']
[(token.text, token.lemma_, token.pos_) for token in nlp('Bota a calça.') if token.pos_ == 'VERB']
doc = nlp('Salvador parece ser uma grande cidade. Recife aparenta ser grande tambem. Aguas de Lindoya, nao')



# TOKENIZING

for token in doc:

    # check if token POS is PROPN

    if token.pos_ == "PROPN":

        # and check if token POS [position + 1] is VERB

        if doc[token.i + 1].pos_ == "VERB":

            print(f"PROPN [Nome Proprio] {token.text} ANTES DE UM VERBO: {doc[token.i + 1].lemma_}")
# portuguese model is incomplete

doc = nlp('Donald curte a praia de Miami')

[(e, e.label_) for e in doc.ents]
nlpEn = spacy.load("en_core_web_sm")

[(e, e.label_) for e in nlpEn('Donald enjoys Miami beach').ents]
# portuguese, Reino Unido = UK

dp = nlp('Apple está pensando em comprar uma startup do Reino Unido por $1 bilhão')

[(e, e.label_) for e in dp.ents]

# english

de = nlpEn('Apple is looking at buying UK startup for $1 billion')

[(e, e.label_) for e in de.ents]
from spacy.matcher import Matcher
doc = nlp('Problemas no iOS: iphones com iOS 14 continuam com problemas na bateria e Wi-Fi')
matcher = Matcher(nlp.vocab)

# pattern with word iOs

pattern = [{'TEXT': 'iOS'}]

# add pattern to the matcher and apply to the doc

matcher.add('MEU_PATTERN_IOS', None, pattern)

matches = matcher(doc)

print('Total de matches encontrados:', len(matches))



for match, start, end in matches:

  print('Match:', doc[start:end].text)

doc = nlp(

           'Problemas no iOS: '

           '- iphones com iOS 14 continuam com problemas na bateria e Wi-Fi '

           '- Arrependido de ter baixado o iOS 14? Aprenda a voltar ao iOS 13 '

         )
# goal: match will return only matchs with version

matcher = Matcher(nlp.vocab)

# pattern with word iOs

pattern = [{'TEXT': 'iOS'}, {'IS_DIGIT': True}]

# add pattern to the matcher and apply to the doc

matcher.add('MEU_PATTERN_IOS_COM_VERSAO', None, pattern)

matches = matcher(doc)

print('Total de matches encontrados:', len(matches))



for match, start, end in matches:

  print('Match:', doc[start:end].text)
tweets = [

          'Sem dúvida o Joaquim, #selecao 8',

          'Neymar 10 é o melhor em campo #selecao',

          'Neymar!!! #selecao 10',

          'Vai Brazil #selecao 10',

          'Melhor ataque é a defesa #selecao 3',

          'Com certeza, Zagallo ... #selecao 13',       

]

txt = ' '.join(tweets)

doc = nlp(txt)



# goal: match will return only matchs with version

matcher = Matcher(nlp.vocab)

# pattern with word iOs

pattern = [{'TEXT': '#'}, {'TEXT': 'selecao'}, {'POS': 'NUM'}]

# add pattern to the matcher and apply to the doc

matcher.add('MEU_PATTERN_IOS_COM_VERSAO', None, pattern)

matches = matcher(doc)

print('Total de matches encontrados:', len(matches))



for match, start, end in matches:

  print('Match:', doc[start:end].text)
# HASH

doc = nlp('Eu tenho a força')

eu = nlp.vocab.strings['Eu']

print('Hash:',eu)

print('Texto:',nlp.vocab.strings[eu])
dochash = [nlp.vocab.strings[token.text] for token in doc]

print('Mensagem Hash:', dochash)  
dochash = [nlp.vocab.strings[token] for token in dochash]

print(*dochash, sep=' ')
from spacy.tokens import Doc



palavras = ['Hello', 'SPACY', '!']

espacos = [True, False, False]

doc = Doc(nlp.vocab, words=palavras, spaces=espacos)

doc.text
palavras = ['Eita', ',', 'que', 'pegadinha', '.']

espacos = [False, True, True, False, False]

doc = Doc(nlp.vocab, words=palavras, spaces=espacos)

doc.text
from spacy.tokens import Span



# create doc with sentence *Melhor piloto: Lewis Hamilton*

palavras = ['Melhor', 'piloto', ':', 'Lewis', 'Hamilton']

espacos = [True, False, True, True, False]

doc = Doc(nlp.vocab, words=palavras, spaces=espacos)

doc.text
# create span from the doc and assign it with label PERSON

span = Span(doc, 3, 5, label='PERSON')

span.text, span.label_
[(ent.text, ent.label_) for ent in doc.ents]
doc.ents = [span]

[(ent.text, ent.label_) for ent in doc.ents]
doc = nlp('Bananas de Pijamas')

vetorBanana = doc[0].vector

doc[0].has_vector, doc[0].vector_norm, doc[0].is_oov
vetorBanana
txt1 = nlp('doce de banana')

doc.similarity(txt1)
txt1 = nlp('Bob Esponja')

doc.similarity(txt1)
txt1 = nlp('banana banana')

t1, t2 = txt1[0], txt1[1]

t1.similarity(t2)
doc = nlp('doce de batata doce')

for t1 in doc:

  for t2 in doc:

    print(t1, t2, t1.similarity(t2))
nlp.pipe_names, nlp.pipeline
def exibeTamanho(doc):

  print('Tamanho do doc: ', len(doc))

  return doc



nlp.pipe_names
# add exibeTamanho component in first position

# if you need remove pipe, use nlp.remove_extension(componentName)

nlp.add_pipe(exibeTamanho, first=True)

nlp.pipe_names
doc = nlp('Customizando componentes')
# remove pipeline component

nlp.remove_pipe('exibeTamanho')
from spacy.matcher import PhraseMatcher
frase = 'Rafael Anderson Thiago Juliano Fabio'.split(' ')

amigo_pattern = list(nlp.pipe(frase))
amigo_pattern
matcher = PhraseMatcher(nlp.vocab)

matcher.add('AMIGO', None, *amigo_pattern)

# if you need remove matcher, use matcher.remove(componentName)
def amigoComponente(doc):

  matches = matcher(doc) # apply matcher to the doc

  spans = [Span(doc, start, end, label='AMIGO') for mId, start, end in matches] # assign label for each span

  doc.ents = spans # changes entities of the doc to matched spans

  return doc

nlp.pipe_names
# add new component after ner component 

# if you need remove pipe, use nlp.remove_pipe(componentName)

nlp.add_pipe(amigoComponente, after='ner')

nlp.pipe_names
doc = nlp('Estavam no churrasco: Joao, Maria, Pedro, Rafael, Carlos, Antonio e o Fabio')

[(ent.text, ent.label_) for ent in doc.ents]
# Removing pipe

nlp.remove_pipe('amigoComponente')
from spacy.tokens import Token
# register the token extension attribute is_animal.

Token.set_extension('is_animal', default=False)



# process the text and set attribute True in gato

doc = nlp('Meu gato é amarelo')

doc[1]._.is_animal = True
[(token.text, token._.is_animal) for token in doc]
# using getter fuction to count letters in each token

# if you need remove extension, use doc.remove_extension('hasNumber')



def getLetras(token):

  return f'Nro. de Caracteres: {len(token.text)}'



Token.set_extension('countLetters', getter=getLetras)



doc = nlp('Eu acordei cedo')
for token in doc:

  print(token.text, '-',token._.countLetters, end='\n')
# detect if token is number

# if you need remove extension, use doc.remove_extension('hasNumber')



doc = nlp('Copa do Mundo 2018')



# using getter

def get_hasNumber(doc):

  return any(token.like_num for token in doc)



doc.set_extension('hasNumber', getter=get_hasNumber)



print('Existe token numerico?', doc._.hasNumber)
# Using method

def codingHtml(span, tag):

    # get span content and convert to html content

    return f"<{tag}>{span.text}</ {tag}>"

try:

  Span.remove_extension('codingHtml')

except:

  pass

# registering method extension codingHtml

Span.set_extension('codingHtml', method=codingHtml)



# Process the text and call the to_html method on the span with the tag bold 'b'

doc = nlp("Eu amo o meu Brasil")

span = doc[3:5]

print(span._.codingHtml('b'))
# DATASET Countries and capitals (admin)

DATASET = {'Afghanistan': 'Kabul', 'Åland Islands': 'Mariehamn', 'Albania': 'Tirana', 'Algeria': 'Algiers', 'American Samoa': 'Pago Pago', 'Andorra': 'Andorra la Vella', 'Angola': 'Luanda', 'Anguilla': 'The Valley', 'Antarctica': '', 'Antigua and Barbuda': "Saint John's", 'Argentina': 'Buenos Aires', 'Armenia': 'Yerevan', 'Aruba': 'Oranjestad', 'Australia': 'Canberra', 'Austria': 'Vienna', 'Azerbaijan': 'Baku', 'Bahamas': 'Nassau', 'Bahrain': 'Manama', 'Bangladesh': 'Dhaka', 'Barbados': 'Bridgetown', 'Belarus': 'Minsk', 'Belgium': 'Brussels', 'Belize': 'Belmopan', 'Benin': 'Porto-Novo', 'Bermuda': 'Hamilton', 'Bhutan': 'Thimphu', 'Bolivia (Plurinational State of)': 'Sucre', 'Bonaire, Sint Eustatius and Saba': 'Kralendijk', 'Bosnia and Herzegovina': 'Sarajevo', 'Botswana': 'Gaborone', 'Bouvet Island': '', 'Brazil': 'Brasília', 'British Indian Ocean Territory': 'Diego Garcia', 'United States Minor Outlying Islands': '', 'Virgin Islands (British)': 'Road Town', 'Virgin Islands (U.S.)': 'Charlotte Amalie', 'Brunei Darussalam': 'Bandar Seri Begawan', 'Bulgaria': 'Sofia', 'Burkina Faso': 'Ouagadougou', 'Burundi': 'Bujumbura', 'Cambodia': 'Phnom Penh', 'Cameroon': 'Yaoundé', 'Canada': 'Ottawa', 'Cabo Verde': 'Praia', 'Cayman Islands': 'George Town', 'Central African Republic': 'Bangui', 'Chad': "N'Djamena", 'Chile': 'Santiago', 'China': 'Beijing', 'Christmas Island': 'Flying Fish Cove', 'Cocos (Keeling) Islands': 'West Island', 'Colombia': 'Bogotá', 'Comoros': 'Moroni', 'Congo': 'Brazzaville', 'Congo (Democratic Republic of the)': 'Kinshasa', 'Cook Islands': 'Avarua', 'Costa Rica': 'San José', 'Croatia': 'Zagreb', 'Cuba': 'Havana', 'Curaçao': 'Willemstad', 'Cyprus': 'Nicosia', 'Czech Republic': 'Prague', 'Denmark': 'Copenhagen', 'Djibouti': 'Djibouti', 'Dominica': 'Roseau', 'Dominican Republic': 'Santo Domingo', 'Ecuador': 'Quito', 'Egypt': 'Cairo', 'El Salvador': 'San Salvador', 'Equatorial Guinea': 'Malabo', 'Eritrea': 'Asmara', 'Estonia': 'Tallinn', 'Ethiopia': 'Addis Ababa', 'Falkland Islands (Malvinas)': 'Stanley', 'Faroe Islands': 'Tórshavn', 'Fiji': 'Suva', 'Finland': 'Helsinki', 'France': 'Paris', 'French Guiana': 'Cayenne', 'French Polynesia': 'Papeetē', 'French Southern Territories': 'Port-aux-Français', 'Gabon': 'Libreville', 'Gambia': 'Banjul', 'Georgia': 'Tbilisi', 'Germany': 'Berlin', 'Ghana': 'Accra', 'Gibraltar': 'Gibraltar', 'Greece': 'Athens', 'Greenland': 'Nuuk', 'Grenada': "St. George's", 'Guadeloupe': 'Basse-Terre', 'Guam': 'Hagåtña', 'Guatemala': 'Guatemala City', 'Guernsey': 'St. Peter Port', 'Guinea': 'Conakry', 'Guinea-Bissau': 'Bissau', 'Guyana': 'Georgetown', 'Haiti': 'Port-au-Prince', 'Heard Island and McDonald Islands': '', 'Holy See': 'Rome', 'Honduras': 'Tegucigalpa', 'Hong Kong': 'City of Victoria', 'Hungary': 'Budapest', 'Iceland': 'Reykjavík', 'India': 'New Delhi', 'Indonesia': 'Jakarta', "Côte d'Ivoire": 'Yamoussoukro', 'Iran (Islamic Republic of)': 'Tehran', 'Iraq': 'Baghdad', 'Ireland': 'Dublin', 'Isle of Man': 'Douglas', 'Israel': 'Jerusalem', 'Italy': 'Rome', 'Jamaica': 'Kingston', 'Japan': 'Tokyo', 'Jersey': 'Saint Helier', 'Jordan': 'Amman', 'Kazakhstan': 'Astana', 'Kenya': 'Nairobi', 'Kiribati': 'South Tarawa', 'Kuwait': 'Kuwait City', 'Kyrgyzstan': 'Bishkek', "Lao People's Democratic Republic": 'Vientiane', 'Latvia': 'Riga', 'Lebanon': 'Beirut', 'Lesotho': 'Maseru', 'Liberia': 'Monrovia', 'Libya': 'Tripoli', 'Liechtenstein': 'Vaduz', 'Lithuania': 'Vilnius', 'Luxembourg': 'Luxembourg', 'Macao': '', 'Macedonia (the former Yugoslav Republic of)': 'Skopje', 'Madagascar': 'Antananarivo', 'Malawi': 'Lilongwe', 'Malaysia': 'Kuala Lumpur', 'Maldives': 'Malé', 'Mali': 'Bamako', 'Malta': 'Valletta', 'Marshall Islands': 'Majuro', 'Martinique': 'Fort-de-France', 'Mauritania': 'Nouakchott', 'Mauritius': 'Port Louis', 'Mayotte': 'Mamoudzou', 'Mexico': 'Mexico City', 'Micronesia (Federated States of)': 'Palikir', 'Moldova (Republic of)': 'Chișinău', 'Monaco': 'Monaco', 'Mongolia': 'Ulan Bator', 'Montenegro': 'Podgorica', 'Montserrat': 'Plymouth', 'Morocco': 'Rabat', 'Mozambique': 'Maputo', 'Myanmar': 'Naypyidaw', 'Namibia': 'Windhoek', 'Nauru': 'Yaren', 'Nepal': 'Kathmandu', 'Netherlands': 'Amsterdam', 'New Caledonia': 'Nouméa', 'New Zealand': 'Wellington', 'Nicaragua': 'Managua', 'Niger': 'Niamey', 'Nigeria': 'Abuja', 'Niue': 'Alofi', 'Norfolk Island': 'Kingston', "Korea (Democratic People's Republic of)": 'Pyongyang', 'Northern Mariana Islands': 'Saipan', 'Norway': 'Oslo', 'Oman': 'Muscat', 'Pakistan': 'Islamabad', 'Palau': 'Ngerulmud', 'Palestine, State of': 'Ramallah', 'Panama': 'Panama City', 'Papua New Guinea': 'Port Moresby', 'Paraguay': 'Asunción', 'Peru': 'Lima', 'Philippines': 'Manila', 'Pitcairn': 'Adamstown', 'Poland': 'Warsaw', 'Portugal': 'Lisbon', 'Puerto Rico': 'San Juan', 'Qatar': 'Doha', 'Republic of Kosovo': 'Pristina', 'Réunion': 'Saint-Denis', 'Romania': 'Bucharest', 'Russian Federation': 'Moscow', 'Rwanda': 'Kigali', 'Saint Barthélemy': 'Gustavia', 'Saint Helena, Ascension and Tristan da Cunha': 'Jamestown', 'Saint Kitts and Nevis': 'Basseterre', 'Saint Lucia': 'Castries', 'Saint Martin (French part)': 'Marigot', 'Saint Pierre and Miquelon': 'Saint-Pierre', 'Saint Vincent and the Grenadines': 'Kingstown', 'Samoa': 'Apia', 'San Marino': 'City of San Marino', 'Sao Tome and Principe': 'São Tomé', 'Saudi Arabia': 'Riyadh', 'Senegal': 'Dakar', 'Serbia': 'Belgrade', 'Seychelles': 'Victoria', 'Sierra Leone': 'Freetown', 'Singapore': 'Singapore', 'Sint Maarten (Dutch part)': 'Philipsburg', 'Slovakia': 'Bratislava', 'Slovenia': 'Ljubljana', 'Solomon Islands': 'Honiara', 'Somalia': 'Mogadishu', 'South Africa': 'Pretoria', 'South Georgia and the South Sandwich Islands': 'King Edward Point', 'Korea (Republic of)': 'Seoul', 'South Sudan': 'Juba', 'Spain': 'Madrid', 'Sri Lanka': 'Colombo', 'Sudan': 'Khartoum', 'Suriname': 'Paramaribo', 'Svalbard and Jan Mayen': 'Longyearbyen', 'Swaziland': 'Lobamba', 'Sweden': 'Stockholm', 'Switzerland': 'Bern', 'Syrian Arab Republic': 'Damascus', 'Taiwan': 'Taipei', 'Tajikistan': 'Dushanbe', 'Tanzania, United Republic of': 'Dodoma', 'Thailand': 'Bangkok', 'Timor-Leste': 'Dili', 'Togo': 'Lomé', 'Tokelau': 'Fakaofo', 'Tonga': "Nuku'alofa", 'Trinidad and Tobago': 'Port of Spain', 'Tunisia': 'Tunis', 'Turkey': 'Ankara', 'Turkmenistan': 'Ashgabat', 'Turks and Caicos Islands': 'Cockburn Town', 'Tuvalu': 'Funafuti', 'Uganda': 'Kampala', 'Ukraine': 'Kiev', 'United Arab Emirates': 'Abu Dhabi', 'United Kingdom of Great Britain and Northern Ireland': 'London', 'United States of America': 'Washington, D.C.', 'Uruguay': 'Montevideo', 'Uzbekistan': 'Tashkent', 'Vanuatu': 'Port Vila', 'Venezuela (Bolivarian Republic of)': 'Caracas', 'Viet Nam': 'Hanoi', 'Wallis and Futuna': 'Mata-Utu', 'Western Sahara': 'El Aaiún', 'Yemen': "Sana'a", 'Zambia': 'Lusaka', 'Zimbabwe': 'Harare'}

countries = []

capitals = []

for i in DATASET:

  countries.append(i)

  capitals.append(DATASET[i])

import pandas as pd

df=pd.DataFrame()

df['Country'] = countries

df['Capital'] = capitals
nlp.pipe_names
matcher = PhraseMatcher(nlp.vocab)

try:

  matcher.remove('PAISES')

except:

  pass

matcher.add("PAISES", None, *list(nlp.pipe([i for i in DATASET.keys()])))





def getPaises(doc):

    # Creating Span with the label GPE

    matches = matcher(doc)

    doc.ents = [Span(doc, start, end, label="GPE") for match_id, start, end in matches]

    return doc





# Removing component if it exists

try:

  nlp.remove_pipe('getPaises')

except:

  pass

# Add component 

nlp.add_pipe(getPaises)

print(nlp.pipe_names)



capitalCity = lambda span: DATASET.get(span.text)



# Removing if exists and create new extension

try:

  Span.remove_extension('capital')

except:

  pass

Span.set_extension("capital", getter=capitalCity)



# Processing the text

doc = nlp("Capitais dos paises: Brazil, Argentina e Chile")

print([(ent.text, ent.label_, ent._.capital) for ent in doc.ents])
Textos = [

  "Grêmio classificado, Inter encaminhado. Mas dupla gaúcha deve futebol. Mas dupla gaúcha deve futebol. Rodrigues comemora o segundo gol do Grêmio na vitória sobre a Universidad Católica - Alexandre Schneider.16 horas atrás ",

  "Confira jogos de futebol de hoje, terça-feira, 29 de setembro (29/09). Algumas partidas estão suspensas por causa da pandemia do COVID ",

  "Palmeiras, SPFC e Flamengo na Liberta; os jogos de quarta e onde assistir. Os jogos de futebol desta quarta-feira (30) se iniciam na parte da tarde com o Campeonato Espanhol: Atlético e Real Madrid entram em campo. ",

  "Fifa lança manual para desenvolver futebol feminino no mundo ",

  "O sucesso do futebol feminino não para de aumentar. ",

  "CBF desmembra rodadas de 16 a 20 do Brasileirão Assaí 2020 - Confederação Brasileira de Futebol ",

  "Foi divulgada, nesta terça-feira (29), pela Diretoria de Competições, a Tabela Detalhada das rodadas 16 a 20 do Brasileirão Assaí 2020. ",

  "Com 'sentimento estranho', Gabriel Menino dá até breve ao Palmeiras ",

  "A magia do futebol brasileiro está no talento e na competitividade. ... nacionais, em campeonatos masculino e feminino, no ano de 2020. ",

  "Treinadores são muito individualistas, não existe classe "

  "Com vitória da LDU, do que SPFC precisa para se classificar na Libertadores ",

  "Grêmio classificado, Inter encaminhado. Mas dupla gaúcha deve futebol. Mas dupla gaúcha deve futebol. Rodrigues comemora o segundo gol do Grêmio na vitória sobre a Universidad Católica - Alexandre Schneider.16 horas atrás ",

  "Confira jogos de futebol de hoje, terça-feira, 29 de setembro (29/09). Algumas partidas estão suspensas por causa da pandemia do COVID ",

  "Palmeiras, SPFC e Flamengo na Liberta; os jogos de quarta e onde assistir. Os jogos de futebol desta quarta-feira (30) se iniciam na parte da tarde com o Campeonato Espanhol: Atlético e Real Madrid entram em campo. ",

  "Fifa lança manual para desenvolver futebol feminino no mundo ",

  "O sucesso do futebol feminino não para de aumentar. ",

  "CBF desmembra rodadas de 16 a 20 do Brasileirão Assaí 2020 - Confederação Brasileira de Futebol ",

  "Foi divulgada, nesta terça-feira (29), pela Diretoria de Competições, a Tabela Detalhada das rodadas 16 a 20 do Brasileirão Assaí 2020. ",

  "Com 'sentimento estranho', Gabriel Menino dá até breve ao Palmeiras ",

  "A magia do futebol brasileiro está no talento e na competitividade. ... nacionais, em campeonatos masculino e feminino, no ano de 2020. ",

  "Treinadores são muito individualistas, não existe classe "

  "Com vitória da LDU, do que SPFC precisa para se classificar na Libertadores ",

  "Grêmio classificado, Inter encaminhado. Mas dupla gaúcha deve futebol. Mas dupla gaúcha deve futebol. Rodrigues comemora o segundo gol do Grêmio na vitória sobre a Universidad Católica - Alexandre Schneider.16 horas atrás ",

  "Confira jogos de futebol de hoje, terça-feira, 29 de setembro (29/09). Algumas partidas estão suspensas por causa da pandemia do COVID ",

  "Palmeiras, SPFC e Flamengo na Liberta; os jogos de quarta e onde assistir. Os jogos de futebol desta quarta-feira (30) se iniciam na parte da tarde com o Campeonato Espanhol: Atlético e Real Madrid entram em campo. ",

  "Fifa lança manual para desenvolver futebol feminino no mundo ",

  "O sucesso do futebol feminino não para de aumentar. ",

  "CBF desmembra rodadas de 16 a 20 do Brasileirão Assaí 2020 - Confederação Brasileira de Futebol ",

  "Foi divulgada, nesta terça-feira (29), pela Diretoria de Competições, a Tabela Detalhada das rodadas 16 a 20 do Brasileirão Assaí 2020. ",

  "Com 'sentimento estranho', Gabriel Menino dá até breve ao Palmeiras ",

  "A magia do futebol brasileiro está no talento e na competitividade. ... nacionais, em campeonatos masculino e feminino, no ano de 2020. ",

  "Treinadores são muito individualistas, não existe classe "

  "Com vitória da LDU, do que SPFC precisa para se classificar na Libertadores ",

  "Grêmio classificado, Inter encaminhado. Mas dupla gaúcha deve futebol. Mas dupla gaúcha deve futebol. Rodrigues comemora o segundo gol do Grêmio na vitória sobre a Universidad Católica - Alexandre Schneider.16 horas atrás ",

  "Confira jogos de futebol de hoje, terça-feira, 29 de setembro (29/09). Algumas partidas estão suspensas por causa da pandemia do COVID ",

  "Palmeiras, SPFC e Flamengo na Liberta; os jogos de quarta e onde assistir. Os jogos de futebol desta quarta-feira (30) se iniciam na parte da tarde com o Campeonato Espanhol: Atlético e Real Madrid entram em campo. ",

  "Fifa lança manual para desenvolver futebol feminino no mundo ",

  "O sucesso do futebol feminino não para de aumentar. ",

  "CBF desmembra rodadas de 16 a 20 do Brasileirão Assaí 2020 - Confederação Brasileira de Futebol ",

  "Foi divulgada, nesta terça-feira (29), pela Diretoria de Competições, a Tabela Detalhada das rodadas 16 a 20 do Brasileirão Assaí 2020. ",

  "Com 'sentimento estranho', Gabriel Menino dá até breve ao Palmeiras ",

  "A magia do futebol brasileiro está no talento e na competitividade. ... nacionais, em campeonatos masculino e feminino, no ano de 2020. ",

  "Treinadores são muito individualistas, não existe classe "

  "Com vitória da LDU, do que SPFC precisa para se classificar na Libertadores ",

  "Grêmio classificado, Inter encaminhado. Mas dupla gaúcha deve futebol. Mas dupla gaúcha deve futebol. Rodrigues comemora o segundo gol do Grêmio na vitória sobre a Universidad Católica - Alexandre Schneider.16 horas atrás ",

  "Confira jogos de futebol de hoje, terça-feira, 29 de setembro (29/09). Algumas partidas estão suspensas por causa da pandemia do COVID ",

  "Palmeiras, SPFC e Flamengo na Liberta; os jogos de quarta e onde assistir. Os jogos de futebol desta quarta-feira (30) se iniciam na parte da tarde com o Campeonato Espanhol: Atlético e Real Madrid entram em campo. ",

  "Fifa lança manual para desenvolver futebol feminino no mundo ",

  "O sucesso do futebol feminino não para de aumentar. ",

  "CBF desmembra rodadas de 16 a 20 do Brasileirão Assaí 2020 - Confederação Brasileira de Futebol ",

  "Foi divulgada, nesta terça-feira (29), pela Diretoria de Competições, a Tabela Detalhada das rodadas 16 a 20 do Brasileirão Assaí 2020. ",

  "Com 'sentimento estranho', Gabriel Menino dá até breve ao Palmeiras ",

  "A magia do futebol brasileiro está no talento e na competitividade. ... nacionais, em campeonatos masculino e feminino, no ano de 2020. ",

  "Treinadores são muito individualistas, não existe classe "

  "Com vitória da LDU, do que SPFC precisa para se classificar na Libertadores ",

  "Grêmio classificado, Inter encaminhado. Mas dupla gaúcha deve futebol. Mas dupla gaúcha deve futebol. Rodrigues comemora o segundo gol do Grêmio na vitória sobre a Universidad Católica - Alexandre Schneider.16 horas atrás ",

  "Confira jogos de futebol de hoje, terça-feira, 29 de setembro (29/09). Algumas partidas estão suspensas por causa da pandemia do COVID ",

  "Palmeiras, SPFC e Flamengo na Liberta; os jogos de quarta e onde assistir. Os jogos de futebol desta quarta-feira (30) se iniciam na parte da tarde com o Campeonato Espanhol: Atlético e Real Madrid entram em campo. ",

  "Fifa lança manual para desenvolver futebol feminino no mundo ",

  "O sucesso do futebol feminino não para de aumentar. ",

  "CBF desmembra rodadas de 16 a 20 do Brasileirão Assaí 2020 - Confederação Brasileira de Futebol ",

  "Foi divulgada, nesta terça-feira (29), pela Diretoria de Competições, a Tabela Detalhada das rodadas 16 a 20 do Brasileirão Assaí 2020. ",

  "Com 'sentimento estranho', Gabriel Menino dá até breve ao Palmeiras ",

  "A magia do futebol brasileiro está no talento e na competitividade. ... nacionais, em campeonatos masculino e feminino, no ano de 2020. ",

  "Treinadores são muito individualistas, não existe classe "

  "Com vitória da LDU, do que SPFC precisa para se classificar na Libertadores ",

  

]

# testing performance



from datetime import datetime



# here, Im calling nlp on each text

a=datetime.now()

print(a)

for text in Textos:

    [token.text for token in nlp(text) if token.pos_ == "ADJ"]

b=datetime.now()

print(b)

print('==>',b-a)
# here Im using nlp.pipe. Faster and better to processing large volumes of text

a=datetime.now()

print(a)

for doc in nlp.pipe(Textos):

    [token.text for token in doc if token.pos_ == "ADJ"]

b=datetime.now()

print(b)

print('==>',b-a)
data = [

        ('TEXTO 01 TEXTO 01 TEXTO 01 TEXTO 01 TEXTO 01 TEXTO 01 TEXTO 01 ', 

         {'livroId': 1, 'titulo': 'Livro 1', 'armarioId': 45, 'autor': 'Monteiro Lobato', 'qtdePaginas': 100}),

        ('texto 02 texto 02 texto 02 texto 02 texto 02 texto 02 texto 02 texto 02 texto 02 ', 

         {'livroId': 2, 'titulo': 'Livro 2', 'armarioId': 3, 'autor': 'Clarisce L', 'qtdePaginas': 70}),

]

livros = nlp.pipe(data, as_tuples=True)



[(context['titulo'], context['livroId'], context['armarioId']) for doc, context in livros]
Doc.set_extension('livroId', default=None)

Doc.set_extension('titulo', default=None)

Doc.set_extension('armarioId', default=None)

Doc.set_extension('autor', default=None)

Doc.set_extension('qtdePaginas', default=None)



# to remove extension execute lines below

# Doc.remove_extension('livroId')

# Doc.remove_extension('titulo')

# Doc.remove_extension('armarioId')

# Doc.remove_extension('autor')

# Doc.remove_extension('qtdePaginas')
livros = nlp.pipe(data, as_tuples=True)

for doc, context in livros:

  doc._.livroId = context['livroId']

  doc._.titulo = context['titulo'] 

  doc._.armarioId = context['armarioId'] 

  doc._.autor = context['autor']

  doc._.qtdePaginas = context['qtdePaginas']

  print(f"Título: {doc._.titulo} \nAutor: {doc._.autor}\nQtde de Pags:{doc._.qtdePaginas}")

  print(f'Resumo:', doc.text, '\n')