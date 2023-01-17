import pandas as pd
#Version1

#df = pd.read_html('https://docs.google.com/spreadsheets/d/e/2PACX-1vRRMS9u_TYyr-Ar3iTrBxTtw7H0lzRCZsJrH8nGlJrkebURo9iLvuPkmla9huyrgXsxtKfDIzBaLYqZ/pubhtml', encoding='utf8')[0]

#Version2

#df = pd.read_html('https://docs.google.com/spreadsheets/d/1J2dWp04si06O6s6C0agTpSZ5g56G2LP3qYuIEgW1ZnQ/edit?usp=sharing', encoding='utf8')[0]

df = pd.read_html('https://docs.google.com/spreadsheets/d/1J2dWp04si06O6s6C0agTpSZ5g56G2LP3qYuIEgW1ZnQ/pubhtml', encoding='utf8')[0]

#df.columns = ['Index','Case Number','Date','Year','Type','Country','Area','Location','Activity','Name','Sex','Age','Injury','Fatal (Y/N)','Time','Species','Investigator or Source']

#df = df.iloc[1:]

df.columns = ['Index','Case Number','Date','Year','Type','Country','Area','Location','Activity','Name','Sex','Age','Injury','Fatal (Y/N)','Time','Species','Investigator or Source','pdf', 'href formula', 'href','Case Number 2','Case Number 3', 'Original Order','Color Type']

df = df.iloc[1:]
df
len(df[df['Country'].isin(['UNKNOWN','NORTH ATLANTIC OCEAN','ATLANTIC OCEAN','INDIAN OCEAN','NORTH PACIFIC OCEAN','PACIFIC OCEAN','NORTH SEA','PALAU','PALESTINIAN TERRITORIES','BAY OF BENGAL','MIDATLANTICOCEAN','MID-PACIFCOCEAN','PERSIANGULF','REDSEA/INDIANOCEAN','SAN DOMINGO','SOLOMON ISLANDS/VANUATU','SOUTH ATLANTIC OCEAN','UNKOWN','TURKS & CAICOS','AFRICA','ASIA?','Coast of AFRICA','DIEGO GARCIA','EQUATORIAL GUINEA/CAMEROON','GULF OF ADEN','IRAN/IRAQ','MAYOTTE','MEDITERRANEAN SEA','NORTHERN ARABIAN SEA','RED SEA','SOUTH CHINA SEA','TASMAN SEA','THE BALKANS','WESTERN SAMOA'])])/len(df)
len(df[df['Color Type'] != df['Type']])
df['Country'].value_counts()
df['Country'] = df['Country'].str.strip()
countries = { # por clusters del open refine

             'RED SEA?': 'RED SEA',

             'INDIAN OCEAN?': 'INDIAN OCEAN',

             'Sierra Leone': 'SIERRA LEONE',

             'Seychelles': 'SEYCHELLES',

             'SUDAN?': 'SUDAN',

             'Fiji':'FIJI',

             'UNITED ARAB EMIRATES (UAE)':'UNITED ARAB EMIRATES',

             'ST. MAARTIN':'ST. MARTIN',

             # cosas a manopla

             'ADMIRALTY ISLANDS':'PAPUA NEW GUINEA',

             'ANDAMAN / NICOBAR ISLANDAS':'INDIA',

             'ANDAMAN ISLANDS':'INDIA',

             'AZORES':'PORTUGAL',

             'ENGLAND':'UNITED KINGDOM',

             'BRITISH ISLES':'UNITED KINGDOM',

             'CEYLON':'SRI LANKA',

             'CEYLON (SRI LANKA)':'SRI LANKA',

             'COLUMBIA':'COLOMBIA',

             'CRETE':'GREECE',

             'FALKLAND ISLANDS':'ARGENTINA', # ;)

             'GRAND CAYMAN':'CAYMAN ISLANDS',

             'IRELAND':'UNITED KINGDOM',

             'JAVA':'INDONESIA',

             'JOHNSTON ISLAND':'USA',

             'MALDIVE ISLANDS':'MALDIVES'

             }



df = df.copy()

df['Country'] = df['Country'].replace(countries)
# Cantidad de Incidentes por Tipo:

df.groupby(["Color Type"]).size().reset_index(name="Incidentes").sort_values(['Incidentes'], ascending=False)
# Para ver la cantidad de incidentes de cada tipo por país primero generamos una lista de todos los paises

# puse un sort para que ponga arriba los casos más frecuentes. Tomamos los primeros 10 

df.groupby(["Country", "Color Type"]).size().reset_index(name="Incidentes").sort_values(['Incidentes'], ascending=False).head(10)