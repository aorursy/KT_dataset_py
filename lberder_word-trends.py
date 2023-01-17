import pandas as pd

import re

import nltk

import nltk.stem

from nltk.tokenize import word_tokenize, sent_tokenize, PunktSentenceTokenizer

from nltk.corpus import stopwords

from string import punctuation



#to plot inside the document

%matplotlib inline

import matplotlib.pyplot as plt
debates = pd.read_csv("../input/un-general-debates.csv")
debates.head()
debates.describe(include="all")
debates[["year", "country"]].groupby("year").count().plot(kind="bar")
debates['text'] = debates['text'].str.lower().map(lambda x: re.sub('\W+',' ', x))
debates['token'] = debates['text'].apply(word_tokenize)
stop_words = set(stopwords.words('english'))

# I noticed that "'s" is not included in stopwords, while I think it doesn't bring much meaning in a text, so I'll add it to the set to remove from the cleaned tokens.

stop_words.add("'s")

stop_words.add("'")

stop_words.add("-")

stop_words.add("'")

debates['clean'] = debates['token'].apply(lambda x: [w for w in x if not w in stop_words and not w in punctuation])
stemmer = nltk.stem.PorterStemmer()
debates['stems'] = [[format(stemmer.stem(token)) for token in speech] for speech in debates['clean']]
debates.head()
all_per_year = debates.groupby('session').agg({'year': 'mean', 'clean': 'sum'})
for i, row in all_per_year.iterrows():

    sess = dict(nltk.FreqDist(row['clean']))

    sort_sess = sorted(sess.items(), key=lambda x: x[1], reverse=True)[0:25]

    plt.bar(range(len(sort_sess)), [val[1] for val in sort_sess], align='center')

    plt.xticks(range(len(sort_sess)), [val[0] for val in sort_sess])

    plt.xticks(rotation=90)

    plt.title("25 most used words in %d's session" % row['year'])

    plt.show()
freqs = {}

for i, speech in debates.iterrows():

    year = speech['year']

    for token in speech['stems']:

        if token not in freqs:

            freqs[token] = {"total_freq":1, year:1}

        else:

            freqs[token]["total_freq"] += 1

            if not freqs[token].get(year):

                freqs[token][year] = 1

            else:

                freqs[token][year] += 1
freqs_df = pd.DataFrame.from_dict(freqs, orient='index')

freqs_df['word'] = freqs_df.index
# Example of data for the stem of the word "peace"

freqs_df[freqs_df.index == "peac"]
new_cols = ["total_freq", "word"] + sorted(freqs_df.columns.tolist()[1:-1])

freqs_df = freqs_df[new_cols]



freqs_df = freqs_df.sort_values('total_freq', ascending=False)



freqs_df.head()
freqs_df.shape
freqs_df.tail(30)
freqs_df.iloc[0:5, 1:47].transpose().iloc[1:].plot(title="Most common words")
freqs_df[freqs_df['word'].isin(['peac', 'war', 'securit', 'cold', 'conflict', 'aggression'])].iloc[:, 1:47].transpose().iloc[1:].plot(title = "War and Peace")
freqs_df[freqs_df['word'].isin(['economi', 'wealth', 'crisi', 'growth', 'inflat', 'trade', 'poverti', 'rich', 'recession', 'income'])].iloc[:, 1:47].transpose().iloc[1:].plot(title="Economy")
freqs_df[freqs_df['word'].isin(['environment', 'sustain', 'green', 'energi', 'ecolog', 'warm', 'temperatur', 'pollution', 'planet'])].iloc[:, 1:47].transpose().iloc[1:].plot(title="Environment")
freqs_df[freqs_df['word'].isin(['peopl', 'inequaliti', 'refuge', 'humanitarian', 'immigr', 'freedom', 'right'])].iloc[:, 1:47].transpose().iloc[1:].plot(title="People")
freqs_df[freqs_df['word'].isin(['democraci', 'republ', 'dictat', 'sovereign', 'politic', 'vote'])].iloc[:, 1:47].transpose().iloc[1:].plot(title="Politics")
freqs_df[freqs_df['word'].isin(['violenc', 'unrest', 'genocid', 'atroc', 'kill', 'death'])].iloc[:, 1:47].transpose().iloc[1:].plot(title="Violence")
freqs_df[freqs_df['word'].isin(['terror'])].iloc[:, 1:47].transpose().iloc[1:].plot(title="Terrorism")
freqs_df[freqs_df['word'].isin(['health', 'disease', 'famin', 'drought', 'hiv', 'aid', 'research'])].iloc[:, 1:47].transpose().iloc[1:].plot(title="Health")
#IPython.load_extensions('usability/hide_input/main');

countries = dict((k, v.lower()) for k,v in {

    'AFG': 'Afghanistan', 

    'ALA': 'Aland Islands', 

    'ALB': 'Albania', 

    'DZA': 'Algeria', 

    'ASM': 'American Samoa', 

    'AND': 'Andorra', 

    'AGO': 'Angola', 

    'AIA': 'Anguilla', 

    'ATA': 'Antarctica', 

    'ATG': 'Antigua and Barbuda', 

    'ARG': 'Argentina', 

    'ARM': 'Armenia', 

    'ABW': 'Aruba', 

    'AUS': 'Australia', 

    'AUT': 'Austria', 

    'AZE': 'Azerbaijan', 

    'BHS': 'Bahamas', 

    'BHR': 'Bahrain', 

    'BGD': 'Bangladesh', 

    'BRB': 'Barbados', 

    'BLR': 'Belarus', 

    'BEL': 'Belgium', 

    'BLZ': 'Belize', 

    'BEN': 'Benin', 

    'BMU': 'Bermuda', 

    'BTN': 'Bhutan', 

    'BOL': 'Bolivia', 

    'BIH': 'Bosnia and Herzegovina', 

    'BWA': 'Botswana', 

    'BVT': 'Bouvet Island', 

    'BRA': 'Brazil', 

    'VGB': 'Virgin Islands', 

    'IOT': 'British Indian Ocean Territory', 

    'BRN': 'Brunei', 

    'BGR': 'Bulgaria', 

    'BFA': 'Burkina Faso', 

    'BDI': 'Burundi', 

    'KHM': 'Cambodia', 

    'CMR': 'Cameroon', 

    'CAN': 'Canada', 

    'CPV': 'Cape Verde', 

    'CYM': 'Cayman Islands', 

    'CAF': 'Central Africa', 

    'TCD': 'Chad', 

    'CHL': 'Chile', 

    'CHN': 'China', 

    'HKG': 'Hong Kong', 

    'MAC': 'Macao', 

    'CXR': 'Christmas Island', 

    'CCK': 'Cocos Islands', 

    'COL': 'Colombia', 

    'COM': 'Comoros', 

    'COG': 'Congo', 

    'COD': 'Democratic Republic of Congo', 

    'COK': 'Cook Islands', 

    'CRI': 'Costa Rica', 

    'CIV': "Cote d'Ivoire", 

    'HRV': 'Croatia', 

    'CUB': 'Cuba', 

    'CYP': 'Cyprus', 

    'CZE': 'Czech Republic', 

    'DNK': 'Denmark', 

    'DJI': 'Djibouti', 

    'DMA': 'Dominica', 

    'DOM': 'Dominican Republic', 

    'ECU': 'Ecuador', 

    'EGY': 'Egypt', 

    'SLV': 'El Salvador', 

    'GNQ': 'Equatorial Guinea', 

    'ERI': 'Eritrea', 

    'EST': 'Estonia', 

    'ETH': 'Ethiopia', 

    'FLK': 'Falkland', 

    'FRO': 'Faroe', 

    'FJI': 'Fiji', 

    'FIN': 'Finland', 

    'FRA': 'France', 

    'GUF': 'French Guiana', 

    'PYF': 'French Polynesia', 

    'ATF': 'French Southern Territories', 

    'GAB': 'Gabon', 

    'GMB': 'Gambia', 

    'GEO': 'Georgia', 

    'DEU': 'Germany', 

    'GHA': 'Ghana', 

    'GIB': 'Gibraltar', 

    'GRC': 'Greece', 

    'GRL': 'Greenland', 

    'GRD': 'Grenada', 

    'GLP': 'Guadeloupe', 

    'GUM': 'Guam', 

    'GTM': 'Guatemala', 

    'GGY': 'Guernsey', 

    'GIN': 'Guinea', 

    'GNB': 'Guinea-Bissau', 

    'GUY': 'Guyana', 

    'HTI': 'Haiti', 

    'HMD': 'Heard and Mcdonald Islands', 

    'VAT': 'Vatican', 

    'HND': 'Honduras', 

    'HUN': 'Hungary', 

    'ISL': 'Iceland', 

    'IND': 'India', 

    'IDN': 'Indonesia', 

    'IRN': 'Iran', 

    'IRQ': 'Iraq', 

    'IRL': 'Ireland', 

    'IMN': 'Isle of Man', 

    'ISR': 'Israel', 

    'ITA': 'Italy', 

    'JAM': 'Jamaica', 

    'JPN': 'Japan', 

    'JEY': 'Jersey', 

    'JOR': 'Jordan', 

    'KAZ': 'Kazakhstan', 

    'KEN': 'Kenya', 

    'KIR': 'Kiribati', 

    'PRK': 'North Korea', 

    'KOR': 'South Korea', 

    'KWT': 'Kuwait', 

    'KGZ': 'Kyrgyzstan', 

    'LAO': 'Lao', 

    'LVA': 'Latvia', 

    'LBN': 'Lebanon', 

    'LSO': 'Lesotho', 

    'LBR': 'Liberia', 

    'LBY': 'Libya', 

    'LIE': 'Liechtenstein', 

    'LTU': 'Lithuania', 

    'LUX': 'Luxembourg', 

    'MKD': 'Macedonia', 

    'MDG': 'Madagascar', 

    'MWI': 'Malawi', 

    'MYS': 'Malaysia', 

    'MDV': 'Maldives', 

    'MLI': 'Mali', 

    'MLT': 'Malta', 

    'MHL': 'Marshall Islands', 

    'MTQ': 'Martinique', 

    'MRT': 'Mauritania', 

    'MUS': 'Mauritius', 

    'MYT': 'Mayotte', 

    'MEX': 'Mexico', 

    'FSM': 'Micronesia', 

    'MDA': 'Moldova', 

    'MCO': 'Monaco', 

    'MNG': 'Mongolia', 

    'MNE': 'Montenegro', 

    'MSR': 'Montserrat', 

    'MAR': 'Morocco', 

    'MOZ': 'Mozambique', 

    'MMR': 'Myanmar', 

    'NAM': 'Namibia', 

    'NRU': 'Nauru', 

    'NPL': 'Nepal', 

    'NLD': 'Netherlands', 

    'ANT': 'Netherlands Antilles', 

    'NCL': 'New Caledonia', 

    'NZL': 'New Zealand', 

    'NIC': 'Nicaragua', 

    'NER': 'Niger', 

    'NGA': 'Nigeria', 

    'NIU': 'Niue', 

    'NFK': 'Norfolk Island', 

    'MNP': 'Northern Mariana Islands', 

    'NOR': 'Norway', 

    'OMN': 'Oman', 

    'PAK': 'Pakistan', 

    'PLW': 'Palau', 

    'PSE': 'Palestine', 

    'PAN': 'Panama', 

    'PNG': 'Papua New Guinea', 

    'PRY': 'Paraguay', 

    'PER': 'Peru', 

    'PHL': 'Philippines', 

    'PCN': 'Pitcairn', 

    'POL': 'Poland', 

    'PRT': 'Portugal', 

    'PRI': 'Puerto Rico', 

    'QAT': 'Qatar', 

    'REU': 'Reunion', 

    'ROU': 'Romania', 

    'RUS': 'Russia', 

    'RWA': 'Rwanda', 

    'BLM': 'Saint-Barthelemy', 

    'SHN': 'Saint Helena', 

    'KNA': 'Saint Kitts', 

    'LCA': 'Saint Lucia', 

    'MAF': 'Saint-Martin', 

    'SPM': 'Saint Pierre and Miquelon', 

    'VCT': 'Saint Vincent and Grenadines', 

    'WSM': 'Samoa', 

    'SMR': 'San Marino', 

    'STP': 'Sao Tome and Principe', 

    'SAU': 'Saudi Arabia', 

    'SEN': 'Senegal', 

    'SRB': 'Serbia', 

    'SYC': 'Seychelles', 

    'SLE': 'Sierra Leone', 

    'SGP': 'Singapore', 

    'SVK': 'Slovakia', 

    'SVN': 'Slovenia', 

    'SLB': 'Solomon Islands', 

    'SOM': 'Somalia', 

    'ZAF': 'South Africa', 

    'SGS': 'South Georgia and the South Sandwich Islands', 

    'SSD': 'South Sudan', 

    'ESP': 'Spain', 

    'LKA': 'Sri Lanka', 

    'SDN': 'Sudan', 

    'SUR': 'Suriname', 

    'SJM': 'Svalbard', 

    'SWZ': 'Swaziland', 

    'SWE': 'Sweden', 

    'CHE': 'Switzerland', 

    'SYR': 'Syria', 

    'TWN': 'Taiwan', 

    'TJK': 'Tajikistan', 

    'TZA': 'Tanzania', 

    'THA': 'Thailand', 

    'TLS': 'Timor', 

    'TGO': 'Togo', 

    'TKL': 'Tokelau', 

    'TON': 'Tonga', 

    'TTO': 'Trinidad', 

    'TUN': 'Tunisia', 

    'TUR': 'Turkey', 

    'TKM': 'Turkmenistan', 

    'TCA': 'Turks and Caicos Islands', 

    'TUV': 'Tuvalu', 

    'UGA': 'Uganda', 

    'UKR': 'Ukraine', 

    'ARE': 'United Arab Emirates', 

    'GBR': 'United Kingdom', 

    'USA': 'United States', 

    'UMI': 'US Minor Outlying Islands', 

    'URY': 'Uruguay', 

    'UZB': 'Uzbekistan', 

    'VUT': 'Vanuatu', 

    'VEN': 'Venezuela', 

    'VNM': 'Viet Nam', 

    'VIR': 'Virgin Islands', 

    'WLF': 'Wallis and Futuna', 

    'ESH': 'Western Sahara', 

    'YEM': 'Yemen', 

    'ZMB': 'Zambia', 

    'ZWE': 'Zimbabwe'

}.items())
debates['countries_mentioned'] = debates['token'].apply(lambda token: {x:token.count(x) for x in token if x in countries.values()})
country_mentions = pd.concat([debates[["year", "country"]],

                              debates['countries_mentioned'].apply(pd.Series)], axis=1).dropna(axis=1, how='all')

country_mentions['country'] = country_mentions['country'].apply(lambda x: countries.get(x))

country_mentions.head()
country_mentions_by_country = country_mentions.groupby("country")[country_mentions.columns[2:]].sum()
# First need to melt country_mentions_by_country to long form

sankey_data = country_mentions_by_country.unstack().reset_index()

sankey_data.columns = ['source','target','value']

sankey_data = sankey_data.sort_values(by='value', ascending=False)

sankey_data.head()
import plotly.plotly as py

from plotly.offline import download_plotlyjs, init_notebook_mode, plot,iplot

init_notebook_mode()



data = dict(

    type='sankey',

    domain = dict(

      x =  [0,1],

      y =  [0,1]

    ),

    orientation = "h",

    valueformat = ".0f",

    valuesuffix = "TWh"   

  )



layout =  dict(

    title = "Which countries mention which in the UN's General Assembly\n(1970-2015)",

    font = dict(

      size = 10

    )

)



data_trace = dict(

    type='sankey',

    width = 1118,

    height = 772,

    domain = dict(

      x =  [0,1],

      y =  [0,1]

    ),

    orientation = "h",

    valueformat = ".0f",

    valuesuffix = "TWh",

    node = dict(

      pad = 15,

      thickness = 15,

      line = dict(

        color = "black",

        width = 0.5

      ),

      label =  sankey_data['target'],

      color =  "black"

  ),

    link = dict(

      source =  sankey_data['source'],

      target =  sankey_data['target'],

      value =  sankey_data['value'],

      label =  sankey_data['source']

  ))



fig = dict(data=[data_trace], layout=layout)

iplot(fig, validate=False)
from ipysankeywidget import SankeyWidget



sankey_data.columns = ['source','target','value']

sankey_data = sankey_data.sort_values(by='value', ascending=False)

links=sankey_data[0:200].dropna()[['source','target','value']].to_dict(orient='records')



SankeyWidget(value={'links': links},

             width=800, height=800,margins=dict(top=0, bottom=0))