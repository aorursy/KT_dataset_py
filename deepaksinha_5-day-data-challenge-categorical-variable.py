import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import re

import datetime



import matplotlib.pyplot as plt

import seaborn as sns

from wordcloud import WordCloud, STOPWORDS



# Input data files are available in the "../input/" directory.



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



%matplotlib inline
# Reading data

ufo_df = pd.read_csv('../input/scrubbed.csv', encoding="ISO-8859-1", low_memory=False)
# Quick look at the data set



print('Shape of Data - Rows: {}\n Columns: {}'.format(ufo_df.shape[0],ufo_df.shape[1]))
ufo_df.columns.values
ufo_df.dtypes
ufo_df.head(10)
ufo_df.info()

# There are nulls for country,state, shape
ufo_df['datetime'] = pd.to_datetime(ufo_df['datetime'], errors='coerce')


ufo_df['datetime'].head()
ufo_df['year'] = ufo_df['datetime'].dt.year
ufo_df[['datetime', 'year']].head()
ufo_df['year'].isnull().sum()
ufo_df['year'] = ufo_df['year'].fillna(0).astype('int')

ufo_df[['datetime', 'year']].head()
ufo_df['city'] = ufo_df['city'].str.title()

ufo_df['state'] = ufo_df['state'].str.upper()

ufo_df['country'] = ufo_df['country'].str.upper()

ufo_df['latitude'] = pd.to_numeric(ufo_df['latitude'], errors = 'coerce')

ufo_df.rename(columns = {'longitude ':'longitude'}, inplace = True)
ufo_df.head()
# Convert country and state to descriptive name

country = {

	"AF":"AFGHANISTAN",

	"AX":"ALAND ISLANDS",

	"AL":"ALBANIA",

	"DZ":"ALGERIA",

	"AS":"AMERICAN SAMOA",

	"AD":"ANDORRA",

	"AO":"ANGOLA",

	"AI":"ANGUILLA",

	"AQ":"ANTARCTICA",

	"AG":"ANTIGUA AND BARBUDA",

	"AR":"ARGENTINA",

	"AM":"ARMENIA",

	"AW":"ARUBA",

	"AU":"AUSTRALIA",

	"AT":"AUSTRIA",

	"AZ":"AZERBAIJAN",

	"BS":"BAHAMAS",

	"BH":"BAHRAIN",

	"BD":"BANGLADESH",

	"BB":"BARBADOS",

	"BY":"BELARUS",

	"BE":"BELGIUM",

	"BZ":"BELIZE",

	"BJ":"BENIN",

	"BM":"BERMUDA",

	"BT":"BHUTAN",

	"BO":"BOLIVIA, PLURINATIONAL STATE OF",

	"BA":"BOSNIA AND HERZEGOVINA",

	"BW":"BOTSWANA",

	"BV":"BOUVET ISLAND",

	"BR":"BRAZIL",

	"IO":"BRITISH INDIAN OCEAN TERRITORY",

	"BN":"BRUNEI DARUSSALAM",

	"BG":"BULGARIA",

	"BF":"BURKINA FASO",

	"BI":"BURUNDI",

	"KH":"CAMBODIA",

	"CM":"CAMEROON",

	"CA":"CANADA",

	"CV":"CAPE VERDE",

	"KY":"CAYMAN ISLANDS",

	"CF":"CENTRAL AFRICAN REPUBLIC",

	"TD":"CHAD",

	"CL":"CHILE",

	"CN":"CHINA",

	"CX":"CHRISTMAS ISLAND",

	"CC":"COCOS (KEELING) ISLANDS",

	"CO":"COLOMBIA",

	"KM":"COMOROS",

	"CG":"CONGO",

	"CD":"CONGO, THE DEMOCRATIC REPUBLIC OF THE",

	"CK":"COOK ISLANDS",

	"CR":"COSTA RICA",

	"CI":"COTE D'IVOIRE",

	"HR":"CROATIA",

	"CU":"CUBA",

	"CY":"CYPRUS",

	"CZ":"CZECH REPUBLIC",

	"DK":"DENMARK",

	"DJ":"DJIBOUTI",

	"DM":"DOMINICA",

	"DO":"DOMINICAN REPUBLIC",

	"EC":"ECUADOR",

	"EG":"EGYPT",

	"SV":"EL SALVADOR",

	"GQ":"EQUATORIAL GUINEA",

	"ER":"ERITREA",

	"EE":"ESTONIA",

	"ET":"ETHIOPIA",

	"FK":"FALKLAND ISLANDS (MALVINAS)",

	"FO":"FAROE ISLANDS",

	"FJ":"FIJI",

	"FI":"FINLAND",

	"FR":"FRANCE",

	"GF":"FRENCH GUIANA",

	"PF":"FRENCH POLYNESIA",

	"TF":"FRENCH SOUTHERN TERRITORIES",

	"GA":"GABON",

	"GM":"GAMBIA",

	"GE":"GEORGIA",

	"DE":"GERMANY",

	"GH":"GHANA",

	"GI":"GIBRALTAR",

	"GR":"GREECE",

	"GL":"GREENLAND",

	"GD":"GRENADA",

	"GP":"GUADELOUPE",

	"GU":"GUAM",

	"GT":"GUATEMALA",

	"GG":"GUERNSEY",

	"GN":"GUINEA",

	"GW":"GUINEA-BISSAU",

	"GY":"GUYANA",

	"HT":"HAITI",

	"HM":"HEARD ISLAND AND MCDONALD ISLANDS",

	"VA":"HOLY SEE (VATICAN CITY STATE)",

	"HN":"HONDURAS",

	"HK":"HONG KONG",

	"HU":"HUNGARY",

	"IS":"ICELAND",

	"IN":"INDIA",

	"ID":"INDONESIA",

	"IR":"IRAN, ISLAMIC REPUBLIC OF",

	"IQ":"IRAQ",

	"IE":"IRELAND",

	"IM":"ISLE OF MAN",

	"IL":"ISRAEL",

	"IT":"ITALY",

	"JM":"JAMAICA",

	"JP":"JAPAN",

	"JE":"JERSEY",

	"JO":"JORDAN",

	"KZ":"KAZAKHSTAN",

	"KE":"KENYA",

	"KI":"KIRIBATI",

	"KP":"KOREA, DEMOCRATIC PEOPLE'S REPUBLIC OF",

	"KR":"KOREA, REPUBLIC OF",

	"KW":"KUWAIT",

	"KG":"KYRGYZSTAN",

	"LA":"LAO PEOPLE'S DEMOCRATIC REPUBLIC",

	"LV":"LATVIA",

	"LB":"LEBANON",

	"LS":"LESOTHO",

	"LR":"LIBERIA",

	"LY":"LIBYAN ARAB JAMAHIRIYA",

	"LI":"LIECHTENSTEIN",

	"LT":"LITHUANIA",

	"LU":"LUXEMBOURG",

	"MO":"MACAO",

	"MK":"MACEDONIA, THE FORMER YUGOSLAV REPUBLIC OF",

	"MG":"MADAGASCAR",

	"MW":"MALAWI",

	"MY":"MALAYSIA",

	"MV":"MALDIVES",

	"ML":"MALI",

	"MT":"MALTA",

	"MH":"MARSHALL ISLANDS",

	"MQ":"MARTINIQUE",

	"MR":"MAURITANIA",

	"MU":"MAURITIUS",

	"YT":"MAYOTTE",

	"MX":"MEXICO",

	"FM":"MICRONESIA, FEDERATED STATES OF",

	"MD":"MOLDOVA, REPUBLIC OF",

	"MC":"MONACO",

	"MN":"MONGOLIA",

	"ME":"MONTENEGRO",

	"MS":"MONTSERRAT",

	"MA":"MOROCCO",

	"MZ":"MOZAMBIQUE",

	"MM":"MYANMAR",

	"NA":"NAMIBIA",

	"NR":"NAURU",

	"NP":"NEPAL",

	"NL":"NETHERLANDS",

	"AN":"NETHERLANDS ANTILLES",

	"NC":"NEW CALEDONIA",

	"NZ":"NEW ZEALAND",

	"NI":"NICARAGUA",

	"NE":"NIGER",

	"NG":"NIGERIA",

	"NU":"NIUE",

	"NF":"NORFOLK ISLAND",

	"MP":"NORTHERN MARIANA ISLANDS",

	"NO":"NORWAY",

	"OM":"OMAN",

	"PK":"PAKISTAN",

	"PW":"PALAU",

	"PS":"PALESTINIAN TERRITORY, OCCUPIED",

	"PA":"PANAMA",

	"PG":"PAPUA NEW GUINEA",

	"PY":"PARAGUAY",

	"PE":"PERU",

	"PH":"PHILIPPINES",

	"PN":"PITCAIRN",

	"PL":"POLAND",

	"PT":"PORTUGAL",

	"PR":"PUERTO RICO",

	"QA":"QATAR",

	"RE":"REUNION",

	"RO":"ROMANIA",

	"RU":"RUSSIAN FEDERATION",

	"RW":"RWANDA",

	"BL":"SAINT BARTHELEMY",

	"SH":"SAINT HELENA, ASCENSION AND TRISTAN DA CUNHA",

	"KN":"SAINT KITTS AND NEVIS",

	"LC":"SAINT LUCIA",

	"MF":"SAINT MARTIN",

	"PM":"SAINT PIERRE AND MIQUELON",

	"VC":"SAINT VINCENT AND THE GRENADINES",

	"WS":"SAMOA",

	"SM":"SAN MARINO",

	"ST":"SAO TOME AND PRINCIPE",

	"SA":"SAUDI ARABIA",

	"SN":"SENEGAL",

	"RS":"SERBIA",

	"SC":"SEYCHELLES",

	"SL":"SIERRA LEONE",

	"SG":"SINGAPORE",

	"SK":"SLOVAKIA",

	"SI":"SLOVENIA",

	"SB":"SOLOMON ISLANDS",

	"SO":"SOMALIA",

	"ZA":"SOUTH AFRICA",

	"GS":"SOUTH GEORGIA AND THE SOUTH SANDWICH ISLANDS",

	"ES":"SPAIN",

	"LK":"SRI LANKA",

	"SD":"SUDAN",

	"SR":"SURINAME",

	"SJ":"SVALBARD AND JAN MAYEN",

	"SZ":"SWAZILAND",

	"SE":"SWEDEN",

	"CH":"SWITZERLAND",

	"SY":"SYRIAN ARAB REPUBLIC",

	"TW":"TAIWAN, PROVINCE OF CHINA",

	"TJ":"TAJIKISTAN",

	"TZ":"TANZANIA, UNITED REPUBLIC OF",

	"TH":"THAILAND",

	"TL":"TIMOR-LESTE",

	"TG":"TOGO",

	"TK":"TOKELAU",

	"TO":"TONGA",

	"TT":"TRINIDAD AND TOBAGO",

	"TN":"TUNISIA",

	"TR":"TURKEY",

	"TM":"TURKMENISTAN",

	"TC":"TURKS AND CAICOS ISLANDS",

	"TV":"TUVALU",

	"UG":"UGANDA",

	"UA":"UKRAINE",

	"AE":"UNITED ARAB EMIRATES",

	"GB":"UNITED KINGDOM",

	"US":"UNITED STATES",

	"UM":"UNITED STATES MINOR OUTLYING ISLANDS",

	"UY":"URUGUAY",

	"UZ":"UZBEKISTAN",

	"VU":"VANUATU",

	"VE":"VENEZUELA, BOLIVARIAN REPUBLIC OF",

	"VN":"VIET NAM",

	"VG":"VIRGIN ISLANDS, BRITISH",

	"VI":"VIRGIN ISLANDS, U.S.",

	"WF":"WALLIS AND FUTUNA",

	"EH":"WESTERN SAHARA",

	"YE":"YEMEN",

	"ZM":"ZAMBIA",

	"ZW ":"ZIMBABWE"

}
country_df = pd.DataFrame(list(country.items()),columns =['countrycode','country'])
country_df.head()
import pycountry

cc={}

t = list(pycountry.countries)



for country in t:

    cc[country.alpha_2]=country.name



print(cc)
t
states = {

        'AK': 'Alaska',

        'AL': 'Alabama',

        'AR': 'Arkansas',

        'AS': 'American Samoa',

        'AZ': 'Arizona',

        'CA': 'California',

        'CO': 'Colorado',

        'CT': 'Connecticut',

        'DC': 'District of Columbia',

        'DE': 'Delaware',

        'FL': 'Florida',

        'GA': 'Georgia',

        'GU': 'Guam',

        'HI': 'Hawaii',

        'IA': 'Iowa',

        'ID': 'Idaho',

        'IL': 'Illinois',

        'IN': 'Indiana',

        'KS': 'Kansas',

        'KY': 'Kentucky',

        'LA': 'Louisiana',

        'MA': 'Massachusetts',

        'MD': 'Maryland',

        'ME': 'Maine',

        'MI': 'Michigan',

        'MN': 'Minnesota',

        'MO': 'Missouri',

        'MP': 'Northern Mariana Islands',

        'MS': 'Mississippi',

        'MT': 'Montana',

        'NA': 'National',

        'NC': 'North Carolina',

        'ND': 'North Dakota',

        'NE': 'Nebraska',

        'NH': 'New Hampshire',

        'NJ': 'New Jersey',

        'NM': 'New Mexico',

        'NV': 'Nevada',

        'NY': 'New York',

        'OH': 'Ohio',

        'OK': 'Oklahoma',

        'OR': 'Oregon',

        'PA': 'Pennsylvania',

        'PR': 'Puerto Rico',

        'RI': 'Rhode Island',

        'SC': 'South Carolina',

        'SD': 'South Dakota',

        'TN': 'Tennessee',

        'TX': 'Texas',

        'UT': 'Utah',

        'VA': 'Virginia',

        'VI': 'Virgin Islands',

        'VT': 'Vermont',

        'WA': 'Washington',

        'WI': 'Wisconsin',

        'WV': 'West Virginia',

        'WY': 'Wyoming'

}
ufo_df['CountryName'] = ufo_df['country'].map(cc)

ufo_df['StateName'] = ufo_df['state'].map(states)
ufo_df[['country','CountryName','state','StateName']]
# Visualize data

ufo_df.year.value_counts().sort_values(ascending=False)[:10]
sns.countplot(y = 'year', data = ufo_df,order = ufo_df.year.value_counts().iloc[:10].index )
plt.figure(figsize = (15,7))

sns.countplot(x = 'year', data = ufo_df,hue = 'CountryName', 

              order = ufo_df.year.value_counts().iloc[:10].index )



plt.title("Top 10 - UFO Sighting by Year and Country", fontsize = 15)

plt.xlabel("Year", fontsize = 15)

plt.ylabel("Frequenct", fontsize = 15)
# UFO Sightings in USA

ufo_df[(ufo_df['CountryName'] == 'United States')]['StateName'].value_counts()[:10]
# DataFrame for State count and values



df_states = pd.DataFrame(list(zip(ufo_df[(ufo_df['CountryName'] == 'United States')]['StateName'].value_counts().index,

                       ufo_df[(ufo_df['CountryName'] == 'United States')]['StateName'].value_counts())), columns = ['State','Counts'],

                       index = None)
df_states.head()
# Generating the wordcloud with the values under the state dataframe

stcloud = WordCloud().generate(" ".join(df_states['State'].values))
plt.figure(figsize = (20,15))

plt.imshow(stcloud, interpolation='bilinear')

plt.axis('off')
df_states_top20 = df_states[df_states.index <= 20]
# Generating the factorplot for Top20

sns.factorplot(x = "Counts", y = "State", data = df_states_top20, kind = 'bar', 

               size=4.25, aspect=1.9, palette="cubehelix")

plt.title('Factorplot of the States and UFO occurences ')
plt.figure(figsize=(20,10))

sns.countplot(x="shape", data=ufo_df)

plt.xticks(rotation=30)