import json
from functools import partial
import ast

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from matplotlib import collections  as mc
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
hex_df = pd.read_csv('../input/train_simplified/hexagon.csv')
hex_df['drawing'] = hex_df['drawing'].apply(ast.literal_eval)
hex_df.head()
hex_df.loc[0, 'drawing']
def to_line_collection(stroke):
    points = list(zip(stroke[0], [255 - x for x in stroke[1]]))
    lc = [[points[i], points[i + 1]] for i in range(len(points) - 1)]
    return mc.LineCollection(lc, linewidth=2)
def visualise_drawing(drawing, ax):
    for stroke in drawing:
        ax.add_collection(to_line_collection(stroke))
    ax.autoscale()
    return ax
f, ax = plt.subplots(2, 5, figsize=(16, 6))
for i in range(10):
    visualise_drawing(hex_df.loc[i, 'drawing'], ax=ax[i//5, i%5])
    ax[i//5, i%5].axis('off')
plt.show()
def invert_y(strokes):
    strokes[:, 1] = 255 - strokes[:, 1]
    return strokes

def decompose_drawing(drawing):
    strokes = [invert_y(np.array(stroke).T) for stroke in drawing]
    return strokes
def relative_turn_distance(three_points):
    vector_1 = three_points[1, :] - three_points[0, :]
    vector_2 = three_points[2, :] - three_points[1, :]
    distance = np.cross(vector_1, vector_2) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2))
    return distance if not np.isnan(distance) else 0.
def stroke_relative_turn_distance(stroke):
    if stroke.shape[0] < 3:
        distance = 0
    else:
        distance = np.sum([relative_turn_distance(stroke[start: start + 3, :]) for start in range(0, stroke.shape[0] - 2)])
    return distance
def drawing_relative_turn_distance(drawing, connect=False):
    strokes = decompose_drawing(drawing)
    if connect:
        distance = stroke_relative_turn_distance(np.concatenate(strokes, axis=0))
    else:
        distance = np.sum([stroke_relative_turn_distance(stroke) for stroke in strokes])
    return distance
# per_stroke_score = hex_df['drawing'].apply(partial(drawing_relative_turn_distance, connect=False))
per_drawing_score = hex_df['drawing'].apply(partial(drawing_relative_turn_distance, connect=True))
# hex_df['sum_per_stroke_score'] = per_stroke_score
hex_df['score'] = per_drawing_score
hex_df.head()
f, ax = plt.subplots(2, 5, figsize=(16, 6))
ordered_subset = hex_df.sort_values('score').iloc[:5, :]
for i, drawing in enumerate(ordered_subset['drawing']):
    visualise_drawing(drawing, ax=ax[0, i])
    ax[0, i].axis('off')
ordered_subset = hex_df.sort_values('score').iloc[-5:, :]
for i, drawing in enumerate(ordered_subset['drawing']):
    visualise_drawing(drawing, ax=ax[1, i])
    ax[1, i].axis('off')
plt.show()
hex_df.describe()
f, ax = plt.subplots(2, 5, figsize=(16, 6))
temp = hex_df.copy()
temp['dev'] = np.abs(temp['score'] - temp['score'].median())
ordered_subset = temp.sort_values('dev', ascending=True).iloc[:10, :]
for i, drawing in enumerate(ordered_subset['drawing']):
    visualise_drawing(drawing, ax=ax[i//5, i%5])
    ax[i//5, i%5].axis('off')
plt.show()
country_codes = '''
Country Name;ISO 3166-1-alpha-2 code
AFGHANISTAN;AF
ÅLAND ISLANDS;AX
ALBANIA;AL
ALGERIA;DZ
AMERICAN SAMOA;AS
ANDORRA;AD
ANGOLA;AO
ANGUILLA;AI
ANTARCTICA;AQ
ANTIGUA AND BARBUDA;AG
ARGENTINA;AR
ARMENIA;AM
ARUBA;AW
AUSTRALIA;AU
AUSTRIA;AT
AZERBAIJAN;AZ
BAHAMAS;BS
BAHRAIN;BH
BANGLADESH;BD
BARBADOS;BB
BELARUS;BY
BELGIUM;BE
BELIZE;BZ
BENIN;BJ
BERMUDA;BM
BHUTAN;BT
BOLIVIA, PLURINATIONAL STATE OF;BO
BONAIRE, SINT EUSTATIUS AND SABA;BQ
BOSNIA AND HERZEGOVINA;BA
BOTSWANA;BW
BOUVET ISLAND;BV
BRAZIL;BR
BRITISH INDIAN OCEAN TERRITORY;IO
BRUNEI DARUSSALAM;BN
BULGARIA;BG
BURKINA FASO;BF
BURUNDI;BI
CAMBODIA;KH
CAMEROON;CM
CANADA;CA
CAPE VERDE;CV
CAYMAN ISLANDS;KY
CENTRAL AFRICAN REPUBLIC;CF
CHAD;TD
CHILE;CL
CHINA;CN
CHRISTMAS ISLAND;CX
COCOS (KEELING) ISLANDS;CC
COLOMBIA;CO
COMOROS;KM
CONGO;CG
CONGO, THE DEMOCRATIC REPUBLIC OF THE;CD
COOK ISLANDS;CK
COSTA RICA;CR
CÔTE D'IVOIRE;CI
CROATIA;HR
CUBA;CU
CURAÇAO;CW
CYPRUS;CY
CZECH REPUBLIC;CZ
DENMARK;DK
DJIBOUTI;DJ
DOMINICA;DM
DOMINICAN REPUBLIC;DO
ECUADOR;EC
EGYPT;EG
EL SALVADOR;SV
EQUATORIAL GUINEA;GQ
ERITREA;ER
ESTONIA;EE
ETHIOPIA;ET
FALKLAND ISLANDS (MALVINAS);FK
FAROE ISLANDS;FO
FIJI;FJ
FINLAND;FI
FRANCE;FR
FRENCH GUIANA;GF
FRENCH POLYNESIA;PF
FRENCH SOUTHERN TERRITORIES;TF
GABON;GA
GAMBIA;GM
GEORGIA;GE
GERMANY;DE
GHANA;GH
GIBRALTAR;GI
GREECE;GR
GREENLAND;GL
GRENADA;GD
GUADELOUPE;GP
GUAM;GU
GUATEMALA;GT
GUERNSEY;GG
GUINEA;GN
GUINEA-BISSAU;GW
GUYANA;GY
HAITI;HT
HEARD ISLAND AND MCDONALD ISLANDS;HM
HOLY SEE (VATICAN CITY STATE);VA
HONDURAS;HN
HONG KONG;HK
HUNGARY;HU
ICELAND;IS
INDIA;IN
INDONESIA;ID
IRAN, ISLAMIC REPUBLIC OF;IR
IRAQ;IQ
IRELAND;IE
ISLE OF MAN;IM
ISRAEL;IL
ITALY;IT
JAMAICA;JM
JAPAN;JP
JERSEY;JE
JORDAN;JO
KAZAKHSTAN;KZ
KENYA;KE
KIRIBATI;KI
KOREA, DEMOCRATIC PEOPLE'S REPUBLIC OF;KP
KOREA, REPUBLIC OF;KR
KUWAIT;KW
KYRGYZSTAN;KG
LAO PEOPLE'S DEMOCRATIC REPUBLIC;LA
LATVIA;LV
LEBANON;LB
LESOTHO;LS
LIBERIA;LR
LIBYA;LY
LIECHTENSTEIN;LI
LITHUANIA;LT
LUXEMBOURG;LU
MACAO;MO
MACEDONIA, THE FORMER YUGOSLAV REPUBLIC OF;MK
MADAGASCAR;MG
MALAWI;MW
MALAYSIA;MY
MALDIVES;MV
MALI;ML
MALTA;MT
MARSHALL ISLANDS;MH
MARTINIQUE;MQ
MAURITANIA;MR
MAURITIUS;MU
MAYOTTE;YT
MEXICO;MX
MICRONESIA, FEDERATED STATES OF;FM
MOLDOVA, REPUBLIC OF;MD
MONACO;MC
MONGOLIA;MN
MONTENEGRO;ME
MONTSERRAT;MS
MOROCCO;MA
MOZAMBIQUE;MZ
MYANMAR;MM
NAMIBIA;NA
NAURU;NR
NEPAL;NP
NETHERLANDS;NL
NEW CALEDONIA;NC
NEW ZEALAND;NZ
NICARAGUA;NI
NIGER;NE
NIGERIA;NG
NIUE;NU
NORFOLK ISLAND;NF
NORTHERN MARIANA ISLANDS;MP
NORWAY;NO
OMAN;OM
PAKISTAN;PK
PALAU;PW
PALESTINE, STATE OF;PS
PANAMA;PA
PAPUA NEW GUINEA;PG
PARAGUAY;PY
PERU;PE
PHILIPPINES;PH
PITCAIRN;PN
POLAND;PL
PORTUGAL;PT
PUERTO RICO;PR
QATAR;QA
RÉUNION;RE
ROMANIA;RO
RUSSIAN FEDERATION;RU
RWANDA;RW
SAINT BARTHÉLEMY;BL
SAINT HELENA, ASCENSION AND TRISTAN DA CUNHA;SH
SAINT KITTS AND NEVIS;KN
SAINT LUCIA;LC
SAINT MARTIN (FRENCH PART);MF
SAINT PIERRE AND MIQUELON;PM
SAINT VINCENT AND THE GRENADINES;VC
SAMOA;WS
SAN MARINO;SM
SAO TOME AND PRINCIPE;ST
SAUDI ARABIA;SA
SENEGAL;SN
SERBIA;RS
SEYCHELLES;SC
SIERRA LEONE;SL
SINGAPORE;SG
SINT MAARTEN (DUTCH PART);SX
SLOVAKIA;SK
SLOVENIA;SI
SOLOMON ISLANDS;SB
SOMALIA;SO
SOUTH AFRICA;ZA
SOUTH GEORGIA AND THE SOUTH SANDWICH ISLANDS;GS
SOUTH SUDAN;SS
SPAIN;ES
SRI LANKA;LK
SUDAN;SD
SURINAME;SR
SVALBARD AND JAN MAYEN;SJ
SWAZILAND;SZ
SWEDEN;SE
SWITZERLAND;CH
SYRIAN ARAB REPUBLIC;SY
TAIWAN;TW
TAJIKISTAN;TJ
TANZANIA, UNITED REPUBLIC OF;TZ
THAILAND;TH
TIMOR-LESTE;TL
TOGO;TG
TOKELAU;TK
TONGA;TO
TRINIDAD AND TOBAGO;TT
TUNISIA;TN
TURKEY;TR
TURKMENISTAN;TM
TURKS AND CAICOS ISLANDS;TC
TUVALU;TV
UGANDA;UG
UKRAINE;UA
UNITED ARAB EMIRATES;AE
UNITED KINGDOM;GB
UNITED STATES;US
UNITED STATES MINOR OUTLYING ISLANDS;UM
URUGUAY;UY
UZBEKISTAN;UZ
VANUATU;VU
VENEZUELA, BOLIVARIAN REPUBLIC OF;VE
VIET NAM;VN
VIRGIN ISLANDS, BRITISH;VG
VIRGIN ISLANDS, U.S.;VI
WALLIS AND FUTUNA;WF
WESTERN SAHARA;EH
YEMEN;YE
ZAMBIA;ZM
ZIMBABWE;ZW
'''
from io import StringIO
country_codes_df = pd.read_csv(StringIO(country_codes), sep=';')
country_codes_df.columns = ['countryname', 'countrycode']
country_codes_df.head()
hex_df = pd.merge(hex_df, country_codes_df, on='countrycode', how='left')
top_countries = hex_df['countrycode'].value_counts()
f, ax = plt.subplots(figsize=(14, 14))
subset = hex_df[hex_df['countrycode'].isin(top_countries.index[:40]) & (np.abs(hex_df['score'] < 10))]
sub_order = subset.groupby('countryname')['score'].mean().sort_values().index
sns.barplot(data=subset, y='countryname', x='score', order=sub_order)
plt.show()
circle_df = pd.read_csv('../input/train_simplified/circle.csv')
circle_df['drawing'] = circle_df['drawing'].apply(ast.literal_eval)
f, ax = plt.subplots(2, 5, figsize=(16, 6))
for i in range(10):
    visualise_drawing(circle_df.loc[i, 'drawing'], ax=ax[i//5, i%5])
    ax[i//5, i%5].axis('off')
plt.show()
circle_df['score'] = circle_df['drawing'].apply(partial(drawing_relative_turn_distance, connect=True))
circle_df = pd.merge(circle_df, country_codes_df, on='countrycode', how='left')
top_countries = circle_df['countrycode'].value_counts()
f, ax = plt.subplots(figsize=(14, 14))
subset = circle_df[circle_df['countrycode'].isin(top_countries.index[:40]) & (np.abs(circle_df['score'] < 10))]
sub_order = subset.groupby('countryname')['score'].mean().sort_values().index
sns.barplot(data=subset, y='countryname', x='score', order=sub_order)
plt.show()
square_df = pd.read_csv('../input/train_simplified/square.csv')
square_df['drawing'] = square_df['drawing'].apply(ast.literal_eval)

f, ax = plt.subplots(2, 5, figsize=(16, 6))
for i in range(10):
    visualise_drawing(square_df.loc[i, 'drawing'], ax=ax[i//5, i%5])
    ax[i//5, i%5].axis('off')
plt.show()
square_df['score'] = square_df['drawing'].apply(partial(drawing_relative_turn_distance, connect=True))
square_df = pd.merge(square_df, country_codes_df, on='countrycode', how='left')
top_countries = square_df['countrycode'].value_counts()
f, ax = plt.subplots(figsize=(14, 14))
subset = square_df[square_df['countrycode'].isin(top_countries.index[:40]) & (np.abs(square_df['score'] < 10))]
sub_order = subset.groupby('countryname')['score'].mean().sort_values().index
sns.barplot(data=subset, y='countryname', x='score', order=sub_order)
plt.show()
octagon_df = pd.read_csv('../input/train_simplified/octagon.csv')
octagon_df['drawing'] = octagon_df['drawing'].apply(ast.literal_eval)

f, ax = plt.subplots(2, 5, figsize=(16, 6))
for i in range(10):
    visualise_drawing(octagon_df.loc[i, 'drawing'], ax=ax[i//5, i%5])
    ax[i//5, i%5].axis('off')
plt.show()
octagon_df['score'] = octagon_df['drawing'].apply(partial(drawing_relative_turn_distance, connect=False))
octagon_df = pd.merge(octagon_df, country_codes_df, on='countrycode', how='left')
top_countries = octagon_df['countrycode'].value_counts()
f, ax = plt.subplots(figsize=(14, 14))
subset = octagon_df[octagon_df['countrycode'].isin(top_countries.index[:40]) & (np.abs(octagon_df['score'] < 10))]
sub_order = subset.groupby('countryname')['score'].mean().sort_values().index
sns.barplot(data=subset, y='countryname', x='score', order=sub_order)
plt.show()
dfs = [square_df, hex_df, octagon_df, circle_df]
combined = pd.concat(dfs, axis=0, sort=False)
combined['n_segs'] = combined['drawing'].apply(lambda x: np.sum([len(s[0]) for s in x]))
combined['per_seg_score'] = combined['score'] / combined['n_segs']
combined.groupby('word')['n_segs'].describe()
combined.groupby('word')['score'].describe()
f, ax = plt.subplots(figsize=(14, 10))
for word in combined['word'].unique():
    sns.kdeplot(data=combined[(combined['word'] == word) & (np.abs(combined['score']) < 10)]['score'], ax=ax, label=word)
ax.set_xlabel('counterclockwiseness score')
plt.show()
f, ax = plt.subplots(figsize=(14, 10))
for word in combined['word'].unique():
    sns.kdeplot(data=combined[(combined['word'] == word) & (np.abs(combined['per_seg_score']) < 2)]['per_seg_score'], ax=ax, label=word)
ax.set_xlabel('counterclockwiseness score per segment')
plt.show()